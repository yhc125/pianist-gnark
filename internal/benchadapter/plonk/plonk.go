// Package plonk provides the single-process PLONK baseline used by the
// distributed-protocol benchmark driver.
package plonk

import (
	"bytes"
	"errors"
	"fmt"
	"math/big"
	"runtime"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	bn254kzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/kzg"
	gnarkbackend "github.com/consensys/gnark/backend"
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	csbn254 "github.com/consensys/gnark/internal/backend/bn254/cs"
	plonkbn254 "github.com/consensys/gnark/internal/backend/bn254/plonk"
	witnessbn254 "github.com/consensys/gnark/internal/backend/bn254/witness"
)

// Config selects the synthetic multiplication instance. Constraints is the
// requested number of sparse constraints and Seed deterministically selects
// the witness.
type Config struct {
	Constraints int
	Seed        uint64
}

// Timings separates the reusable universal SRS construction from PLONK's
// circuit-specific setup. Production benchmarks may load an existing SRS and
// omit SRS from the reported online/preprocessing times.
type Timings struct {
	Compile       time.Duration
	SRS           time.Duration
	Setup         time.Duration
	WitnessSolve  time.Duration
	ProveCrypto   time.Duration
	ProveEndToEnd time.Duration
	Verify        time.Duration
}

// Result contains backend-independent scalar measurements so cmd packages can
// translate them directly into their own JSON schema.
type Result struct {
	Timings              Timings
	ProofBytes           int64
	HeapAllocBytes       uint64
	HeapSysBytes         uint64
	RequestedConstraints int
	RealizedConstraints  int
	PublicVariables      int
	DomainSize           uint64
	Accepted             bool
}

type syntheticMulCircuit struct {
	X frontend.Variable
	Y frontend.Variable `gnark:",public"`

	// steps is compile-time metadata and is intentionally excluded from the
	// witness schema. The final equality contributes one more constraint.
	steps int
}

func (c *syntheticMulCircuit) Define(api frontend.API) error {
	acc := c.X
	for i := 0; i < c.steps; i++ {
		acc = api.Mul(acc, c.X)
	}
	api.AssertIsEqual(acc, c.Y)
	return nil
}

func assignments(cfg Config) (*syntheticMulCircuit, *syntheticMulCircuit) {
	steps := cfg.Constraints - 1

	var x fr.Element
	x.SetUint64(cfg.Seed%251 + 2)
	y := x
	for i := 0; i < steps; i++ {
		y.Mul(&y, &x)
	}

	return &syntheticMulCircuit{steps: steps}, &syntheticMulCircuit{
		X:     x,
		Y:     y,
		steps: steps,
	}
}

// Run compiles, sets up, proves, and verifies one BN254 PLONK instance. The
// SRS is generated deterministically for benchmarking only; it is not a
// production trusted setup.
func Run(cfg Config) (Result, error) {
	if cfg.Constraints < 2 {
		return Result{}, errors.New("constraints must be at least 2")
	}

	circuit, witnessAssignment := assignments(cfg)
	result := Result{RequestedConstraints: cfg.Constraints}

	started := time.Now()
	ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, circuit)
	result.Timings.Compile = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("compile circuit: %w", err)
	}
	result.RealizedConstraints = ccs.GetNbConstraints()

	fullWitness, err := frontend.NewWitness(witnessAssignment, ecc.BN254)
	if err != nil {
		return Result{}, fmt.Errorf("build full witness: %w", err)
	}
	publicWitness, err := frontend.NewWitness(
		witnessAssignment,
		ecc.BN254,
		frontend.PublicOnly(),
	)
	if err != nil {
		return Result{}, fmt.Errorf("build public witness: %w", err)
	}
	spr, ok := ccs.(*csbn254.SparseR1CS)
	if !ok {
		return Result{}, fmt.Errorf("unexpected constraint system type %T", ccs)
	}
	witnessVector, ok := fullWitness.Vector.(*witnessbn254.Witness)
	if !ok {
		return Result{}, fmt.Errorf("unexpected witness vector type %T", fullWitness.Vector)
	}

	// PLONK needs enough powers for the constraint system plus its public-input
	// placeholder constraints. This follows gnark's test.NewKZGSRS sizing rule.
	_, _, publicVariables := ccs.GetNbVariables()
	result.PublicVariables = publicVariables
	srsSize := ecc.NextPowerOfTwo(uint64(ccs.GetNbConstraints()+publicVariables)) + 3
	result.DomainSize = srsSize - 3
	started = time.Now()
	srs, err := bn254kzg.NewSRS(srsSize, big.NewInt(42))
	result.Timings.SRS = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("build benchmark SRS: %w", err)
	}

	started = time.Now()
	pk, vk, err := gnarkplonk.Setup(ccs, srs)
	result.Timings.Setup = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("plonk setup: %w", err)
	}
	concretePK, ok := pk.(*plonkbn254.ProvingKey)
	if !ok {
		return Result{}, fmt.Errorf("unexpected plonk proving key type %T", pk)
	}
	proverConfig, err := gnarkbackend.NewProverConfig()
	if err != nil {
		return Result{}, fmt.Errorf("build plonk prover config: %w", err)
	}

	started = time.Now()
	solution, err := plonkbn254.Solve(spr, *witnessVector, proverConfig)
	result.Timings.WitnessSolve = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("plonk witness solve: %w", err)
	}

	started = time.Now()
	proof, err := plonkbn254.ProveWithSolution(spr, concretePK, solution)
	result.Timings.ProveCrypto = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("plonk prove: %w", err)
	}
	result.Timings.ProveEndToEnd = result.Timings.WitnessSolve + result.Timings.ProveCrypto

	var encoded bytes.Buffer
	written, err := proof.WriteTo(&encoded)
	if err != nil {
		return Result{}, fmt.Errorf("serialize plonk proof: %w", err)
	}
	if written != int64(encoded.Len()) {
		return Result{}, fmt.Errorf(
			"serialize plonk proof: writer reported %d bytes, buffer contains %d",
			written,
			encoded.Len(),
		)
	}
	result.ProofBytes = written

	started = time.Now()
	err = gnarkplonk.Verify(proof, vk, publicWitness)
	result.Timings.Verify = time.Since(started)
	if err != nil {
		return Result{}, fmt.Errorf("plonk verify: %w", err)
	}
	result.Accepted = true

	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	result.HeapAllocBytes = memory.Alloc
	result.HeapSysBytes = memory.HeapSys

	return result, nil
}
