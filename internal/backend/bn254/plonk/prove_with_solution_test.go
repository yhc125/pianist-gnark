package plonk_test

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	bn254kzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/kzg"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
	bn254plonk "github.com/consensys/gnark/internal/backend/bn254/plonk"
	bn254witness "github.com/consensys/gnark/internal/backend/bn254/witness"
)

type splitProveCircuit struct {
	X frontend.Variable
	Y frontend.Variable `gnark:",public"`
}

func (c *splitProveCircuit) Define(api frontend.API) error {
	api.AssertIsEqual(api.Mul(c.X, c.X), c.Y)
	return nil
}

func TestProveWithSolutionVerifies(t *testing.T) {
	ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, &splitProveCircuit{})
	if err != nil {
		t.Fatalf("compile circuit: %v", err)
	}
	spr := ccs.(*cs.SparseR1CS)
	assignment := &splitProveCircuit{X: 3, Y: 9}
	full, err := frontend.NewWitness(assignment, ecc.BN254)
	if err != nil {
		t.Fatalf("build full witness: %v", err)
	}
	fullBN254 := full.Vector.(*bn254witness.Witness)
	public, err := frontend.NewWitness(assignment, ecc.BN254, frontend.PublicOnly())
	if err != nil {
		t.Fatalf("build public witness: %v", err)
	}
	publicBN254 := public.Vector.(*bn254witness.Witness)

	_, _, publicVariables := ccs.GetNbVariables()
	srsSize := ecc.NextPowerOfTwo(uint64(ccs.GetNbConstraints()+publicVariables)) + 3
	srs, err := bn254kzg.NewSRS(srsSize, big.NewInt(42))
	if err != nil {
		t.Fatalf("build SRS: %v", err)
	}
	pk, vk, err := bn254plonk.Setup(spr, srs)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}

	solution, err := bn254plonk.Solve(spr, *fullBN254, backend.ProverConfig{})
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	proof, err := bn254plonk.ProveWithSolution(spr, pk, solution)
	if err != nil {
		t.Fatalf("prove with solution: %v", err)
	}
	if err := bn254plonk.Verify(proof, vk, *publicBN254); err != nil {
		t.Fatalf("verify split proof: %v", err)
	}
}
