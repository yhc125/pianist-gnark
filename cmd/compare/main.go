package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkbackend "github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/backend/gpiano"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	csbn254 "github.com/consensys/gnark/internal/backend/bn254/cs"
	gpianobn254 "github.com/consensys/gnark/internal/backend/bn254/gpiano"
	witnessbn254 "github.com/consensys/gnark/internal/backend/bn254/witness"
	plonkadapter "github.com/consensys/gnark/internal/benchadapter/plonk"
	resourceadapter "github.com/consensys/gnark/internal/benchadapter/resources"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

type config struct {
	backend     string
	curve       string
	circuit     string
	constraints int
	workers     uint64
	seed        uint64
	jsonOutput  bool
}

type syntheticMulCircuit struct {
	X frontend.Variable
	Y frontend.Variable `gnark:",public"`

	// steps is compile-time circuit metadata and is intentionally not part of
	// the witness schema. Each step contributes one multiplication constraint.
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

type adapterResult struct {
	Timings       map[string]float64 `json:"timings_ms"`
	Communication map[string]float64 `json:"communication,omitempty"`
	Artifacts     map[string]float64 `json:"artifacts,omitempty"`
	Resources     map[string]float64 `json:"resources,omitempty"`
	Operations    map[string]float64 `json:"operations,omitempty"`
	Verification  verification       `json:"verification"`
}

type verification struct {
	Accepted bool `json:"accepted"`
}

func milliseconds(elapsed time.Duration) float64 {
	return float64(elapsed) / float64(time.Millisecond)
}

func boolMetric(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

// synchronizeWorld is benchmark scaffolding, not part of either proof. It
// makes the root wall clock include the slowest rank at every measured phase.
// Its one-byte messages are deliberately excluded from the communication
// snapshots below.
func synchronizeWorld() error {
	if mpi.WorldSize == 1 {
		return nil
	}
	const byteCount = 1
	if mpi.SelfRank == 0 {
		for rank := uint64(1); rank < mpi.WorldSize; rank++ {
			if _, err := mpi.ReceiveBytes(byteCount, rank); err != nil {
				return err
			}
		}
		for rank := uint64(1); rank < mpi.WorldSize; rank++ {
			if err := mpi.SendBytes([]byte{0}, rank); err != nil {
				return err
			}
		}
		return nil
	}
	if err := mpi.SendBytes([]byte{0}, 0); err != nil {
		return err
	}
	_, err := mpi.ReceiveBytes(byteCount, 0)
	return err
}

func gpianoProofInventory(proof gpiano.Proof) (g1Elements, fieldElements int, err error) {
	concrete, ok := proof.(*gpianobn254.Proof)
	if !ok {
		return 0, 0, fmt.Errorf("unexpected gpiano proof type %T", proof)
	}
	g1Elements = len(concrete.LRO) + 1 + 1 + len(concrete.Hx) + len(concrete.Hy)
	g1Elements += 1 + len(concrete.PartialBatchedProof.ClaimedDigests)
	g1Elements += 2 // PartialZShiftedProof.H and ClaimedDigest.
	g1Elements += 1 // BatchedProof.H.
	g1Elements += 1 // WShiftedProof.H.
	fieldElements = len(concrete.BatchedProof.ClaimedValues) + 1
	return g1Elements, fieldElements, nil
}

const resourceSampleBytes = 9

func encodeResourceSample(sample resourceadapter.Sample) []byte {
	encoded := make([]byte, resourceSampleBytes)
	if sample.PeakRSSSupported {
		encoded[0] = 1
	}
	binary.LittleEndian.PutUint64(encoded[1:], sample.PeakRSSBytes)
	return encoded
}

func decodeResourceSample(encoded []byte) (resourceadapter.Sample, error) {
	if len(encoded) != resourceSampleBytes || encoded[0] > 1 {
		return resourceadapter.Sample{}, errors.New("invalid benchmark resource sample")
	}
	return resourceadapter.Sample{
		PeakRSSSupported: encoded[0] == 1,
		PeakRSSBytes:     binary.LittleEndian.Uint64(encoded[1:]),
	}, nil
}

func sendWorkerResourceSample() error {
	sample, err := resourceadapter.Read()
	if err != nil {
		return err
	}
	return mpi.SendBytes(encodeResourceSample(sample), 0)
}

func collectRootResourceSamples() (coordinator, workerMax, rankSum uint64, supportedRanks int, err error) {
	local, err := resourceadapter.Read()
	if err != nil {
		return 0, 0, 0, 0, err
	}
	coordinator = local.PeakRSSBytes
	rankSum = local.PeakRSSBytes
	if local.PeakRSSSupported {
		supportedRanks++
	}
	for rank := uint64(1); rank < mpi.WorldSize; rank++ {
		encoded, receiveErr := mpi.ReceiveBytes(resourceSampleBytes, rank)
		if receiveErr != nil {
			return 0, 0, 0, 0, receiveErr
		}
		sample, decodeErr := decodeResourceSample(encoded)
		if decodeErr != nil {
			return 0, 0, 0, 0, decodeErr
		}
		if sample.PeakRSSSupported {
			supportedRanks++
		}
		if sample.PeakRSSBytes > workerMax {
			workerMax = sample.PeakRSSBytes
		}
		rankSum += sample.PeakRSSBytes
	}
	return coordinator, workerMax, rankSum, supportedRanks, nil
}

func parseConfig() config {
	var cfg config
	flag.StringVar(&cfg.backend, "backend", "gpiano", "backend: gpiano, dlinkzg, or plonk")
	flag.StringVar(&cfg.curve, "curve", "bn254", "curve (currently bn254)")
	flag.StringVar(&cfg.circuit, "circuit", "synthetic-mul", "benchmark circuit")
	flag.IntVar(&cfg.constraints, "constraints", 4096, "target number of constraints")
	flag.Uint64Var(&cfg.workers, "workers", 1, "expected worker process count")
	flag.Uint64Var(&cfg.seed, "seed", 1, "deterministic witness seed")
	flag.BoolVar(&cfg.jsonOutput, "json", false, "emit the adapter JSON contract")
	flag.Parse()
	return cfg
}

func validateConfig(cfg config) error {
	if cfg.curve != "bn254" {
		return fmt.Errorf("unsupported curve %q", cfg.curve)
	}
	if cfg.circuit != "synthetic-mul" {
		return fmt.Errorf("unsupported circuit %q", cfg.circuit)
	}
	if cfg.constraints < 2 {
		return errors.New("constraints must be at least 2")
	}
	if cfg.workers == 0 {
		return errors.New("workers must be positive")
	}
	if (cfg.backend == "gpiano" || cfg.backend == "dlinkzg") && cfg.workers != mpi.WorldSize {
		return fmt.Errorf("--workers=%d does not match MPI world size %d", cfg.workers, mpi.WorldSize)
	}
	if cfg.backend == "plonk" && (cfg.workers != 1 || mpi.WorldSize != 1) {
		return fmt.Errorf("plonk requires --workers=1 and a single-process MPI world; got workers=%d world=%d", cfg.workers, mpi.WorldSize)
	}
	return nil
}

func assignment(cfg config, steps int) (*syntheticMulCircuit, *syntheticMulCircuit) {
	var x fr.Element
	x.SetUint64(cfg.seed%251 + 2)
	y := x
	for i := 0; i < steps; i++ {
		y.Mul(&y, &x)
	}

	circuit := &syntheticMulCircuit{steps: steps}
	witness := &syntheticMulCircuit{X: x, Y: y, steps: steps}
	return circuit, witness
}

func runGPiano(cfg config) (adapterResult, error) {
	// AssertIsEqual contributes the final constraint, so this construction
	// realizes the requested count exactly with the current sparse builder.
	steps := cfg.constraints - 1
	circuit, witnessAssignment := assignment(cfg, steps)

	compileStarted := time.Now()
	ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, circuit)
	if err != nil {
		return adapterResult{}, fmt.Errorf("compile circuit: %w", err)
	}
	compileElapsed := time.Since(compileStarted)

	fullWitness, err := frontend.NewWitness(witnessAssignment, ecc.BN254)
	if err != nil {
		return adapterResult{}, fmt.Errorf("build full witness: %w", err)
	}
	publicWitness, err := frontend.NewWitness(
		witnessAssignment,
		ecc.BN254,
		frontend.PublicOnly(),
	)
	if err != nil {
		return adapterResult{}, fmt.Errorf("build public witness: %w", err)
	}
	spr, ok := ccs.(*csbn254.SparseR1CS)
	if !ok {
		return adapterResult{}, fmt.Errorf("unexpected constraint system type %T", ccs)
	}
	witnessVector, ok := fullWitness.Vector.(*witnessbn254.Witness)
	if !ok {
		return adapterResult{}, fmt.Errorf("unexpected witness vector type %T", fullWitness.Vector)
	}

	if err := synchronizeWorld(); err != nil {
		return adapterResult{}, fmt.Errorf("synchronize before gpiano setup: %w", err)
	}
	setupBytesSentBefore := mpi.BytesSent
	setupBytesReceivedBefore := mpi.BytesReceived

	setupStarted := time.Now()
	pk, vk, err := gpiano.Setup(ccs, publicWitness)
	if err != nil {
		return adapterResult{}, fmt.Errorf("gpiano setup: %w", err)
	}
	setupBytesSent := mpi.BytesSent - setupBytesSentBefore
	setupBytesReceived := mpi.BytesReceived - setupBytesReceivedBefore
	if err := synchronizeWorld(); err != nil {
		return adapterResult{}, fmt.Errorf("synchronize after gpiano setup: %w", err)
	}
	setupElapsed := time.Since(setupStarted)
	concretePK, ok := pk.(*gpianobn254.ProvingKey)
	if !ok {
		return adapterResult{}, fmt.Errorf("unexpected gpiano proving key type %T", pk)
	}
	proverConfig, err := gnarkbackend.NewProverConfig()
	if err != nil {
		return adapterResult{}, fmt.Errorf("build gpiano prover config: %w", err)
	}

	solveStarted := time.Now()
	solution, err := gpianobn254.Solve(spr, *witnessVector, proverConfig)
	if err != nil {
		return adapterResult{}, fmt.Errorf("gpiano witness solve: %w", err)
	}
	if err := synchronizeWorld(); err != nil {
		return adapterResult{}, fmt.Errorf("synchronize after gpiano witness solve: %w", err)
	}
	solveElapsed := time.Since(solveStarted)

	proveBytesSentBefore := mpi.BytesSent
	proveBytesReceivedBefore := mpi.BytesReceived

	proveStarted := time.Now()
	proof, err := gpianobn254.ProveWithSolution(spr, concretePK, solution)
	if err != nil {
		return adapterResult{}, fmt.Errorf("gpiano prove: %w", err)
	}
	proveBytesSent := mpi.BytesSent - proveBytesSentBefore
	proveBytesReceived := mpi.BytesReceived - proveBytesReceivedBefore
	if err := synchronizeWorld(); err != nil {
		return adapterResult{}, fmt.Errorf("synchronize after gpiano prove: %w", err)
	}
	proveCryptoElapsed := time.Since(proveStarted)
	proveEndToEndElapsed := solveElapsed + proveCryptoElapsed

	// Only the root receives the complete Y-direction proof and verifies it.
	if mpi.SelfRank != 0 {
		if err := sendWorkerResourceSample(); err != nil {
			return adapterResult{}, fmt.Errorf("send worker resource sample: %w", err)
		}
		return adapterResult{}, nil
	}

	verifyStarted := time.Now()
	err = gpiano.Verify(proof, vk, publicWitness)
	verifyElapsed := time.Since(verifyStarted)
	if err != nil {
		return adapterResult{}, fmt.Errorf("gpiano verify: %w", err)
	}
	coordinatorPeakRSS, workerPeakRSS, rankSumPeakRSS, supportedRSSRanks, err := collectRootResourceSamples()
	if err != nil {
		return adapterResult{}, fmt.Errorf("collect resource samples: %w", err)
	}
	g1Elements, fieldElements, err := gpianoProofInventory(proof)
	if err != nil {
		return adapterResult{}, err
	}
	var encodedProof bytes.Buffer
	compressedProofBytes, err := proof.WriteTo(&encodedProof)
	if err != nil {
		return adapterResult{}, fmt.Errorf("serialize gpiano proof: %w", err)
	}
	if compressedProofBytes != int64(encodedProof.Len()) {
		return adapterResult{}, fmt.Errorf(
			"serialize gpiano proof: writer reported %d bytes, buffer contains %d",
			compressedProofBytes,
			encodedProof.Len(),
		)
	}
	compressedPayloadBytes := g1Elements*bn254.SizeOfG1AffineCompressed + fieldElements*fr.Bytes
	encodingOverheadBytes := int(compressedProofBytes) - compressedPayloadBytes
	uncompressedProofBytes := g1Elements*bn254.SizeOfG1AffineUncompressed + fieldElements*fr.Bytes + encodingOverheadBytes

	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	_, _, publicVariables := ccs.GetNbVariables()
	localDomainSize := concretePK.Domain[0].Cardinality
	result := adapterResult{
		Timings: map[string]float64{
			"compile":             milliseconds(compileElapsed),
			"setup":               milliseconds(setupElapsed),
			"setup_including_srs": milliseconds(setupElapsed),
			"witness_solve":       milliseconds(solveElapsed),
			"prove":               milliseconds(proveCryptoElapsed),
			"prove_crypto_wall":   milliseconds(proveCryptoElapsed),
			"prove_end_to_end":    milliseconds(proveEndToEndElapsed),
			"verify":              milliseconds(verifyElapsed),
		},
		Communication: map[string]float64{
			"bytes_sent":                               float64(setupBytesSent + proveBytesSent),
			"bytes_received":                           float64(setupBytesReceived + proveBytesReceived),
			"setup_application_payload_bytes_sent":     float64(setupBytesSent),
			"setup_application_payload_bytes_received": float64(setupBytesReceived),
			"prove_application_payload_bytes_sent":     float64(proveBytesSent),
			"prove_application_payload_bytes_received": float64(proveBytesReceived),
			"benchmark_sync_barriers":                  4,
		},
		Artifacts: map[string]float64{
			// proof_bytes is the actual canonical WriteTo encoding. The raw size
			// applies the same framing to gnark-crypto's uncompressed G1 form.
			"proof_bytes":                   float64(compressedProofBytes),
			"proof_bytes_compressed":        float64(compressedProofBytes),
			"proof_bytes_uncompressed":      float64(uncompressedProofBytes),
			"proof_encoding_overhead_bytes": float64(encodingOverheadBytes),
			"proof_g1_elements":             float64(g1Elements),
			"proof_field_elements":          float64(fieldElements),
		},
		Resources: map[string]float64{
			"heap_alloc_bytes":              float64(memory.Alloc),
			"heap_sys_bytes":                float64(memory.HeapSys),
			"peak_rss_coordinator_bytes":    float64(coordinatorPeakRSS),
			"peak_rss_worker_max_bytes":     float64(workerPeakRSS),
			"peak_rss_rank_sum_bytes":       float64(rankSumPeakRSS),
			"peak_rss_supported_rank_count": float64(supportedRSSRanks),
		},
		Operations: map[string]float64{
			"requested_constraints": float64(cfg.constraints),
			"realized_constraints":  float64(ccs.GetNbConstraints()),
			"public_variables":      float64(publicVariables),
			"partitions":            float64(mpi.WorldSize),
			"local_domain_size":     float64(localDomainSize),
			"global_padded_rows":    float64(localDomainSize * mpi.WorldSize),
			"quotient_domain_size":  float64(concretePK.Domain[1].Cardinality),
		},
		Verification: verification{Accepted: true},
	}
	return result, nil
}

func runPLONK(cfg config) (adapterResult, error) {
	measured, err := plonkadapter.Run(plonkadapter.Config{
		Constraints: cfg.constraints,
		Seed:        cfg.seed,
	})
	if err != nil {
		return adapterResult{}, err
	}
	resourceSample, err := resourceadapter.Read()
	if err != nil {
		return adapterResult{}, err
	}
	const (
		plonkG1Elements    = 9
		plonkFieldElements = 8
	)
	plonkPayloadBytes := plonkG1Elements*bn254.SizeOfG1AffineCompressed + plonkFieldElements*fr.Bytes
	plonkEncodingOverhead := int(measured.ProofBytes) - plonkPayloadBytes
	plonkUncompressedBytes := plonkG1Elements*bn254.SizeOfG1AffineUncompressed + plonkFieldElements*fr.Bytes + plonkEncodingOverhead
	setupIncludingSRS := measured.Timings.SRS + measured.Timings.Setup
	return adapterResult{
		Timings: map[string]float64{
			"compile":             milliseconds(measured.Timings.Compile),
			"srs":                 milliseconds(measured.Timings.SRS),
			"setup":               milliseconds(measured.Timings.Setup),
			"setup_including_srs": milliseconds(setupIncludingSRS),
			"witness_solve":       milliseconds(measured.Timings.WitnessSolve),
			"prove":               milliseconds(measured.Timings.ProveCrypto),
			"prove_crypto_wall":   milliseconds(measured.Timings.ProveCrypto),
			"prove_end_to_end":    milliseconds(measured.Timings.ProveEndToEnd),
			"verify":              milliseconds(measured.Timings.Verify),
		},
		Artifacts: map[string]float64{
			"proof_bytes":                   float64(measured.ProofBytes),
			"proof_bytes_compressed":        float64(measured.ProofBytes),
			"proof_bytes_uncompressed":      float64(plonkUncompressedBytes),
			"proof_encoding_overhead_bytes": float64(plonkEncodingOverhead),
			"proof_g1_elements":             plonkG1Elements,
			"proof_field_elements":          plonkFieldElements,
		},
		Resources: map[string]float64{
			"heap_alloc_bytes":              float64(measured.HeapAllocBytes),
			"heap_sys_bytes":                float64(measured.HeapSysBytes),
			"peak_rss_coordinator_bytes":    float64(resourceSample.PeakRSSBytes),
			"peak_rss_worker_max_bytes":     0,
			"peak_rss_rank_sum_bytes":       float64(resourceSample.PeakRSSBytes),
			"peak_rss_supported_rank_count": boolMetric(resourceSample.PeakRSSSupported),
		},
		Operations: map[string]float64{
			"requested_constraints": float64(measured.RequestedConstraints),
			"realized_constraints":  float64(measured.RealizedConstraints),
			"public_variables":      float64(measured.PublicVariables),
			"partitions":            1,
			"local_domain_size":     float64(measured.DomainSize),
			"global_padded_rows":    float64(measured.DomainSize),
		},
		Verification: verification{Accepted: measured.Accepted},
	}, nil
}

func run(cfg config) (adapterResult, error) {
	if err := validateConfig(cfg); err != nil {
		return adapterResult{}, err
	}
	switch cfg.backend {
	case "gpiano":
		return runGPiano(cfg)
	case "plonk":
		return runPLONK(cfg)
	case "dlinkzg":
		return adapterResult{}, fmt.Errorf("backend %q is not wired yet", cfg.backend)
	default:
		return adapterResult{}, fmt.Errorf("unknown backend %q", cfg.backend)
	}
}

func main() {
	cfg := parseConfig()
	result, err := run(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if mpi.SelfRank != 0 {
		return
	}
	if !cfg.jsonOutput {
		fmt.Printf("%s accepted: prove=%.3fms verify=%.3fms\n", cfg.backend,
			result.Timings["prove"], result.Timings["verify"])
		return
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
