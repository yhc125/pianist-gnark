package main

import (
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkbackend "github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	csbn254 "github.com/consensys/gnark/internal/backend/bn254/cs"
	dlinkzgbackend "github.com/consensys/gnark/internal/backend/bn254/dlinkzg"
	witnessbn254 "github.com/consensys/gnark/internal/backend/bn254/witness"
	resourceadapter "github.com/consensys/gnark/internal/benchadapter/resources"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

func validTestConfig(backend string, workers uint64) config {
	return config{
		backend: backend, curve: "bn254", circuit: "synthetic-mul",
		constraints: 4096, workers: workers, seed: 7, threads: 1,
	}
}

func TestValidateConfigTopology(t *testing.T) {
	previousRank, previousWorld := mpi.SelfRank, mpi.WorldSize
	t.Cleanup(func() {
		mpi.SelfRank = previousRank
		mpi.WorldSize = previousWorld
	})

	tests := []struct {
		name      string
		cfg       config
		worldSize uint64
		wantError bool
	}{
		{"gpiano M2 world2", validTestConfig("gpiano", 2), 2, false},
		{"gpiano mismatch", validTestConfig("gpiano", 2), 3, true},
		{"dlinkzg M2 world3", validTestConfig("dlinkzg", 2), 3, false},
		{"dlinkzg M4 world5", validTestConfig("dlinkzg", 4), 5, false},
		{"dlinkzg missing coordinator", validTestConfig("dlinkzg", 2), 2, true},
		{"dlinkzg non power of two", validTestConfig("dlinkzg", 3), 4, true},
		{"dlinkzg M1", validTestConfig("dlinkzg", 1), 2, true},
		{"plonk singleton", validTestConfig("plonk", 1), 1, false},
		{"plonk distributed", validTestConfig("plonk", 1), 2, true},
		{"unknown backend", validTestConfig("unknown", 1), 1, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mpi.SelfRank = 0
			mpi.WorldSize = test.worldSize
			err := validateConfig(test.cfg)
			if (err != nil) != test.wantError {
				t.Fatalf("validateConfig() error = %v, wantError=%v", err, test.wantError)
			}
		})
	}
}

func TestInitializeMPIWorldSingleProcessAndMissingDistributedEnvironment(t *testing.T) {
	previousArgs := os.Args
	previousRank, previousWorld := mpi.SelfRank, mpi.WorldSize
	t.Cleanup(func() {
		os.Args = previousArgs
		mpi.SelfRank = previousRank
		mpi.WorldSize = previousWorld
	})

	os.Args = []string{"compare"}
	t.Setenv("PIANIST_MPI_DISABLE", "1")
	mpi.SelfRank = 99
	mpi.WorldSize = 99
	if err := initializeMPIWorld(validTestConfig("plonk", 1)); err != nil {
		t.Fatalf("initialize singleton: %v", err)
	}
	if mpi.SelfRank != 0 || mpi.WorldSize != 1 {
		t.Fatalf("singleton MPI identity = rank %d/world %d", mpi.SelfRank, mpi.WorldSize)
	}

	os.Args = []string{"compare", "Slave"}
	if err := initializeMPIWorld(validTestConfig("plonk", 1)); err == nil {
		t.Fatal("single-process worker invocation was accepted")
	}

	os.Args = []string{"compare"}
	t.Setenv("PIANIST_MPI_DISABLE", "")
	t.Setenv("PIANIST_MPI_IP_FILE", "")
	t.Setenv("PIANIST_MPI_SSH_KEY", "")
	t.Setenv("PIANIST_MPI_SSH_USER", "")
	if err := initializeMPIWorld(validTestConfig("dlinkzg", 2)); err == nil {
		t.Fatal("distributed initialization without PIANIST_MPI_* was accepted")
	}

	// The legacy Pianist DKZG package normally initializes simpleMPI before
	// main. Dummy paths prove that initializeMPIWorld does not try to launch a
	// second root or worker when the runtime already has a nonzero world.
	t.Setenv("PIANIST_MPI_IP_FILE", "/does/not/exist/ip.txt")
	t.Setenv("PIANIST_MPI_SSH_KEY", "/does/not/exist/key")
	t.Setenv("PIANIST_MPI_SSH_USER", "nobody")
	mpi.SelfRank = 0
	mpi.WorldSize = 3
	if err := initializeMPIWorld(validTestConfig("dlinkzg", 2)); err != nil {
		t.Fatalf("reuse initialized root world: %v", err)
	}
	os.Args = []string{"compare", "127.0.0.1", "9999", "Slave"}
	mpi.SelfRank = 1
	if err := initializeMPIWorld(validTestConfig("dlinkzg", 2)); err != nil {
		t.Fatalf("reuse initialized worker world: %v", err)
	}
}

func testTranscriptDigest(tag byte) dlinkzgbackend.TranscriptDigest {
	var digest dlinkzgbackend.TranscriptDigest
	for index := range digest {
		digest[index] = tag + byte(index)
	}
	return digest
}

func TestBuildDLinKZGStatementCopiesPublicPrefixAndBindsMetadata(t *testing.T) {
	metadata := dlinkzgbackend.CompiledSetupMetadata{
		SRSDigest:            testTranscriptDigest(1),
		IndexDigest:          testTranscriptDigest(2),
		IndexPrefixDigest:    testTranscriptDigest(3),
		PartyManifestDigest:  testTranscriptDigest(4),
		PartitionCount:       4,
		LocalDomainSize:      16,
		PublicVariableCount:  2,
		LocalDomainGenerator: fr.NewElement(17),
		IndexShift: dlinkzgbackend.IndexShift{
			Counter: 9,
			Sigma:   fr.NewElement(19),
		},
	}
	publicValues := []fr.Element{fr.NewElement(3), fr.NewElement(11)}
	cfg := validTestConfig("dlinkzg", 4)
	statement, err := buildDLinKZGStatement(metadata, publicValues, cfg)
	if err != nil {
		t.Fatal(err)
	}
	context := statement.OuterContext
	if context.ProtocolVersion != dlinkzgCompareProtocolVersion ||
		context.PublicStatementDigest != dlinkzgbackend.CompiledPublicStatementDigest(statement.PublicSolutionPrefix) ||
		context.SRSDigest != metadata.SRSDigest || context.IndexDigest != metadata.IndexDigest ||
		context.IndexPrefixDigest != metadata.IndexPrefixDigest ||
		context.PartyManifestDigest != metadata.PartyManifestDigest ||
		context.ShiftCounter != metadata.IndexShift.Counter ||
		!context.Shift.Equal(&metadata.IndexShift.Sigma) ||
		context.PartitionCount != uint64(metadata.PartitionCount) ||
		context.LocalDomainSize != uint64(metadata.LocalDomainSize) ||
		!context.LocalDomainGenerator.Equal(&metadata.LocalDomainGenerator) ||
		context.SessionNonce == ([32]byte{}) {
		t.Fatalf("statement context did not bind setup metadata: %+v", context)
	}

	originalFirst := statement.PublicSolutionPrefix[0]
	publicValues[0].SetUint64(99)
	if !statement.PublicSolutionPrefix[0].Equal(&originalFirst) {
		t.Fatal("statement retained the caller's public-prefix backing slice")
	}
	repeated, err := buildDLinKZGStatement(metadata, []fr.Element{fr.NewElement(3), fr.NewElement(11)}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if repeated.OuterContext.SessionNonce != context.SessionNonce {
		t.Fatal("same benchmark statement produced a different deterministic nonce")
	}
	differentSeed := cfg
	differentSeed.seed++
	rebound, err := buildDLinKZGStatement(metadata, statement.PublicSolutionPrefix, differentSeed)
	if err != nil {
		t.Fatal(err)
	}
	if rebound.OuterContext.SessionNonce == context.SessionNonce {
		t.Fatal("session nonce did not bind the benchmark seed")
	}
	if _, err := buildDLinKZGStatement(metadata, statement.PublicSolutionPrefix[:1], cfg); err == nil {
		t.Fatal("public-prefix length mismatch was accepted")
	}
}

func TestDLinKZGPublicOnlyWitnessIsSolutionPrefixAndSetupIsFullyValidated(t *testing.T) {
	cfg := validTestConfig("dlinkzg", 2)
	cfg.constraints = 16
	circuit, witnessAssignment := assignment(cfg, cfg.constraints-1)
	ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, circuit)
	if err != nil {
		t.Fatal(err)
	}
	spr, ok := ccs.(*csbn254.SparseR1CS)
	if !ok {
		t.Fatalf("unexpected constraint system type %T", ccs)
	}
	fullWitness, err := frontend.NewWitness(witnessAssignment, ecc.BN254)
	if err != nil {
		t.Fatal(err)
	}
	publicWitness, err := frontend.NewWitness(witnessAssignment, ecc.BN254, frontend.PublicOnly())
	if err != nil {
		t.Fatal(err)
	}
	fullVector, ok := fullWitness.Vector.(*witnessbn254.Witness)
	if !ok {
		t.Fatalf("unexpected full witness type %T", fullWitness.Vector)
	}
	publicVector, ok := publicWitness.Vector.(*witnessbn254.Witness)
	if !ok {
		t.Fatalf("unexpected public witness type %T", publicWitness.Vector)
	}
	proverConfig, err := gnarkbackend.NewProverConfig()
	if err != nil {
		t.Fatal(err)
	}
	solution, err := spr.Solve([]fr.Element(*fullVector), proverConfig)
	if err != nil {
		t.Fatal(err)
	}
	if len(*publicVector) != int(spr.NbPublicVariables) {
		t.Fatalf("public witness length = %d, SparseR1CS public variables = %d", len(*publicVector), spr.NbPublicVariables)
	}
	for index := range *publicVector {
		if !(*publicVector)[index].Equal(&solution[index]) {
			t.Fatalf("public witness value %d is not solution prefix", index)
		}
	}

	setup, err := dlinkzgbackend.NewDeterministicCompiledSetup(
		spr, 2, fr.NewElement(1009), fr.NewElement(1019),
	)
	if err != nil {
		t.Fatal(err)
	}
	vk, rootRole, err := validateAndProvisionDLinKZGSetup(setup, 0)
	if err != nil {
		t.Fatalf("validate and provision root: %v", err)
	}
	if rootRole.Coordinator == nil || rootRole.Party != nil || vk.Metadata != setup.Metadata {
		t.Fatal("root role provisioning returned the wrong role or VK")
	}
	_, partyRole, err := validateAndProvisionDLinKZGSetup(setup, 1)
	if err != nil {
		t.Fatalf("validate and provision party: %v", err)
	}
	if partyRole.Party == nil || partyRole.Coordinator != nil || partyRole.Party.Rank != 0 {
		t.Fatal("party role provisioning returned the wrong slot")
	}

	// The cheap online check validates only the SRS shape. Mutating a point
	// while retaining all lengths must still be rejected by our full setup
	// authentication boundary before role extraction.
	setup.Parties[0].RowSRS.G1Row[0] = setup.Parties[0].RowSRS.G1Row[1]
	if _, _, err := validateAndProvisionDLinKZGSetup(setup, 0); err == nil {
		t.Fatal("full setup validation accepted an SRS point mutation")
	}
}

func TestExpectedDLinKZGRootAccountingM2M4(t *testing.T) {
	tests := []struct {
		partitions                           uint64
		payloadSent, payloadRecv             uint64
		wireSent, wireRecv                   uint64
		framingSent, framingRecv             float64
		fieldElements, encoded, uncompressed float64
	}{
		{2, 4416, 2880, 4776, 3200, 360, 320, 49, 2164, 2740},
		{4, 9600, 5760, 10320, 6400, 720, 640, 55, 2356, 2932},
	}
	for _, test := range tests {
		payloadSent, payloadRecv, wireSent, wireRecv, err :=
			expectedDLinKZGRootAccounting(test.partitions)
		if err != nil {
			t.Fatal(err)
		}
		if payloadSent != test.payloadSent || payloadRecv != test.payloadRecv ||
			wireSent != test.wireSent || wireRecv != test.wireRecv {
			t.Fatalf(
				"M=%d accounting = payload %d/%d wire %d/%d",
				test.partitions, payloadSent, payloadRecv, wireSent, wireRecv,
			)
		}
		accounting := dlinkzgbackend.ProtocolMPIAccounting{
			Rank: 0, WorldSize: test.partitions + 1,
			PartitionCount: test.partitions, Operations: dlinkzgProtocolOperations,
			Total: dlinkzgbackend.ProtocolMPIPhaseAccounting{
				PayloadBytesSent: payloadSent, PayloadBytesRecv: payloadRecv,
				WireBytesSent: wireSent, WireBytesRecv: wireRecv,
			},
		}
		metrics, err := dlinkzgCommunicationMetrics(accounting, test.partitions)
		if err != nil {
			t.Fatal(err)
		}
		if metrics["prove_framing_bytes_sent"] != test.framingSent ||
			metrics["prove_framing_bytes_received"] != test.framingRecv ||
			metrics["bytes_sent"] != float64(payloadSent) ||
			metrics["bytes_received"] != float64(payloadRecv) {
			t.Fatalf("M=%d communication metrics = %+v", test.partitions, metrics)
		}

		artifacts, err := dlinkzgArtifactMetrics(test.partitions, int(test.encoded))
		if err != nil {
			t.Fatal(err)
		}
		if artifacts["proof_g1_elements"] != 18 ||
			artifacts["proof_field_elements"] != test.fieldElements ||
			artifacts["proof_bytes"] != test.encoded ||
			artifacts["proof_bytes_uncompressed"] != test.uncompressed ||
			artifacts["proof_encoding_overhead_bytes"] != 20 {
			t.Fatalf("M=%d artifact metrics = %+v", test.partitions, artifacts)
		}
	}
	if _, _, _, _, err := expectedDLinKZGRootAccounting(3); err == nil {
		t.Fatal("non-power-of-two accounting request was accepted")
	}
	if _, err := dlinkzgArtifactMetrics(2, 1); err == nil {
		t.Fatal("wrong encoded proof length was accepted")
	}
}

func TestDLinKZGCommunicationMetricsRejectsNoncanonicalLedger(t *testing.T) {
	payloadSent, payloadRecv, wireSent, wireRecv, err := expectedDLinKZGRootAccounting(2)
	if err != nil {
		t.Fatal(err)
	}
	accounting := dlinkzgbackend.ProtocolMPIAccounting{
		Rank: 0, WorldSize: 3, PartitionCount: 2, Operations: dlinkzgProtocolOperations,
		Total: dlinkzgbackend.ProtocolMPIPhaseAccounting{
			PayloadBytesSent: payloadSent, PayloadBytesRecv: payloadRecv,
			WireBytesSent: wireSent, WireBytesRecv: wireRecv,
		},
	}
	accounting.Operations--
	if _, err := dlinkzgCommunicationMetrics(accounting, 2); err == nil {
		t.Fatal("wrong operation count was accepted")
	}
	accounting.Operations = dlinkzgProtocolOperations
	accounting.Total.WireBytesSent++
	if _, err := dlinkzgCommunicationMetrics(accounting, 2); err == nil {
		t.Fatal("wrong wire byte count was accepted")
	}
}

func TestResourceSampleEncoding(t *testing.T) {
	want := resourceadapter.Sample{PeakRSSSupported: true, PeakRSSBytes: 1234567}
	got, err := decodeResourceSample(encodeResourceSample(want))
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("resource sample = %+v, want %+v", got, want)
	}
	if _, err := decodeResourceSample(make([]byte, resourceSampleBytes-1)); err == nil {
		t.Fatal("truncated resource sample was accepted")
	}
	malformed := make([]byte, resourceSampleBytes)
	malformed[0] = 2
	if _, err := decodeResourceSample(malformed); err == nil {
		t.Fatal("invalid resource support flag was accepted")
	}
}
