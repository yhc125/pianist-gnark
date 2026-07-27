package dlinkzg

import (
	"fmt"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend"
)

type compiledProtocolTestFixture struct {
	setup     *CompiledSetup
	vk        CompiledVerifyingKey
	statement CompiledStatement
	solution  []fr.Element
	proof     CompiledProof
}

func TestCompiledProtocolRealSparseR1CSM2M4(t *testing.T) {
	expectedSizes := map[int]int{2: 2132, 4: 2324}
	for _, partitions := range []int{2, 4} {
		t.Run(adapterWorldName(partitions), func(t *testing.T) {
			fixture := newCompiledProtocolTestFixture(t, partitions, byte(10+partitions))
			if err := CompiledVerify(fixture.vk, fixture.statement, fixture.proof); err != nil {
				t.Fatalf("verify honest proof: %v", err)
			}

			encoded, err := fixture.proof.MarshalBinary(uint64(partitions))
			if err != nil {
				t.Fatalf("marshal proof: %v", err)
			}
			wantSize, err := CompiledProofEncodedSize(uint64(partitions))
			if err != nil {
				t.Fatalf("proof size: %v", err)
			}
			if len(encoded) != wantSize || wantSize != expectedSizes[partitions] {
				t.Fatalf("encoded size = %d, helper = %d, want %d", len(encoded), wantSize, expectedSizes[partitions])
			}
			decoded, err := DecodeCompiledProof(encoded, uint64(partitions))
			if err != nil {
				t.Fatalf("decode proof: %v", err)
			}
			if err := CompiledVerify(fixture.vk, fixture.statement, *decoded); err != nil {
				t.Fatalf("verify decoded proof: %v", err)
			}
			g1, fields, err := CompiledProofElementCounts(uint64(partitions))
			if err != nil {
				t.Fatal(err)
			}
			if g1 != 17 || fields != 6*(bitsForCompiledProtocol(partitions))+43 {
				t.Fatalf("proof inventory = %d G1, %d Fr", g1, fields)
			}
		})
	}
}

func TestCompiledProtocolSetupReuseAcrossTwoWitnesses(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 4, 31)
	metadataBefore := fixture.setup.Metadata

	const rounds = 16
	x := fr.NewElement(3)
	y := x
	for round := 0; round < rounds; round++ {
		y.Mul(&y, &x)
	}
	secondWitness := localPIOPConcreteWitness(t, &localPIOPAdapterCircuit{
		X: x, Y: y, rounds: rounds,
	})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	secondSolution, err := fixture.setup.Parties[0].ConstraintSystem.Solve(secondWitness, proverConfig)
	if err != nil {
		t.Fatalf("solve second witness: %v", err)
	}
	secondStatement := compiledProtocolStatementForTest(fixture.setup, secondSolution, 32)
	secondProof, err := CompiledProve(fixture.setup, secondStatement, secondSolution)
	if err != nil {
		t.Fatalf("prove second witness: %v", err)
	}
	if err := CompiledVerify(fixture.vk, secondStatement, secondProof); err != nil {
		t.Fatalf("verify second witness: %v", err)
	}
	if fixture.setup.Metadata != metadataBefore {
		t.Fatal("online proving mutated reusable setup metadata")
	}
	if fixture.statement.OuterContext.PublicStatementDigest == secondStatement.OuterContext.PublicStatementDigest {
		t.Fatal("distinct public witnesses produced the same statement digest")
	}
}

func TestCompiledProtocolRejectsEveryPublicPhaseTamper(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 2, 41)
	one := fr.One()
	pointIncrement := compiledProofTestG1(997)
	tests := []struct {
		name   string
		mutate func(*CompiledProof)
	}{
		{"W0", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.W0.WitnessCommitments[0], &pointIncrement)
		}},
		{"W1", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.W1.AccumulatorCommitment, &pointIncrement)
		}},
		{"W2", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.W2.QuotientCommitments[1], &pointIncrement)
		}},
		{"ProductCheck", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.ProductCheck.Commitments[0], &pointIncrement)
		}},
		{"SumCheck", func(proof *CompiledProof) {
			proof.SumCheckRounds[0].Coefficients[0].Add(&proof.SumCheckRounds[0].Coefficients[0], &one)
		}},
		{"FinalCircuit", func(proof *CompiledProof) {
			proof.FinalEvaluations.Terminal[0].Add(&proof.FinalEvaluations.Terminal[0], &one)
		}},
		{"FinalTree", func(proof *CompiledProof) {
			proof.FinalEvaluations.ProductCheck[0].Add(&proof.FinalEvaluations.ProductCheck[0], &one)
		}},
		{"U0", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.U0.PartialCommitments[0], &pointIncrement)
		}},
		{"U1", func(proof *CompiledProof) {
			proof.U1.LinkEvaluations[0].Add(&proof.U1.LinkEvaluations[0], &one)
		}},
		{"U2", func(proof *CompiledProof) {
			proof.U2.PartialAtBeta[0].Add(&proof.U2.PartialAtBeta[0], &one)
		}},
		{"U3", func(proof *CompiledProof) {
			sourceCommitmentAdd(&proof.U3.WN, &pointIncrement)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tampered := cloneCompiledProtocolProof(fixture.proof)
			test.mutate(&tampered)
			if err := CompiledVerify(fixture.vk, fixture.statement, tampered); err == nil {
				t.Fatal("tampered proof was accepted")
			}
		})
	}
}

func TestCompiledProtocolPublicStatementAndContextBinding(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 2, 51)
	one := fr.One()

	unchangedDigest := cloneCompiledProtocolStatement(fixture.statement)
	unchangedDigest.PublicSolutionPrefix[0].Add(&unchangedDigest.PublicSolutionPrefix[0], &one)
	if err := CompiledVerify(fixture.vk, unchangedDigest, fixture.proof); err == nil {
		t.Fatal("changed public prefix with stale digest was accepted")
	}

	reboundDigest := cloneCompiledProtocolStatement(fixture.statement)
	reboundDigest.PublicSolutionPrefix[0].Add(&reboundDigest.PublicSolutionPrefix[0], &one)
	reboundDigest.OuterContext.PublicStatementDigest = CompiledPublicStatementDigest(reboundDigest.PublicSolutionPrefix)
	if err := CompiledVerify(fixture.vk, reboundDigest, fixture.proof); err == nil {
		t.Fatal("changed public prefix and rebound digest reused the old proof")
	}

	contextTamper := cloneCompiledProtocolStatement(fixture.statement)
	contextTamper.OuterContext.SessionNonce[0] ^= 0x80
	if err := CompiledVerify(fixture.vk, contextTamper, fixture.proof); err == nil {
		t.Fatal("changed session context reused the old proof")
	}

	metadataTamper := cloneCompiledProtocolStatement(fixture.statement)
	metadataTamper.OuterContext.IndexDigest[0] ^= 0x01
	if err := CompiledVerify(fixture.vk, metadataTamper, fixture.proof); err == nil {
		t.Fatal("changed setup context was accepted")
	}

	solutionMismatch := cloneCompiledProtocolStatement(fixture.statement)
	solutionMismatch.PublicSolutionPrefix[0].Add(&solutionMismatch.PublicSolutionPrefix[0], &one)
	solutionMismatch.OuterContext.PublicStatementDigest = CompiledPublicStatementDigest(solutionMismatch.PublicSolutionPrefix)
	if _, err := CompiledProve(fixture.setup, solutionMismatch, fixture.solution); err == nil {
		t.Fatal("prover accepted a solution/public-prefix mismatch")
	}
}

func TestCompiledVerifierKeyIsIndependentOfPartyAndCoordinatorRoles(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 4, 61)
	fixture.setup.Parties = nil
	fixture.setup.Coordinator = CompiledCoordinatorSetup{}
	if err := CompiledVerify(fixture.vk, fixture.statement, fixture.proof); err != nil {
		t.Fatalf("verifier depended on removed prover/coordinator roles: %v", err)
	}

	tamperedVK := fixture.vk
	tamperedVK.Metadata.SetupDigest[0] ^= 0x01
	if err := CompiledVerify(tamperedVK, fixture.statement, fixture.proof); err == nil {
		t.Fatal("tampered verifier-key metadata was accepted")
	}
}

func TestCompiledProverRejectsConstraintSystemMutationAfterSetup(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 2, 71)
	if len(fixture.setup.Parties[0].ConstraintSystem.Coefficients) == 0 {
		t.Fatal("fixture has no coefficient to mutate")
	}
	one := fr.One()
	fixture.setup.Parties[0].ConstraintSystem.Coefficients[0].Add(
		&fixture.setup.Parties[0].ConstraintSystem.Coefficients[0], &one,
	)
	if _, err := CompiledProve(fixture.setup, fixture.statement, fixture.solution); err == nil {
		t.Fatal("prover accepted a constraint system mutated after authenticated setup")
	}
}

func TestCompiledPublicStatementDigestBindsLengthAndOrder(t *testing.T) {
	a := fr.NewElement(1)
	b := fr.NewElement(2)
	if CompiledPublicStatementDigest([]fr.Element{a, b}) == CompiledPublicStatementDigest([]fr.Element{b, a}) {
		t.Fatal("statement digest does not bind order")
	}
	if CompiledPublicStatementDigest([]fr.Element{a}) == CompiledPublicStatementDigest([]fr.Element{a, {}}) {
		t.Fatal("statement digest does not bind length")
	}
}

func newCompiledProtocolTestFixture(t *testing.T, partitions int, nonceByte byte) compiledProtocolTestFixture {
	t.Helper()
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}
	setup, err := NewDeterministicCompiledSetup(
		spr,
		partitions,
		fr.NewElement(uint64(1009+partitions)),
		fr.NewElement(uint64(1019+partitions)),
	)
	if err != nil {
		t.Fatalf("compiled setup: %v", err)
	}
	statement := compiledProtocolStatementForTest(setup, solution, nonceByte)
	proof, err := CompiledProve(setup, statement, solution)
	if err != nil {
		t.Fatalf("compiled prove: %v", err)
	}
	vk, err := setup.VerifyingKey()
	if err != nil {
		t.Fatalf("verifying key: %v", err)
	}
	return compiledProtocolTestFixture{
		setup: setup, vk: vk, statement: statement,
		solution: append([]fr.Element(nil), solution...), proof: proof,
	}
}

func compiledProtocolStatementForTest(setup *CompiledSetup, solution []fr.Element, nonceByte byte) CompiledStatement {
	publicVariables := setup.Coordinator.PublicInputPlacement.PublicVariables()
	prefix := append([]fr.Element(nil), solution[:publicVariables]...)
	metadata := setup.Metadata
	var nonce [32]byte
	nonce[0] = nonceByte
	nonce[31] = nonceByte ^ 0xa5
	context := OuterTranscriptContext{
		ProtocolVersion:       "dlinkzg-compiled-protocol-test/v1",
		PublicStatementDigest: CompiledPublicStatementDigest(prefix),
		SRSDigest:             metadata.SRSDigest,
		IndexDigest:           metadata.IndexDigest,
		IndexPrefixDigest:     metadata.IndexPrefixDigest,
		ShiftCounter:          metadata.IndexShift.Counter,
		Shift:                 metadata.IndexShift.Sigma,
		PartyManifestDigest:   metadata.PartyManifestDigest,
		SessionNonce:          nonce,
		PartitionCount:        uint64(metadata.PartitionCount),
		LocalDomainSize:       uint64(metadata.LocalDomainSize),
		LocalDomainGenerator:  metadata.LocalDomainGenerator,
	}
	return CompiledStatement{PublicSolutionPrefix: prefix, OuterContext: context}
}

func cloneCompiledProtocolProof(proof CompiledProof) CompiledProof {
	result := proof
	result.SumCheckRounds = append([]OuterSumCheckRoundMessage(nil), proof.SumCheckRounds...)
	return result
}

func cloneCompiledProtocolStatement(statement CompiledStatement) CompiledStatement {
	result := statement
	result.PublicSolutionPrefix = append([]fr.Element(nil), statement.PublicSolutionPrefix...)
	return result
}

func bitsForCompiledProtocol(value int) int {
	result := 0
	for value > 1 {
		value >>= 1
		result++
	}
	return result
}

func ExampleCompiledPublicStatementDigest() {
	prefix := []fr.Element{fr.NewElement(3), fr.NewElement(9)}
	digest := CompiledPublicStatementDigest(prefix)
	fmt.Println(digest != (TranscriptDigest{}))
	// Output: true
}
