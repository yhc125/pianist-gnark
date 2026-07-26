package dlinkzg

import (
	"bytes"
	"encoding/hex"
	"errors"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestOpeningTranscriptDeterministicSchedule(t *testing.T) {
	context := testTranscriptContext()
	u0, u1, u2, u3 := testOpeningMessages()

	firstTranscript, first := runTestOpeningTranscript(t, "", context, u0, u1, u2, u3)
	secondTranscript, second := runTestOpeningTranscript(t, "", context, u0, u1, u2, u3)

	for i := range first {
		if first[i].ID != second[i].ID || first[i].Counter != second[i].Counter || !first[i].Value.Equal(&second[i].Value) {
			t.Fatalf("challenge %d is not deterministic", i)
		}
	}
	if !bytes.Equal(firstTranscript.Bytes(), secondTranscript.Bytes()) {
		t.Fatal("canonical transcript bytes are not deterministic")
	}
	if firstTranscript.Digest() != secondTranscript.Digest() {
		t.Fatal("transcript digests differ")
	}

	for _, index := range []int{0, 1, 4, 5} {
		if first[index].Value.IsZero() {
			t.Fatalf("%s must be sampled from F*", first[index].ID)
		}
	}
	betaForbidden := betaExclusionSet(first[2].Value)
	if fieldElementInSet(first[3].Value, betaForbidden) {
		t.Fatalf("beta lies in its exclusion set (counter %d)", first[3].Counter)
	}
}

func TestOpeningTranscriptRejectionCounterIsCanonicalAndBound(t *testing.T) {
	transcript, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	u0, _, _, _ := testOpeningMessages()
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}

	var firstInField fr.Element
	var firstInFieldCounter uint32
	for {
		candidate, inField := transcript.challengeCandidate(ChallengeXi, firstInFieldCounter)
		if inField {
			firstInField = candidate
			break
		}
		firstInFieldCounter++
	}

	out, err := transcript.sampleChallenge(ChallengeXi, []fr.Element{firstInField})
	if err != nil {
		t.Fatal(err)
	}
	if out.Counter <= firstInFieldCounter {
		t.Fatalf("forbidden candidate at counter %d was not rejected; accepted counter %d", firstInFieldCounter, out.Counter)
	}
	if out.Value.Equal(&firstInField) {
		t.Fatal("sampler returned the explicitly forbidden field element")
	}
	for counter := uint32(0); counter < out.Counter; counter++ {
		candidate, inField := transcript.challengeCandidate(ChallengeXi, counter)
		if inField && !candidate.Equal(&firstInField) {
			t.Fatalf("counter %d was admissible before reported counter %d", counter, out.Counter)
		}
	}

	// ChallengeOut includes the accepted counter, so later challenges bind it.
	left, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	right, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	if err := left.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	if err := right.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	left.appendChallenge(ChallengeOut{ID: ChallengeXi, Value: out.Value, Counter: out.Counter})
	right.appendChallenge(ChallengeOut{ID: ChallengeXi, Value: out.Value, Counter: out.Counter + 1})
	leftNu, err := left.sampleChallenge(ChallengeNu, nil)
	if err != nil {
		t.Fatal(err)
	}
	rightNu, err := right.sampleChallenge(ChallengeNu, nil)
	if err != nil {
		t.Fatal(err)
	}
	if leftNu.Value.Equal(&rightNu.Value) {
		t.Fatal("mutating a rejection counter did not change the next challenge input")
	}
}

func TestOpeningTranscriptBindsContextDomainAndPhasePayload(t *testing.T) {
	context := testTranscriptContext()
	u0, u1, _, _ := testOpeningMessages()

	base := deriveThroughBeta(t, "", context, u0, u1)

	mutatedContext := testTranscriptContext()
	mutatedContext.PublicStatementDigest = NewTranscriptDigest([]byte("statement-B"))
	contextSeparated := deriveThroughBeta(t, "", mutatedContext, u0, u1)
	if base[0].Value.Equal(&contextSeparated[0].Value) {
		t.Fatal("changing the public context did not domain-separate xi")
	}
	mutatedPrefix := testTranscriptContext()
	mutatedPrefix.PriorTranscriptDigest = NewTranscriptDigest([]byte("outer-PIOP-prefix-B"))
	prefixSeparated := deriveThroughBeta(t, "", mutatedPrefix, u0, u1)
	if base[0].Value.Equal(&prefixSeparated[0].Value) {
		t.Fatal("changing the outer transcript prefix did not change xi")
	}

	domainSeparated := deriveThroughBeta(t, "DLinKZG/opening-transcript/test-suite", context, u0, u1)
	if base[0].Value.Equal(&domainSeparated[0].Value) {
		t.Fatal("changing the transcript domain did not domain-separate xi")
	}

	mutatedU0 := u0
	mutatedU0.PartialCommitments[0] = testG1(91)
	u0Separated := deriveThroughBeta(t, "", context, mutatedU0, u1)
	if base[0].Value.Equal(&u0Separated[0].Value) {
		t.Fatal("mutating U0 did not change xi")
	}

	mutatedU1 := u1
	mutatedU1.LinkEvaluations[1].Add(&mutatedU1.LinkEvaluations[1], new(fr.Element).SetOne())
	u1Separated := deriveThroughBeta(t, "", context, u0, mutatedU1)
	for i := 0; i < 3; i++ {
		if !base[i].Value.Equal(&u1Separated[i].Value) {
			t.Fatalf("U1 mutation changed preceding challenge %s", base[i].ID)
		}
	}
	if base[3].Value.Equal(&u1Separated[3].Value) {
		t.Fatal("mutating the U1 phase payload did not change beta")
	}
}

func TestOpeningTranscriptBindsU2AndU3Payloads(t *testing.T) {
	context := testTranscriptContext()
	u0, u1, u2, u3 := testOpeningMessages()
	_, base := runTestOpeningTranscript(t, "", context, u0, u1, u2, u3)

	mutatedU2 := u2
	mutatedU2.BatchAtBeta[2].Add(&mutatedU2.BatchAtBeta[2], new(fr.Element).SetOne())
	_, afterU2Mutation := runTestOpeningTranscript(t, "", context, u0, u1, mutatedU2, u3)
	for i := 0; i < 4; i++ {
		if !base[i].Value.Equal(&afterU2Mutation[i].Value) {
			t.Fatalf("U2 mutation changed preceding challenge %s", base[i].ID)
		}
	}
	if base[4].Value.Equal(&afterU2Mutation[4].Value) {
		t.Fatal("mutating U2 did not change kappa")
	}

	mutatedU3 := u3
	mutatedU3.WL = testG1(101)
	_, afterU3Mutation := runTestOpeningTranscript(t, "", context, u0, u1, u2, mutatedU3)
	for i := 0; i < 5; i++ {
		if !base[i].Value.Equal(&afterU3Mutation[i].Value) {
			t.Fatalf("U3 mutation changed preceding challenge %s", base[i].ID)
		}
	}
	if base[5].Value.Equal(&afterU3Mutation[5].Value) {
		t.Fatal("mutating U3 did not change delta")
	}
}

func TestOpeningTranscriptRejectsOutOfOrderCalls(t *testing.T) {
	transcript, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	_, err = transcript.DeriveXi()
	if !errors.Is(err, ErrTranscriptOrder) {
		t.Fatalf("DeriveXi before U0: got %v, want ErrTranscriptOrder", err)
	}

	u0, u1, _, _ := testOpeningMessages()
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	_, err = transcript.DeriveNu()
	if !errors.Is(err, ErrTranscriptOrder) {
		t.Fatalf("DeriveNu before xi: got %v, want ErrTranscriptOrder", err)
	}
	if err := transcript.AppendU1(u1); !errors.Is(err, ErrTranscriptOrder) {
		t.Fatalf("AppendU1 before U0 challenges: got %v, want ErrTranscriptOrder", err)
	}
}

func TestOpeningTranscriptRejectsMalformedCommitmentBeforeHashing(t *testing.T) {
	transcript, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	before := transcript.Digest()
	u0, _, _, _ := testOpeningMessages()
	u0.PartialCommitments[1].X.SetUint64(1)
	u0.PartialCommitments[1].Y.SetUint64(1)
	if err := transcript.AppendU0(u0); !errors.Is(err, ErrInvalidTranscriptMessage) {
		t.Fatalf("malformed commitment: got %v, want ErrInvalidTranscriptMessage", err)
	}
	if transcript.Digest() != before {
		t.Fatal("rejected commitment mutated the transcript")
	}
}

func TestOpeningTranscriptRejectsMalformedFieldBeforeHashing(t *testing.T) {
	transcript, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	u0, u1, _, _ := testOpeningMessages()
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveXi(); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveNu(); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveZChallenge(); err != nil {
		t.Fatal(err)
	}
	before := transcript.Digest()
	u1.LinkEvaluations[0] = fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	if err := transcript.AppendU1(u1); !errors.Is(err, ErrInvalidTranscriptMessage) {
		t.Fatalf("malformed field element: got %v, want ErrInvalidTranscriptMessage", err)
	}
	if transcript.Digest() != before {
		t.Fatal("rejected field element mutated the transcript")
	}
}

func TestOpeningTranscriptNilReceiverReturnsOrderError(t *testing.T) {
	var transcript *OpeningTranscript
	u0, u1, u2, u3 := testOpeningMessages()
	calls := []struct {
		name string
		call func() error
	}{
		{"AppendU0", func() error { return transcript.AppendU0(u0) }},
		{"DeriveXi", func() error { _, err := transcript.DeriveXi(); return err }},
		{"DeriveNu", func() error { _, err := transcript.DeriveNu(); return err }},
		{"DeriveZChallenge", func() error { _, err := transcript.DeriveZChallenge(); return err }},
		{"AppendU1", func() error { return transcript.AppendU1(u1) }},
		{"DeriveBeta", func() error { _, err := transcript.DeriveBeta(); return err }},
		{"AppendU2", func() error { return transcript.AppendU2(u2) }},
		{"DeriveKappa", func() error { _, err := transcript.DeriveKappa(); return err }},
		{"AppendU3", func() error { return transcript.AppendU3(u3) }},
		{"DeriveDelta", func() error { _, err := transcript.DeriveDelta(); return err }},
	}
	for _, test := range calls {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); !errors.Is(err, ErrTranscriptOrder) {
				t.Fatalf("got %v, want ErrTranscriptOrder", err)
			}
		})
	}
}

func TestOpeningTranscriptContextValidationAndCopy(t *testing.T) {
	invalid := testTranscriptContext()
	invalid.SessionNonce = [32]byte{}
	if _, err := NewOpeningTranscript(invalid); !errors.Is(err, ErrInvalidTranscriptContext) {
		t.Fatalf("empty nonce: got %v, want ErrInvalidTranscriptContext", err)
	}
	if _, err := NewOpeningTranscriptWithDomain("", testTranscriptContext()); !errors.Is(err, ErrInvalidTranscriptContext) {
		t.Fatalf("empty domain: got %v, want ErrInvalidTranscriptContext", err)
	}
	invalid = testTranscriptContext()
	invalid.IndexDigest = TranscriptDigest{}
	if _, err := NewOpeningTranscript(invalid); !errors.Is(err, ErrInvalidTranscriptContext) {
		t.Fatalf("zero index digest: got %v, want ErrInvalidTranscriptContext", err)
	}

	context := testTranscriptContext()
	transcript, err := NewOpeningTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	context.PublicStatementDigest[0] ^= 0xff
	comparison, err := NewOpeningTranscript(testTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	u0, _, _, _ := testOpeningMessages()
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	if err := comparison.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	left, err := transcript.DeriveXi()
	if err != nil {
		t.Fatal(err)
	}
	right, err := comparison.DeriveXi()
	if err != nil {
		t.Fatal(err)
	}
	if !left.Value.Equal(&right.Value) || left.Counter != right.Counter {
		t.Fatal("constructor retained mutable context aliases")
	}
}

func TestOpeningTranscriptGoldenVector(t *testing.T) {
	context := testTranscriptContext()
	u0, u1, u2, u3 := testOpeningMessages()
	transcript, challenges := runTestOpeningTranscript(t, "", context, u0, u1, u2, u3)
	expectedCounters := [6]uint32{2, 0, 6, 1, 9, 4}
	expectedValues := [6]string{
		"296b0d79cfeac9493072fc5e62fe481e673f3fdb32ea44ada3eb1412a74d188d",
		"08633833d9f56385f92b435885491c61164fda8f616e589a11e5b2e2c442a10e",
		"08745dc2ad7c27982e71b7cb2ab561ea40728d94354e1dc117d8997e6979fe6b",
		"0be18123a8f1abf2948de81425c081849f167a41fe67b1698a2c858fb57b8b1b",
		"07d90dd5f36c4145f857d6cb6aad98a714b3f71cc344ec3b6253bba31dd8ad90",
		"1236d3211173813704cafe2e1fc8b109daadc4bba16663887e26f41d1b3fea65",
	}
	for i := range challenges {
		encoded := challenges[i].Value.Bytes()
		if challenges[i].Counter != expectedCounters[i] || hex.EncodeToString(encoded[:]) != expectedValues[i] {
			t.Fatalf("golden %s mismatch: counter=%d value=%s", challenges[i].ID, challenges[i].Counter, hex.EncodeToString(encoded[:]))
		}
	}
	digest := transcript.Digest()
	const expectedDigest = "aca800f702a3b8213ac57b288b4403b105a74764ec7d824b1cb4c0467ccb5998"
	if hex.EncodeToString(digest[:]) != expectedDigest {
		t.Fatalf("golden transcript digest mismatch: %s", hex.EncodeToString(digest[:]))
	}
}

func runTestOpeningTranscript(
	t *testing.T,
	domain string,
	context TranscriptContext,
	u0 U0Message,
	u1 U1Message,
	u2 U2Message,
	u3 U3Message,
) (*OpeningTranscript, [6]ChallengeOut) {
	t.Helper()
	var transcript *OpeningTranscript
	var err error
	if domain == "" {
		transcript, err = NewOpeningTranscript(context)
	} else {
		transcript, err = NewOpeningTranscriptWithDomain(domain, context)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	var challenges [6]ChallengeOut
	if challenges[0], err = transcript.DeriveXi(); err != nil {
		t.Fatal(err)
	}
	if challenges[1], err = transcript.DeriveNu(); err != nil {
		t.Fatal(err)
	}
	if challenges[2], err = transcript.DeriveZChallenge(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU1(u1); err != nil {
		t.Fatal(err)
	}
	if challenges[3], err = transcript.DeriveBeta(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU2(u2); err != nil {
		t.Fatal(err)
	}
	if challenges[4], err = transcript.DeriveKappa(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU3(u3); err != nil {
		t.Fatal(err)
	}
	if challenges[5], err = transcript.DeriveDelta(); err != nil {
		t.Fatal(err)
	}
	return transcript, challenges
}

func deriveThroughBeta(t *testing.T, domain string, context TranscriptContext, u0 U0Message, u1 U1Message) [4]ChallengeOut {
	t.Helper()
	var transcript *OpeningTranscript
	var err error
	if domain == "" {
		transcript, err = NewOpeningTranscript(context)
	} else {
		transcript, err = NewOpeningTranscriptWithDomain(domain, context)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	var challenges [4]ChallengeOut
	if challenges[0], err = transcript.DeriveXi(); err != nil {
		t.Fatal(err)
	}
	if challenges[1], err = transcript.DeriveNu(); err != nil {
		t.Fatal(err)
	}
	if challenges[2], err = transcript.DeriveZChallenge(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU1(u1); err != nil {
		t.Fatal(err)
	}
	if challenges[3], err = transcript.DeriveBeta(); err != nil {
		t.Fatal(err)
	}
	return challenges
}

func betaExclusionSet(zChallenge fr.Element) []fr.Element {
	zero := fr.Element{}
	one := fr.NewElement(1)
	var minusOne fr.Element
	minusOne.Neg(&one)
	result := []fr.Element{zero, one, minusOne, zChallenge}
	if !zChallenge.IsZero() {
		var inverse fr.Element
		inverse.Inverse(&zChallenge)
		result = append(result, inverse)
	}
	return result
}

func testTranscriptContext() TranscriptContext {
	return TranscriptContext{
		ProtocolVersion:       "dlinkzg-test-v1",
		PublicStatementDigest: NewTranscriptDigest([]byte("statement-A")),
		SRSDigest:             NewTranscriptDigest([]byte("srs-A")),
		IndexDigest:           NewTranscriptDigest([]byte("index-A")),
		PartyManifestDigest:   NewTranscriptDigest([]byte("P1,P2,P3")),
		SessionNonce:          [32]byte(NewTranscriptDigest([]byte("session-0001"))),
		PriorTranscriptDigest: NewTranscriptDigest([]byte("outer-PIOP-prefix-A")),
	}
}

func testOpeningMessages() (U0Message, U1Message, U2Message, U3Message) {
	u0 := U0Message{PartialCommitments: [3]bn254.G1Affine{testG1(1), testG1(2), testG1(3)}}
	u1 := U1Message{
		LinkEvaluations:   [3]fr.Element{fr.NewElement(11), fr.NewElement(12), fr.NewElement(13)},
		LinkCommitment:    testG1(4),
		LaurentCommitment: testG1(5),
	}
	u2 := U2Message{
		PartialAtBeta:        [3]fr.Element{fr.NewElement(21), fr.NewElement(22), fr.NewElement(23)},
		PartialAtBetaInverse: [3]fr.Element{fr.NewElement(31), fr.NewElement(32), fr.NewElement(33)},
		BatchAtBeta:          [4]fr.Element{fr.NewElement(41), fr.NewElement(42), fr.NewElement(43), fr.NewElement(44)},
		BatchAtBetaInverse:   [4]fr.Element{fr.NewElement(51), fr.NewElement(52), fr.NewElement(53), fr.NewElement(54)},
	}
	u3 := U3Message{WG: testG1(6), WL: testG1(7), PiZ: testG1(8), PiY: testG1(9)}
	return u0, u1, u2, u3
}

func testG1(scalar int64) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var result bn254.G1Affine
	result.ScalarMultiplication(&generator, big.NewInt(scalar))
	return result
}
