package dlinkzg

import (
	"bytes"
	"encoding/hex"
	"errors"
	"math/big"
	"math/bits"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func TestOuterTranscriptDeterministicScheduleAndOpeningComposition(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	firstTranscript, first := runTestOuterTranscript(t, context, messages)
	secondTranscript, second := runTestOuterTranscript(t, context, messages)

	assertOuterChallengeSlicesEqual(t, first.ordered(), second.ordered())
	if !bytes.Equal(firstTranscript.Bytes(), secondTranscript.Bytes()) {
		t.Fatal("canonical outer transcript bytes are not deterministic")
	}
	if firstTranscript.Digest() != secondTranscript.Digest() {
		t.Fatal("outer transcript digests differ")
	}

	openingContext, err := firstTranscript.OpeningContext()
	if err != nil {
		t.Fatal(err)
	}
	prior, err := firstTranscript.PriorTranscriptDigest()
	if err != nil {
		t.Fatal(err)
	}
	if openingContext.PriorTranscriptDigest != prior {
		t.Fatal("opening context did not carry the completed outer digest")
	}
	opening, err := NewOpeningTranscript(openingContext)
	if err != nil {
		t.Fatalf("compose opening transcript: %v", err)
	}
	u0, _, _, _ := testOpeningMessages()
	if err := opening.AppendU0(u0); err != nil {
		t.Fatal(err)
	}
	if _, err := opening.DeriveXi(); err != nil {
		t.Fatalf("derive opening xi from composed prior digest: %v", err)
	}
}

func TestOuterTranscriptRetainedW3BoundaryAddsNoPublicBytes(t *testing.T) {
	transcript := deriveOuterThroughAlpha(t, testOuterTranscriptContext(), testOuterTranscriptMessages())
	before := transcript.Bytes()
	if err := transcript.MarkRetainedW3Complete(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, transcript.Bytes()) {
		t.Fatal("accountability-only W3 transition changed the accepted public transcript")
	}
	if OuterTranscriptBoundaryNotice == "" {
		t.Fatal("outer transcript boundary notice must remain explicit")
	}
}

func TestOuterTranscriptRejectsOutOfOrderCalls(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveInitialChallenges(); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("initial challenges before W0: got %v", err)
	}
	if err := transcript.AppendW0(messages.W0); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW1(messages.W1); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("W1 before initial challenges: got %v", err)
	}
	if _, err := transcript.DeriveInitialChallenges(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW1(messages.W1); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveLambda(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW2(messages.W2); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveAlpha(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendProductCheckCommitments(messages.ProductCheck); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("ProductCheck commitments before W3 boundary: got %v", err)
	}
	if err := transcript.MarkRetainedW3Complete(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendProductCheckCommitments(messages.ProductCheck); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendSumCheckRound(messages.Rounds[0]); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("SumCheck round before zeta/theta: got %v", err)
	}
	if _, err := transcript.DeriveZetaTheta(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendSumCheckRound(messages.Rounds[0]); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendSumCheckRound(messages.Rounds[1]); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("next round before r_0: got %v", err)
	}
	if err := transcript.AppendFinalEvaluations(messages.Final); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("final evaluations before r_0: got %v", err)
	}
	if _, err := transcript.DeriveSumCheckChallenge(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendSumCheckRound(messages.Rounds[1]); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveSumCheckChallenge(); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.PriorTranscriptDigest(); !errors.Is(err, ErrOuterTranscriptIncomplete) {
		t.Fatalf("prior digest before y_final/mu: got %v", err)
	}
	if err := transcript.AppendFinalEvaluations(messages.Final); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.PriorTranscriptDigest(); !errors.Is(err, ErrOuterTranscriptIncomplete) {
		t.Fatalf("prior digest before mu: got %v", err)
	}
	if _, err := transcript.DeriveSourceCompressionChallenges(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.MarkRetainedW3Complete(); !errors.Is(err, ErrOuterTranscriptOrder) {
		t.Fatalf("repeated W3 transition: got %v", err)
	}
}

func TestOuterTranscriptTamperPropagation(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	_, base := runTestOuterTranscript(t, context, messages)

	contextTamper := context
	contextTamper.IndexPrefixDigest = NewTranscriptDigest([]byte("outer-index-prefix-B"))
	shift, err := DeriveIndexShift(
		contextTamper.IndexPrefixDigest,
		contextTamper.LocalDomainSize,
		contextTamper.LocalDomainGenerator,
	)
	if err != nil {
		t.Fatal(err)
	}
	contextTamper.ShiftCounter = shift.Counter
	contextTamper.Shift = shift.Sigma
	_, changedContext := runTestOuterTranscript(t, contextTamper, messages)
	assertOuterChallengeChanged(t, base.Initial.EtaPart, changedContext.Initial.EtaPart, "context -> eta_part")

	w0Tamper := testOuterTranscriptMessages()
	w0Tamper.W0.WitnessCommitments[0] = testG1(901)
	_, changedW0 := runTestOuterTranscript(t, context, w0Tamper)
	assertOuterChallengeChanged(t, base.Initial.EtaPart, changedW0.Initial.EtaPart, "W0 -> eta_part")

	productTamper := testOuterTranscriptMessages()
	productTamper.ProductCheck.Commitments[1] = testG1(902)
	_, changedProduct := runTestOuterTranscript(t, context, productTamper)
	assertOuterChallengeEqual(t, base.Alpha, changedProduct.Alpha, "ProductCheck must not change alpha")
	assertOuterChallengeChanged(t, base.ZetaTheta.Zeta, changedProduct.ZetaTheta.Zeta, "ProductCheck -> zeta")

	roundTamper := testOuterTranscriptMessages()
	roundTamper.Rounds[0].Coefficients[3].Add(&roundTamper.Rounds[0].Coefficients[3], outerFieldPointer(1))
	_, changedRound := runTestOuterTranscript(t, context, roundTamper)
	assertOuterChallengeEqual(t, base.ZetaTheta.Zeta, changedRound.ZetaTheta.Zeta, "round 0 must not change zeta")
	assertOuterChallengeChanged(t, base.SumCheck[0], changedRound.SumCheck[0], "round 0 -> r_0")

	finalTamper := testOuterTranscriptMessages()
	finalTamper.Final.ProductCheck[2].Add(&finalTamper.Final.ProductCheck[2], outerFieldPointer(1))
	_, changedFinal := runTestOuterTranscript(t, context, finalTamper)
	assertOuterChallengeEqual(t, base.SumCheck[1], changedFinal.SumCheck[1], "y_final must not change r_1")
	assertOuterChallengeChanged(t, base.Mu[0], changedFinal.Mu[0], "y_final -> mu_0")
}

func TestOuterTranscriptDomainAndCounterRecordsAreBound(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	standard, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	domainSeparated, err := NewOuterTranscriptWithDomain("DLinKZG/outer-test-suite", context)
	if err != nil {
		t.Fatal(err)
	}
	for _, transcript := range []*OuterTranscript{standard, domainSeparated} {
		if err := transcript.AppendW0(messages.W0); err != nil {
			t.Fatal(err)
		}
	}
	standardInitial, err := standard.DeriveInitialChallenges()
	if err != nil {
		t.Fatal(err)
	}
	domainInitial, err := domainSeparated.DeriveInitialChallenges()
	if err != nil {
		t.Fatal(err)
	}
	assertOuterChallengeChanged(t, standardInitial.EtaPart, domainInitial.EtaPart, "outer suite domain")

	left, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	right, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	for _, transcript := range []*OuterTranscript{left, right} {
		if err := transcript.AppendW0(messages.W0); err != nil {
			t.Fatal(err)
		}
		if _, err := transcript.DeriveInitialChallenges(); err != nil {
			t.Fatal(err)
		}
		if err := transcript.AppendW1(messages.W1); err != nil {
			t.Fatal(err)
		}
	}
	lambda, err := left.DeriveLambda()
	if err != nil {
		t.Fatal(err)
	}
	right.appendChallengeRecord("OuterChallenge", ChallengeOut{
		ID: lambda.ID, Value: lambda.Value, Counter: lambda.Counter + 1,
	})
	right.next = outerStepW2
	for _, transcript := range []*OuterTranscript{left, right} {
		if err := transcript.AppendW2(messages.W2); err != nil {
			t.Fatal(err)
		}
	}
	leftAlpha, err := left.DeriveAlpha()
	if err != nil {
		t.Fatal(err)
	}
	rightAlpha, err := right.DeriveAlpha()
	if err != nil {
		t.Fatal(err)
	}
	assertOuterChallengeChanged(t, leftAlpha, rightAlpha, "rejection counter record -> alpha")
}

func TestOuterTranscriptAlphaExclusionAndCanonicalCounter(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	transcript := deriveOuterThroughW2(t, context, messages)
	domain := fft.NewDomain(context.LocalDomainSize)
	domainPoints := make([]fr.Element, int(context.LocalDomainSize))
	domainPoints[0].SetOne()
	for i := 1; i < len(domainPoints); i++ {
		domainPoints[i].Mul(&domainPoints[i-1], &domain.Generator)
	}

	for i := range domainPoints {
		if !transcript.alphaForbidden(domainPoints[i]) {
			t.Fatalf("H_X element %d was not rejected", i)
		}
	}
	if !transcript.alphaForbidden(fr.Element{}) {
		t.Fatal("zero alpha was not rejected")
	}
	for i := 1; i < len(domainPoints); i++ {
		translated := domainPoints[i]
		translated.Add(&translated, &context.Shift)
		if !transcript.alphaForbidden(translated) {
			t.Fatalf("sigma+B_T element %d was not rejected", i)
		}
		var inverseOmega fr.Element
		inverseOmega.Inverse(&context.LocalDomainGenerator)
		translated.Mul(&translated, &inverseOmega)
		if !transcript.alphaForbidden(translated) {
			t.Fatalf("omega^-1(sigma+B_T) element %d was not rejected", i)
		}
	}

	prefix := transcript.Bytes()
	alpha, err := transcript.DeriveAlpha()
	if err != nil {
		t.Fatal(err)
	}
	if transcript.alphaForbidden(alpha.Value) {
		t.Fatalf("accepted alpha is forbidden at counter %d", alpha.Counter)
	}
	for counter := uint32(0); counter < alpha.Counter; counter++ {
		candidate, inField := transcript.outerChallengeCandidate(ChallengeAlpha, counter, prefix)
		if inField && !transcript.alphaForbidden(candidate) {
			t.Fatalf("counter %d was admissible before accepted alpha counter %d", counter, alpha.Counter)
		}
	}
}

func TestOuterTranscriptRejectionCounterAndSumCheckExclusion(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW0(messages.W0); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveInitialChallenges(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW1(messages.W1); err != nil {
		t.Fatal(err)
	}
	prefix := transcript.Bytes()
	var firstInField fr.Element
	var firstCounter uint32
	for {
		candidate, inField := transcript.outerChallengeCandidate(ChallengeLambda, firstCounter, prefix)
		if inField {
			firstInField = candidate
			break
		}
		firstCounter++
	}
	forced, err := transcript.sampleOuterChallenge(ChallengeLambda, prefix, func(candidate fr.Element) bool {
		return candidate.Equal(&firstInField)
	})
	if err != nil {
		t.Fatal(err)
	}
	if forced.Counter <= firstCounter || forced.Value.Equal(&firstInField) {
		t.Fatalf("forced rejection did not advance canonically: first=%d accepted=%d", firstCounter, forced.Counter)
	}

	full, output := runTestOuterTranscript(t, context, messages)
	one := fr.One()
	for i := range output.SumCheck {
		if output.SumCheck[i].Value.IsZero() || output.SumCheck[i].Value.Equal(&one) {
			t.Fatalf("r_%d lies in {0,1}", i)
		}
	}
	if _, err := full.PriorTranscriptDigest(); err != nil {
		t.Fatal(err)
	}
}

func TestOuterTranscriptMuChallengesUseOneIndependentPrefix(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	transcript := deriveOuterThroughFinalEvaluations(t, context, messages)
	prefix := transcript.Bytes()
	mus, err := transcript.DeriveSourceCompressionChallenges()
	if err != nil {
		t.Fatal(err)
	}
	ids := []ChallengeID{ChallengeMuAlpha, ChallengeMuOmega, ChallengeMuXStar}
	for i := range mus {
		candidate, inField := transcript.outerChallengeCandidate(ids[i], mus[i].Counter, prefix)
		if !inField || !candidate.Equal(&mus[i].Value) {
			t.Fatalf("mu_%d was not sampled from the common y_final prefix", i)
		}
	}
}

func TestOuterTranscriptRejectsMalformedCanonicalObjectsBeforeHashing(t *testing.T) {
	context := testOuterTranscriptContext()
	messages := testOuterTranscriptMessages()
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	before := transcript.Digest()
	malformedW0 := messages.W0
	malformedW0.WitnessCommitments[1].X.SetUint64(1)
	malformedW0.WitnessCommitments[1].Y.SetUint64(1)
	if err := transcript.AppendW0(malformedW0); !errors.Is(err, ErrInvalidOuterTranscriptMessage) {
		t.Fatalf("malformed W0: got %v", err)
	}
	if transcript.Digest() != before {
		t.Fatal("rejected W0 mutated the transcript")
	}

	transcript = deriveOuterThroughProductCommitments(t, context, messages)
	if _, err := transcript.DeriveZetaTheta(); err != nil {
		t.Fatal(err)
	}
	malformedRound := messages.Rounds[0]
	malformedRound.Coefficients[4] = fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	before = transcript.Digest()
	if err := transcript.AppendSumCheckRound(malformedRound); !errors.Is(err, ErrInvalidOuterTranscriptMessage) {
		t.Fatalf("malformed round field: got %v", err)
	}
	if transcript.Digest() != before {
		t.Fatal("rejected SumCheck round mutated the transcript")
	}
}

func TestOuterTranscriptContextValidationBoundary(t *testing.T) {
	if _, err := NewOuterTranscriptWithDomain("", testOuterTranscriptContext()); !errors.Is(err, ErrInvalidOuterTranscriptContext) {
		t.Fatalf("empty suite domain: got %v", err)
	}

	context := testOuterTranscriptContext()
	context.IndexPrefixDigest = TranscriptDigest{}
	if _, err := NewOuterTranscript(context); !errors.Is(err, ErrInvalidOuterTranscriptContext) {
		t.Fatalf("zero pref_index digest: got %v", err)
	}

	context = testOuterTranscriptContext()
	context.LocalDomainGenerator.SetOne()
	if _, err := NewOuterTranscript(context); !errors.Is(err, ErrInvalidOuterTranscriptContext) {
		t.Fatalf("generator of wrong order: got %v", err)
	}

	context = testOuterTranscriptContext()
	domain := fft.NewDomain(context.LocalDomainSize)
	var xStar fr.Element
	xStar.Inverse(&domain.Generator)
	context.Shift.Sub(&xStar, &domain.Generator)
	if _, err := NewOuterTranscript(context); !errors.Is(err, ErrInvalidOuterTranscriptContext) {
		t.Fatalf("x_star-sigma in B_T: got %v", err)
	}

	context = testOuterTranscriptContext()
	context.ShiftCounter++
	if _, err := NewOuterTranscript(context); !errors.Is(err, ErrInvalidOuterTranscriptContext) ||
		!errors.Is(err, ErrIndexShiftNonMinimal) {
		t.Fatalf("nonminimal shift counter: got %v", err)
	}
}

func TestOuterTranscriptLargeDomainKeepsConstantSizeState(t *testing.T) {
	context := testOuterTranscriptContext()
	context.LocalDomainSize = 1 << 20
	context.LocalDomainGenerator = testOuterRootOfUnity(context.LocalDomainSize)
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	if len(transcript.Bytes()) > 1024 {
		t.Fatalf("context encoding grew with T: %d bytes", len(transcript.Bytes()))
	}
	if !transcript.alphaForbidden(context.LocalDomainGenerator) {
		t.Fatal("large-domain generator was not recognized as an H_X member")
	}
}

func TestOuterTranscriptGoldenVector(t *testing.T) {
	transcript, output := runTestOuterTranscript(t, testOuterTranscriptContext(), testOuterTranscriptMessages())
	challenges := output.ordered()
	expectedCounters := []uint32{0, 1, 3, 12, 5, 1, 3, 8, 0, 3, 4, 9, 5}
	expectedValues := []string{
		"1e180c279df01f6f36d2ce6094c710914d564bde858656746174e05274dfc322",
		"2ff6242afa29663c93215a30772c3ead0d1b95b1538a8d3fdf809d8594918f88",
		"1bd4576e00295ea471b4cb12e5d5d42ab33f4932fc889399813f57b8370e669b",
		"03f4f36f2309b0cc0d46c83c92121fc57fe3065449fd8c4e3fae37a22fbe32e3",
		"2fc59e40a679b5eff5cb9293f8a9bdd7eb7da1f01d6be07982ea932ae63b603a",
		"0d6409d3d0c39547fc5cc3b485ceeb72fd7719d21fdc8bf2e9ba811be3b9c7d2",
		"248c2926d2df1e108f98eb0abd703ada1f5914b7b8fd695af7e833fd59737c25",
		"27e85a61a8fcf1f9d135338153baff0a78c7e16ac67cb9741301ecc2e6c3bba7",
		"190586b316b0c317cc1cf0a9e9ed568094f17d1d3e78b8b6a1afd095ce56613e",
		"16197f52a4e20087a87ee89811e6fc75155ec59946bc06915ed088bdf9492157",
		"19bd4592bcef619b2c61573baa581f126064b46f7e3fe3c48f535709972badc1",
		"284dcf6170560e8a4e7012e6d838e377ffcbbfe56adc31d53f3ca1136a826787",
		"0c788664a7dca054ff4d0bcc3b4b31de0248bf03d6201f1eb7303eebe3e8e073",
	}
	for i := range challenges {
		encoded := challenges[i].Value.Bytes()
		if challenges[i].Counter != expectedCounters[i] || hex.EncodeToString(encoded[:]) != expectedValues[i] {
			t.Fatalf("golden %s mismatch: counter=%d value=%s", challenges[i].ID, challenges[i].Counter, hex.EncodeToString(encoded[:]))
		}
	}
	digest := transcript.Digest()
	const expectedDigest = "fbef8b1e4a8041b09bc8005153b398bc6ee6855a3a923dcc355f98b1e203907b"
	if hex.EncodeToString(digest[:]) != expectedDigest {
		t.Fatalf("golden outer transcript digest mismatch: %s", hex.EncodeToString(digest[:]))
	}
}

type outerTestMessages struct {
	W0           OuterW0Message
	W1           OuterW1Message
	W2           OuterW2Message
	ProductCheck OuterProductCheckCommitmentsMessage
	Rounds       []OuterSumCheckRoundMessage
	Final        OuterFinalEvaluationsMessage
}

type outerTestOutput struct {
	Initial   OuterInitialChallenges
	Lambda    ChallengeOut
	Alpha     ChallengeOut
	ZetaTheta OuterZetaThetaChallenges
	SumCheck  []ChallengeOut
	Mu        [LocalCompressedSourceCount]ChallengeOut
}

func (o outerTestOutput) ordered() []ChallengeOut {
	result := []ChallengeOut{o.Initial.EtaPart, o.Initial.EtaX, o.Initial.Gamma, o.Lambda, o.Alpha, o.ZetaTheta.Zeta}
	result = append(result, o.ZetaTheta.Theta...)
	result = append(result, o.SumCheck...)
	result = append(result, o.Mu[:]...)
	return result
}

func runTestOuterTranscript(t *testing.T, context OuterTranscriptContext, messages outerTestMessages) (*OuterTranscript, outerTestOutput) {
	t.Helper()
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW0(messages.W0); err != nil {
		t.Fatal(err)
	}
	var output outerTestOutput
	if output.Initial, err = transcript.DeriveInitialChallenges(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW1(messages.W1); err != nil {
		t.Fatal(err)
	}
	if output.Lambda, err = transcript.DeriveLambda(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW2(messages.W2); err != nil {
		t.Fatal(err)
	}
	if output.Alpha, err = transcript.DeriveAlpha(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.MarkRetainedW3Complete(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendProductCheckCommitments(messages.ProductCheck); err != nil {
		t.Fatal(err)
	}
	if output.ZetaTheta, err = transcript.DeriveZetaTheta(); err != nil {
		t.Fatal(err)
	}
	output.SumCheck = make([]ChallengeOut, len(messages.Rounds))
	for round := range messages.Rounds {
		if err := transcript.AppendSumCheckRound(messages.Rounds[round]); err != nil {
			t.Fatal(err)
		}
		if output.SumCheck[round], err = transcript.DeriveSumCheckChallenge(); err != nil {
			t.Fatal(err)
		}
	}
	if err := transcript.AppendFinalEvaluations(messages.Final); err != nil {
		t.Fatal(err)
	}
	if output.Mu, err = transcript.DeriveSourceCompressionChallenges(); err != nil {
		t.Fatal(err)
	}
	return transcript, output
}

func deriveOuterThroughW2(t *testing.T, context OuterTranscriptContext, messages outerTestMessages) *OuterTranscript {
	t.Helper()
	transcript, err := NewOuterTranscript(context)
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW0(messages.W0); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveInitialChallenges(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW1(messages.W1); err != nil {
		t.Fatal(err)
	}
	if _, err := transcript.DeriveLambda(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendW2(messages.W2); err != nil {
		t.Fatal(err)
	}
	return transcript
}

func deriveOuterThroughAlpha(t *testing.T, context OuterTranscriptContext, messages outerTestMessages) *OuterTranscript {
	t.Helper()
	transcript := deriveOuterThroughW2(t, context, messages)
	if _, err := transcript.DeriveAlpha(); err != nil {
		t.Fatal(err)
	}
	return transcript
}

func deriveOuterThroughProductCommitments(t *testing.T, context OuterTranscriptContext, messages outerTestMessages) *OuterTranscript {
	t.Helper()
	transcript := deriveOuterThroughAlpha(t, context, messages)
	if err := transcript.MarkRetainedW3Complete(); err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendProductCheckCommitments(messages.ProductCheck); err != nil {
		t.Fatal(err)
	}
	return transcript
}

func deriveOuterThroughFinalEvaluations(t *testing.T, context OuterTranscriptContext, messages outerTestMessages) *OuterTranscript {
	t.Helper()
	transcript := deriveOuterThroughProductCommitments(t, context, messages)
	if _, err := transcript.DeriveZetaTheta(); err != nil {
		t.Fatal(err)
	}
	for round := range messages.Rounds {
		if err := transcript.AppendSumCheckRound(messages.Rounds[round]); err != nil {
			t.Fatal(err)
		}
		if _, err := transcript.DeriveSumCheckChallenge(); err != nil {
			t.Fatal(err)
		}
	}
	if err := transcript.AppendFinalEvaluations(messages.Final); err != nil {
		t.Fatal(err)
	}
	return transcript
}

func testOuterTranscriptContext() OuterTranscriptContext {
	domain := fft.NewDomain(4)
	indexPrefixDigest := NewTranscriptDigest([]byte("outer-index-prefix-A"))
	shift, err := DeriveIndexShift(indexPrefixDigest, 4, domain.Generator)
	if err != nil {
		panic(err)
	}
	return OuterTranscriptContext{
		ProtocolVersion:       "dlinkzg-outer-test-v1",
		PublicStatementDigest: NewTranscriptDigest([]byte("outer-statement-A")),
		SRSDigest:             NewTranscriptDigest([]byte("outer-srs-A")),
		IndexDigest:           NewTranscriptDigest([]byte("outer-index-A")),
		IndexPrefixDigest:     indexPrefixDigest,
		ShiftCounter:          shift.Counter,
		Shift:                 shift.Sigma,
		PartyManifestDigest:   NewTranscriptDigest([]byte("outer-P1,P2,P3,P4")),
		SessionNonce:          [32]byte(NewTranscriptDigest([]byte("outer-session-0001"))),
		PartitionCount:        4,
		LocalDomainSize:       4,
		LocalDomainGenerator:  domain.Generator,
	}
}

func testOuterTranscriptMessages() outerTestMessages {
	messages := outerTestMessages{
		W0:           OuterW0Message{WitnessCommitments: [3]bn254.G1Affine{testG1(101), testG1(102), testG1(103)}},
		W1:           OuterW1Message{AccumulatorCommitment: testG1(104)},
		W2:           OuterW2Message{QuotientCommitments: [3]bn254.G1Affine{testG1(105), testG1(106), testG1(107)}},
		ProductCheck: OuterProductCheckCommitmentsMessage{Commitments: [2]bn254.G1Affine{testG1(108), testG1(109)}},
		Rounds:       make([]OuterSumCheckRoundMessage, 2),
	}
	for round := range messages.Rounds {
		for coefficient := range messages.Rounds[round].Coefficients {
			messages.Rounds[round].Coefficients[coefficient] = fr.NewElement(uint64(200 + 10*round + coefficient))
		}
	}
	for i := range messages.Final.Terminal {
		messages.Final.Terminal[i] = fr.NewElement(uint64(300 + i))
	}
	for i := range messages.Final.ProductCheck {
		messages.Final.ProductCheck[i] = fr.NewElement(uint64(400 + i))
	}
	return messages
}

func assertOuterChallengeSlicesEqual(t *testing.T, left, right []ChallengeOut) {
	t.Helper()
	if len(left) != len(right) {
		t.Fatalf("challenge lengths differ: %d != %d", len(left), len(right))
	}
	for i := range left {
		assertOuterChallengeEqual(t, left[i], right[i], string(left[i].ID))
	}
}

func assertOuterChallengeEqual(t *testing.T, left, right ChallengeOut, label string) {
	t.Helper()
	if left.ID != right.ID || left.Counter != right.Counter || !left.Value.Equal(&right.Value) {
		t.Fatalf("%s challenge differs", label)
	}
}

func assertOuterChallengeChanged(t *testing.T, left, right ChallengeOut, label string) {
	t.Helper()
	if left.Counter == right.Counter && left.Value.Equal(&right.Value) {
		t.Fatalf("%s did not change challenge", label)
	}
}

func outerFieldPointer(value uint64) *fr.Element {
	element := fr.NewElement(value)
	return &element
}

func testOuterRootOfUnity(order uint64) fr.Element {
	var root fr.Element
	root.SetString("19103219067921713944291392827692070036145651957329286315305642004821462161904")
	const maxOrderRoot = 28
	logOrder := bits.TrailingZeros64(order)
	exponent := uint64(1) << uint(maxOrderRoot-logOrder)
	root.Exp(root, new(big.Int).SetUint64(exponent))
	return root
}
