package dlinkzg

// This file implements the accepted-path public Fiat--Shamir prefix that ends
// immediately before U0. The manifest-ordered W3 records are retained for the
// optimistic accountability trace, but Appendix B does not put either their
// raw M-by-21 payload or an invented digest/root in the accepted public proof.
// MarkRetainedW3Complete therefore enforces the protocol transition without
// appending bytes. The following C_t0,C_t1 commitments, SumCheck transcript,
// terminal values, and source-compression challenges bind the public suffix.

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// OuterTranscriptDomain is the protocol-wide domain separator for the
	// accepted public W0--SumCheck--mu prefix.
	OuterTranscriptDomain = "DLinKZG/outer-piop-transcript/v1"

	outerChallengeHashDomain = "DLinKZG/outer-piop-hash-to-field/sha256/v1"

	// OuterTranscriptBoundaryNotice records what this component deliberately
	// does not invent. W3 records remain outside the accepted public proof,
	// while the concrete benchmark suite validates its setup index shift before
	// accepting the transcript context.
	OuterTranscriptBoundaryNotice = "accepted public outer PIOP transcript through mu: W3 is an accountability-only retained transition; the BN254/SHA-256 benchmark suite validates the canonical setup index shift"

	OuterTerminalEvaluationCount = LocalTerminalTotalCount
	OuterProductCheckClaimCount  = 5
)

const (
	ChallengeEtaPart ChallengeID = "eta_part"
	ChallengeEtaX    ChallengeID = "eta_X"
	ChallengeGamma   ChallengeID = "gamma"
	ChallengeLambda  ChallengeID = "lambda"
	ChallengeAlpha   ChallengeID = "alpha"
	ChallengeZeta    ChallengeID = "zeta"
	ChallengeMuAlpha ChallengeID = "mu/alpha"
	ChallengeMuOmega ChallengeID = "mu/omega_alpha"
	ChallengeMuXStar ChallengeID = "mu/x_star"
)

var (
	ErrInvalidOuterTranscriptContext = errors.New("dlinkzg: invalid outer transcript context")
	ErrOuterTranscriptOrder          = errors.New("dlinkzg: invalid outer transcript order")
	ErrInvalidOuterTranscriptMessage = errors.New("dlinkzg: invalid outer transcript message")
	ErrOuterTranscriptIncomplete     = errors.New("dlinkzg: outer transcript has not reached the U0 boundary")
	ErrOuterChallengeSampling        = errors.New("dlinkzg: outer Fiat--Shamir rejection counter exhausted")
)

// OuterTranscriptContext is the immutable public context for the outer PIOP
// prefix. IndexPrefixDigest canonically represents pref_index. ShiftCounter
// and Shift are bound as public data and must be the first admissible output
// of the concrete benchmark suite's HashToField instantiation.
type OuterTranscriptContext struct {
	ProtocolVersion       string
	PublicStatementDigest TranscriptDigest
	SRSDigest             TranscriptDigest
	IndexDigest           TranscriptDigest
	IndexPrefixDigest     TranscriptDigest
	ShiftCounter          uint32
	Shift                 fr.Element
	PartyManifestDigest   TranscriptDigest
	SessionNonce          [32]byte

	PartitionCount       uint64
	LocalDomainSize      uint64
	LocalDomainGenerator fr.Element
}

// OuterW0Message fixes C_A,C_B,C_C before (eta_part,eta_X,gamma).
type OuterW0Message struct {
	WitnessCommitments [3]bn254.G1Affine
}

// OuterW1Message fixes C_Z before lambda.
type OuterW1Message struct {
	AccumulatorCommitment bn254.G1Affine
}

// OuterW2Message fixes C_H0,C_H1,C_H2 before alpha.
type OuterW2Message struct {
	QuotientCommitments [3]bn254.G1Affine
}

// OuterProductCheckCommitmentsMessage fixes C_t0,C_t1 before zeta and theta.
// The retained W3 party records must already be complete before this message.
type OuterProductCheckCommitmentsMessage struct {
	Commitments [2]bn254.G1Affine
}

// OuterSumCheckRoundMessage is the canonical six-coefficient, ascending-monomial
// encoding of one degree-at-most-five SumCheck round polynomial.
type OuterSumCheckRoundMessage struct {
	Coefficients [SumCheckRoundCoefficients]fr.Element
}

// OuterFinalEvaluationsMessage is y_final: the 21 folded circuit evaluations in
// the 13/1/7 order followed by the five ProductCheck evaluations.
type OuterFinalEvaluationsMessage struct {
	Terminal     [OuterTerminalEvaluationCount]fr.Element
	ProductCheck [OuterProductCheckClaimCount]fr.Element
}

// OuterInitialChallenges is sampled as one independent challenge batch from the
// fixed W0 prefix. No member depends on another member of the same batch.
type OuterInitialChallenges struct {
	EtaPart ChallengeOut
	EtaX    ChallengeOut
	Gamma   ChallengeOut
}

// OuterZetaThetaChallenges is sampled as one independent batch from the fixed
// C_t0,C_t1 prefix. Theta is in little-endian SumCheck-coordinate order.
type OuterZetaThetaChallenges struct {
	Zeta  ChallengeOut
	Theta []ChallengeOut
}

type outerTranscriptStep uint8

const (
	outerStepW0 outerTranscriptStep = iota
	outerStepInitialChallenges
	outerStepW1
	outerStepLambda
	outerStepW2
	outerStepAlpha
	outerStepRetainedW3
	outerStepProductCheckCommitments
	outerStepZetaTheta
	outerStepSumCheckRound
	outerStepSumCheckChallenge
	outerStepFinalEvaluations
	outerStepMu
	outerStepComplete
)

// OuterTranscript constructs and replays
//
//	W0 -> (eta_part,eta_X,gamma) -> W1 -> lambda -> W2 -> alpha
//	   -> retained-W3 boundary -> (C_t0,C_t1) -> (zeta,theta)
//	   -> [SumCheck round_k -> r_k]_k -> y_final -> (mu_0,mu_1,mu_2).
//
// Its completed digest is suitable as TranscriptContext.PriorTranscriptDigest
// for NewOpeningTranscript. The type is not safe for concurrent mutation.
type OuterTranscript struct {
	outerDomain string
	context     OuterTranscriptContext
	bytes       []byte
	next        outerTranscriptStep
	round       int
	roundCount  int
	omega       fr.Element
}

// NewOuterTranscript initializes the standard outer transcript suite.
func NewOuterTranscript(context OuterTranscriptContext) (*OuterTranscript, error) {
	return newOuterTranscript("", context)
}

// NewOuterTranscriptWithDomain adds a nonempty suite-level domain separator.
func NewOuterTranscriptWithDomain(domain string, context OuterTranscriptContext) (*OuterTranscript, error) {
	if domain == "" {
		return nil, fmt.Errorf("%w: empty domain", ErrInvalidOuterTranscriptContext)
	}
	return newOuterTranscript(domain, context)
}

func newOuterTranscript(outerDomain string, context OuterTranscriptContext) (*OuterTranscript, error) {
	if err := validateOuterTranscriptContext(context); err != nil {
		return nil, err
	}

	t := &OuterTranscript{
		outerDomain: outerDomain,
		context:     context,
		next:        outerStepW0,
		roundCount:  bits.Len64(context.PartitionCount) - 1,
		omega:       context.LocalDomainGenerator,
	}

	payload := make([]byte, 0, 512)
	payload = appendLengthPrefixed(payload, []byte(OuterTranscriptDomain))
	payload = appendLengthPrefixed(payload, []byte(outerDomain))
	payload = appendLengthPrefixed(payload, []byte(context.ProtocolVersion))
	payload = appendLengthPrefixed(payload, context.PublicStatementDigest[:])
	payload = appendLengthPrefixed(payload, context.SRSDigest[:])
	payload = appendLengthPrefixed(payload, context.IndexDigest[:])
	payload = appendLengthPrefixed(payload, context.IndexPrefixDigest[:])
	payload = appendUint32(payload, context.ShiftCounter)
	payload = appendField(payload, &context.Shift)
	payload = appendLengthPrefixed(payload, context.PartyManifestDigest[:])
	payload = appendLengthPrefixed(payload, context.SessionNonce[:])
	payload = appendUint64(payload, context.PartitionCount)
	payload = appendUint64(payload, context.LocalDomainSize)
	payload = appendField(payload, &context.LocalDomainGenerator)
	t.appendRecord("context", payload)
	return t, nil
}

// AppendW0 appends the aggregate witness commitments.
func (t *OuterTranscript) AppendW0(message OuterW0Message) error {
	if err := t.expect(outerStepW0); err != nil {
		return err
	}
	payload, err := encodeOuterG1Array(message.WitnessCommitments[:], "W0")
	if err != nil {
		return err
	}
	t.appendRecord("W0", payload)
	t.next = outerStepInitialChallenges
	return nil
}

// DeriveInitialChallenges samples eta_part, eta_X, and gamma independently
// from the same fixed W0 prefix and appends one ordered challenge batch.
func (t *OuterTranscript) DeriveInitialChallenges() (OuterInitialChallenges, error) {
	if err := t.expect(outerStepInitialChallenges); err != nil {
		return OuterInitialChallenges{}, err
	}
	outs, err := t.sampleIndependentBatch([]ChallengeID{ChallengeEtaPart, ChallengeEtaX, ChallengeGamma}, nil)
	if err != nil {
		return OuterInitialChallenges{}, err
	}
	t.appendChallengeBatch("InitialChallenges", outs)
	t.next = outerStepW1
	return OuterInitialChallenges{EtaPart: outs[0], EtaX: outs[1], Gamma: outs[2]}, nil
}

// AppendW1 appends the aggregate accumulator commitment.
func (t *OuterTranscript) AppendW1(message OuterW1Message) error {
	if err := t.expect(outerStepW1); err != nil {
		return err
	}
	if err := validateTranscriptG1(&message.AccumulatorCommitment); err != nil {
		return fmt.Errorf("%w: W1 accumulator commitment", ErrInvalidOuterTranscriptMessage)
	}
	payload := appendG1(nil, &message.AccumulatorCommitment)
	t.appendRecord("W1", payload)
	t.next = outerStepLambda
	return nil
}

// DeriveLambda samples unrestricted lambda after W1.
func (t *OuterTranscript) DeriveLambda() (ChallengeOut, error) {
	return t.deriveSingleton(outerStepLambda, ChallengeLambda, nil, outerStepW2)
}

// AppendW2 appends the three quotient-chunk commitments.
func (t *OuterTranscript) AppendW2(message OuterW2Message) error {
	if err := t.expect(outerStepW2); err != nil {
		return err
	}
	payload, err := encodeOuterG1Array(message.QuotientCommitments[:], "W2")
	if err != nil {
		return err
	}
	t.appendRecord("W2", payload)
	t.next = outerStepAlpha
	return nil
}

// DeriveAlpha rejection-samples alpha outside
// H_X union {0} union (sigma+B_T) union omega^-1(sigma+B_T).
func (t *OuterTranscript) DeriveAlpha() (ChallengeOut, error) {
	if err := t.expect(outerStepAlpha); err != nil {
		return ChallengeOut{}, err
	}
	out, err := t.sampleOuterChallenge(ChallengeAlpha, t.bytes, t.alphaForbidden)
	if err != nil {
		return ChallengeOut{}, err
	}
	t.appendChallengeRecord("AlphaChallenge", out)
	t.next = outerStepRetainedW3
	return out, nil
}

// MarkRetainedW3Complete enforces the W3 -> C_t0,C_t1 transition. It appends
// no public bytes: raw W3 records remain authenticated coordinator state for
// optimistic tracing and are not part of the accepted proof.
func (t *OuterTranscript) MarkRetainedW3Complete() error {
	if err := t.expect(outerStepRetainedW3); err != nil {
		return err
	}
	t.next = outerStepProductCheckCommitments
	return nil
}

// AppendProductCheckCommitments appends C_t0,C_t1 after retained W3 completes.
func (t *OuterTranscript) AppendProductCheckCommitments(message OuterProductCheckCommitmentsMessage) error {
	if err := t.expect(outerStepProductCheckCommitments); err != nil {
		return err
	}
	payload, err := encodeOuterG1Array(message.Commitments[:], "ProductCheck")
	if err != nil {
		return err
	}
	t.appendRecord("ProductCheckCommitments", payload)
	t.next = outerStepZetaTheta
	return nil
}

// DeriveZetaTheta samples zeta and all theta coordinates independently from
// the same fixed C_t0,C_t1 prefix, then appends one ordered batch record.
func (t *OuterTranscript) DeriveZetaTheta() (OuterZetaThetaChallenges, error) {
	if err := t.expect(outerStepZetaTheta); err != nil {
		return OuterZetaThetaChallenges{}, err
	}
	ids := make([]ChallengeID, 1+t.roundCount)
	ids[0] = ChallengeZeta
	for i := 0; i < t.roundCount; i++ {
		ids[1+i] = thetaChallengeID(i)
	}
	outs, err := t.sampleIndependentBatch(ids, nil)
	if err != nil {
		return OuterZetaThetaChallenges{}, err
	}
	t.appendChallengeBatch("ZetaThetaChallenges", outs)
	t.next = outerStepSumCheckRound
	return OuterZetaThetaChallenges{Zeta: outs[0], Theta: append([]ChallengeOut(nil), outs[1:]...)}, nil
}

// AppendSumCheckRound appends round k's six coefficients before r_k.
func (t *OuterTranscript) AppendSumCheckRound(message OuterSumCheckRoundMessage) error {
	if err := t.expect(outerStepSumCheckRound); err != nil {
		return err
	}
	for i := range message.Coefficients {
		if err := validateTranscriptField(&message.Coefficients[i]); err != nil {
			return fmt.Errorf("%w: SumCheck round %d coefficient %d", ErrInvalidOuterTranscriptMessage, t.round, i)
		}
	}
	payload := appendUint64(nil, uint64(t.round))
	for i := range message.Coefficients {
		payload = appendField(payload, &message.Coefficients[i])
	}
	t.appendRecord("SumCheckRound", payload)
	t.next = outerStepSumCheckChallenge
	return nil
}

// DeriveSumCheckChallenge rejection-samples r_k from F\{0,1} only after the
// corresponding round polynomial is fixed.
func (t *OuterTranscript) DeriveSumCheckChallenge() (ChallengeOut, error) {
	if err := t.expect(outerStepSumCheckChallenge); err != nil {
		return ChallengeOut{}, err
	}
	zero := fr.Element{}
	one := fr.One()
	id := sumCheckChallengeID(t.round)
	out, err := t.sampleOuterChallenge(id, t.bytes, fieldSetForbidden([]fr.Element{zero, one}))
	if err != nil {
		return ChallengeOut{}, err
	}
	t.appendChallengeRecord("SumCheckChallenge", out)
	t.round++
	if t.round == t.roundCount {
		t.next = outerStepFinalEvaluations
	} else {
		t.next = outerStepSumCheckRound
	}
	return out, nil
}

// AppendFinalEvaluations appends y_final only after the last SumCheck
// challenge has fixed r.
func (t *OuterTranscript) AppendFinalEvaluations(message OuterFinalEvaluationsMessage) error {
	if err := t.expect(outerStepFinalEvaluations); err != nil {
		return err
	}
	payload := make([]byte, 0, (OuterTerminalEvaluationCount+OuterProductCheckClaimCount)*fr.Bytes)
	for i := range message.Terminal {
		if err := validateTranscriptField(&message.Terminal[i]); err != nil {
			return fmt.Errorf("%w: terminal evaluation %d", ErrInvalidOuterTranscriptMessage, i)
		}
		payload = appendField(payload, &message.Terminal[i])
	}
	for i := range message.ProductCheck {
		if err := validateTranscriptField(&message.ProductCheck[i]); err != nil {
			return fmt.Errorf("%w: ProductCheck evaluation %d", ErrInvalidOuterTranscriptMessage, i)
		}
		payload = appendField(payload, &message.ProductCheck[i])
	}
	t.appendRecord("FinalEvaluations", payload)
	t.next = outerStepMu
	return nil
}

// DeriveSourceCompressionChallenges samples the three mu challenges from the
// same y_final prefix using distinct point labels. No mu depends on a sibling
// mu record. The resulting digest is the exact boundary immediately before U0.
func (t *OuterTranscript) DeriveSourceCompressionChallenges() ([LocalCompressedSourceCount]ChallengeOut, error) {
	var result [LocalCompressedSourceCount]ChallengeOut
	if err := t.expect(outerStepMu); err != nil {
		return result, err
	}
	ids := []ChallengeID{ChallengeMuAlpha, ChallengeMuOmega, ChallengeMuXStar}
	outs, err := t.sampleIndependentBatch(ids, nil)
	if err != nil {
		return result, err
	}
	copy(result[:], outs)
	t.appendChallengeBatch("SourceCompressionChallenges", outs)
	t.next = outerStepComplete
	return result, nil
}

// PriorTranscriptDigest returns the composition digest consumed by
// OpeningTranscript. It fails before all three mu records are fixed.
func (t *OuterTranscript) PriorTranscriptDigest() (TranscriptDigest, error) {
	if err := t.expect(outerStepComplete); err != nil {
		return TranscriptDigest{}, fmt.Errorf("%w: %v", ErrOuterTranscriptIncomplete, err)
	}
	return TranscriptDigest(t.Digest()), nil
}

// OpeningContext returns the shared public context with the completed outer
// digest installed as PriorTranscriptDigest.
func (t *OuterTranscript) OpeningContext() (TranscriptContext, error) {
	digest, err := t.PriorTranscriptDigest()
	if err != nil {
		return TranscriptContext{}, err
	}
	return TranscriptContext{
		ProtocolVersion:       t.context.ProtocolVersion,
		PublicStatementDigest: t.context.PublicStatementDigest,
		SRSDigest:             t.context.SRSDigest,
		IndexDigest:           t.context.IndexDigest,
		PartyManifestDigest:   t.context.PartyManifestDigest,
		SessionNonce:          t.context.SessionNonce,
		PriorTranscriptDigest: digest,
	}, nil
}

// Bytes returns a copy of the canonical public transcript prefix.
func (t *OuterTranscript) Bytes() []byte {
	if t == nil {
		return nil
	}
	return append([]byte(nil), t.bytes...)
}

// Digest returns the SHA-256 digest of the prefix accumulated so far.
func (t *OuterTranscript) Digest() [sha256.Size]byte {
	if t == nil {
		return sha256.Sum256(nil)
	}
	return sha256.Sum256(t.bytes)
}

func validateOuterTranscriptContext(context OuterTranscriptContext) error {
	if context.ProtocolVersion == "" {
		return fmt.Errorf("%w: empty protocol version", ErrInvalidOuterTranscriptContext)
	}
	requiredDigests := []struct {
		name  string
		value TranscriptDigest
	}{
		{"public statement", context.PublicStatementDigest},
		{"SRS identifier", context.SRSDigest},
		{"index", context.IndexDigest},
		{"index prefix", context.IndexPrefixDigest},
		{"party manifest", context.PartyManifestDigest},
	}
	for _, item := range requiredDigests {
		if item.value == (TranscriptDigest{}) {
			return fmt.Errorf("%w: zero %s digest", ErrInvalidOuterTranscriptContext, item.name)
		}
	}
	if context.SessionNonce == ([32]byte{}) {
		return fmt.Errorf("%w: zero session nonce", ErrInvalidOuterTranscriptContext)
	}
	if context.PartitionCount < 2 || !isPowerOfTwo64(context.PartitionCount) {
		return fmt.Errorf("%w: partition count %d is not a power of two >= 2", ErrInvalidOuterTranscriptContext, context.PartitionCount)
	}
	if context.LocalDomainSize < 4 || !isPowerOfTwo64(context.LocalDomainSize) || context.PartitionCount > context.LocalDomainSize {
		return fmt.Errorf("%w: require 2 <= M <= T, power-of-two T >= 4", ErrInvalidOuterTranscriptContext)
	}
	if context.LocalDomainSize > (math.MaxUint64-5)/3 {
		return fmt.Errorf("%w: local domain size is unsupported", ErrInvalidOuterTranscriptContext)
	}
	fieldRequirement := new(big.Int).SetUint64(3*context.LocalDomainSize + 5)
	if fr.Modulus().Cmp(fieldRequirement) <= 0 || fr.Modulus().Cmp(new(big.Int).SetUint64(context.PartitionCount)) <= 0 {
		return fmt.Errorf("%w: field-size requirement is not met", ErrInvalidOuterTranscriptContext)
	}
	if err := validateTranscriptField(&context.Shift); err != nil {
		return fmt.Errorf("%w: noncanonical shift", ErrInvalidOuterTranscriptContext)
	}
	if err := validateTranscriptField(&context.LocalDomainGenerator); err != nil || context.LocalDomainGenerator.IsZero() {
		return fmt.Errorf("%w: invalid local-domain generator", ErrInvalidOuterTranscriptContext)
	}
	one := fr.One()
	fullOrder := outerPower(context.LocalDomainGenerator, context.LocalDomainSize)
	halfOrder := outerPower(context.LocalDomainGenerator, context.LocalDomainSize/2)
	if !fullOrder.Equal(&one) || halfOrder.Equal(&one) {
		return fmt.Errorf("%w: generator does not have exact order T", ErrInvalidOuterTranscriptContext)
	}

	if err := VerifyIndexShift(
		context.IndexPrefixDigest,
		context.LocalDomainSize,
		context.LocalDomainGenerator,
		context.ShiftCounter,
		context.Shift,
	); err != nil {
		return fmt.Errorf("%w: invalid canonical index shift: %w", ErrInvalidOuterTranscriptContext, err)
	}
	return nil
}

func (t *OuterTranscript) expect(want outerTranscriptStep) error {
	if t == nil {
		return fmt.Errorf("%w: nil transcript", ErrOuterTranscriptOrder)
	}
	if t.next != want {
		return fmt.Errorf("%w: got step %d, want step %d", ErrOuterTranscriptOrder, t.next, want)
	}
	return nil
}

func (t *OuterTranscript) deriveSingleton(step outerTranscriptStep, id ChallengeID, forbidden func(fr.Element) bool, next outerTranscriptStep) (ChallengeOut, error) {
	if err := t.expect(step); err != nil {
		return ChallengeOut{}, err
	}
	out, err := t.sampleOuterChallenge(id, t.bytes, forbidden)
	if err != nil {
		return ChallengeOut{}, err
	}
	t.appendChallengeRecord("OuterChallenge", out)
	t.next = next
	return out, nil
}

func (t *OuterTranscript) sampleIndependentBatch(ids []ChallengeID, forbidden []func(fr.Element) bool) ([]ChallengeOut, error) {
	prefix := append([]byte(nil), t.bytes...)
	outs := make([]ChallengeOut, len(ids))
	for i := range ids {
		var predicate func(fr.Element) bool
		if forbidden != nil {
			predicate = forbidden[i]
		}
		out, err := t.sampleOuterChallenge(ids[i], prefix, predicate)
		if err != nil {
			return nil, err
		}
		outs[i] = out
	}
	return outs, nil
}

func (t *OuterTranscript) sampleOuterChallenge(id ChallengeID, prefix []byte, forbidden func(fr.Element) bool) (ChallengeOut, error) {
	for counter := uint32(0); ; counter++ {
		candidate, inField := t.outerChallengeCandidate(id, counter, prefix)
		if inField && (forbidden == nil || !forbidden(candidate)) {
			return ChallengeOut{ID: id, Value: candidate, Counter: counter}, nil
		}
		if counter == math.MaxUint32 {
			break
		}
	}
	return ChallengeOut{}, ErrOuterChallengeSampling
}

func (t *OuterTranscript) outerChallengeCandidate(id ChallengeID, counter uint32, prefix []byte) (fr.Element, bool) {
	input := make([]byte, 0, len(prefix)+128)
	input = appendLengthPrefixed(input, []byte(outerChallengeHashDomain))
	input = appendLengthPrefixed(input, []byte(OuterTranscriptDomain))
	input = appendLengthPrefixed(input, []byte(t.outerDomain))
	input = appendLengthPrefixed(input, []byte(id))
	input = appendLengthPrefixed(input, prefix)
	input = appendUint32(input, counter)
	digest := sha256.Sum256(input)

	var integer big.Int
	integer.SetBytes(digest[:])
	if integer.Cmp(fr.Modulus()) >= 0 {
		return fr.Element{}, false
	}
	var candidate fr.Element
	candidate.SetBytes(digest[:])
	return candidate, true
}

func (t *OuterTranscript) alphaForbidden(candidate fr.Element) bool {
	if candidate.IsZero() || outerInDomain(candidate, t.context.LocalDomainSize) {
		return true
	}
	translated := candidate
	translated.Sub(&translated, &t.context.Shift)
	if outerInBoundarySet(translated, t.context.LocalDomainSize) {
		return true
	}
	translated.Mul(&candidate, &t.omega)
	translated.Sub(&translated, &t.context.Shift)
	return outerInBoundarySet(translated, t.context.LocalDomainSize)
}

func (t *OuterTranscript) appendChallengeRecord(label string, out ChallengeOut) {
	t.appendRecord(label, encodeOuterChallenge(out))
}

func (t *OuterTranscript) appendChallengeBatch(label string, outs []ChallengeOut) {
	payload := appendUint64(nil, uint64(len(outs)))
	for i := range outs {
		payload = appendLengthPrefixed(payload, encodeOuterChallenge(outs[i]))
	}
	t.appendRecord(label, payload)
}

func (t *OuterTranscript) appendRecord(label string, payload []byte) {
	t.bytes = appendLengthPrefixed(t.bytes, []byte(label))
	t.bytes = appendLengthPrefixed(t.bytes, payload)
}

func encodeOuterChallenge(out ChallengeOut) []byte {
	payload := appendLengthPrefixed(nil, []byte(out.ID))
	payload = appendUint32(payload, out.Counter)
	payload = appendField(payload, &out.Value)
	return payload
}

func encodeOuterG1Array(points []bn254.G1Affine, phase string) ([]byte, error) {
	payload := make([]byte, 0, len(points)*bn254.SizeOfG1AffineCompressed)
	for i := range points {
		if err := validateTranscriptG1(&points[i]); err != nil {
			return nil, fmt.Errorf("%w: %s commitment %d", ErrInvalidOuterTranscriptMessage, phase, i)
		}
		payload = appendG1(payload, &points[i])
	}
	return payload, nil
}

func appendUint32(dst []byte, value uint32) []byte {
	var encoded [4]byte
	binary.BigEndian.PutUint32(encoded[:], value)
	return append(dst, encoded[:]...)
}

func appendUint64(dst []byte, value uint64) []byte {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], value)
	return append(dst, encoded[:]...)
}

func thetaChallengeID(coordinate int) ChallengeID {
	return ChallengeID(fmt.Sprintf("theta/%d", coordinate))
}

func sumCheckChallengeID(round int) ChallengeID {
	return ChallengeID(fmt.Sprintf("sumcheck/r/%d", round))
}

func fieldSetForbidden(set []fr.Element) func(fr.Element) bool {
	return func(candidate fr.Element) bool {
		return outerInSet(candidate, set)
	}
}

func outerInBoundarySet(value fr.Element, domainSize uint64) bool {
	if value.IsOne() {
		return false
	}
	return outerInDomain(value, domainSize)
}

func outerInDomain(value fr.Element, domainSize uint64) bool {
	one := fr.One()
	power := outerPower(value, domainSize)
	return power.Equal(&one)
}

func outerInSet(value fr.Element, set []fr.Element) bool {
	for i := range set {
		if value.Equal(&set[i]) {
			return true
		}
	}
	return false
}

func outerPower(base fr.Element, exponent uint64) fr.Element {
	var result fr.Element
	result.Exp(base, new(big.Int).SetUint64(exponent))
	return result
}

func isPowerOfTwo64(value uint64) bool {
	return value != 0 && value&(value-1) == 0
}
