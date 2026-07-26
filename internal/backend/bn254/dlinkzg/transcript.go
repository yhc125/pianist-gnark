package dlinkzg

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// OpeningTranscriptDomain is the protocol-wide domain separator for the
	// experimental DLinKZG opening transcript.
	OpeningTranscriptDomain = "DLinKZG/opening-transcript/v1"

	challengeHashDomain = "DLinKZG/hash-to-field/sha256/v1"
)

// ChallengeID names a Fiat--Shamir challenge in the DLinKZG opening
// schedule. The string values are part of the transcript format.
type ChallengeID string

const (
	ChallengeXi    ChallengeID = "xi"
	ChallengeNu    ChallengeID = "nu"
	ChallengeZ     ChallengeID = "z_ch"
	ChallengeBeta  ChallengeID = "beta"
	ChallengeKappa ChallengeID = "kappa"
	ChallengeDelta ChallengeID = "delta"
)

var (
	// ErrInvalidTranscriptContext is returned when a context omits a field
	// required by the paper's transcript convention.
	ErrInvalidTranscriptContext = errors.New("dlinkzg: invalid transcript context")
	// ErrTranscriptOrder is returned when a message or challenge is appended
	// outside the fixed U0--U3 Fiat--Shamir schedule.
	ErrTranscriptOrder = errors.New("dlinkzg: invalid transcript order")
	// ErrInvalidTranscriptMessage is returned before hashing a malformed group
	// element. This makes the compressed point encoding injective over the
	// accepted transcript language.
	ErrInvalidTranscriptMessage = errors.New("dlinkzg: invalid transcript message")
	// ErrChallengeSampling is returned if the uint32 rejection counter is
	// exhausted. Reaching this error has negligible probability.
	ErrChallengeSampling = errors.New("dlinkzg: Fiat--Shamir rejection counter exhausted")
)

// TranscriptDigest is the SHA-256 digest of one canonical public encoding.
// Fixing the digest algorithm and width gives every context object exactly one
// transcript representation.
type TranscriptDigest [sha256.Size]byte

// NewTranscriptDigest hashes a canonical public encoding for use in
// TranscriptContext.
func NewTranscriptDigest(canonicalEncoding []byte) TranscriptDigest {
	return TranscriptDigest(sha256.Sum256(canonicalEncoding))
}

// TranscriptContext is the immutable public context that prefixes every
// DLinKZG transcript. PriorTranscriptDigest hashes the complete canonical
// outer-protocol prefix ending immediately before U0. SessionNonce is exactly
// 32 bytes; every other byte array is a SHA-256 digest.
type TranscriptContext struct {
	ProtocolVersion       string
	PublicStatementDigest TranscriptDigest
	SRSDigest             TranscriptDigest
	IndexDigest           TranscriptDigest
	PartyManifestDigest   TranscriptDigest
	SessionNonce          [32]byte
	PriorTranscriptDigest TranscriptDigest
}

// U0Message fixes the three partial-polynomial commitments before xi, nu,
// and z_ch are sampled.
type U0Message struct {
	PartialCommitments [3]bn254.G1Affine
}

// U1Message fixes the three link evaluations and the two auxiliary
// commitments before beta is sampled.
type U1Message struct {
	LinkEvaluations   [3]fr.Element
	LinkCommitment    bn254.G1Affine
	LaurentCommitment bn254.G1Affine
}

// U2Message contains, in polynomial order, the evaluations fixed before
// kappa. The first two arrays are for (g_0,g_1,g_2); the latter two are for
// (h_xi,t_0,t_1,S^lin).
type U2Message struct {
	PartialAtBeta        [3]fr.Element
	PartialAtBetaInverse [3]fr.Element
	BatchAtBeta          [4]fr.Element
	BatchAtBetaInverse   [4]fr.Element
}

// U3Message fixes the two same-set quotients and the two directional source
// quotients before the verifier-only delta challenge is sampled.
type U3Message struct {
	WG  bn254.G1Affine
	WL  bn254.G1Affine
	PiZ bn254.G1Affine
	PiY bn254.G1Affine
}

// ChallengeOut is the canonical public challenge record. Counter is the
// smallest counter whose SHA-256 digest represents a field element outside
// the challenge's exclusion set.
type ChallengeOut struct {
	ID      ChallengeID
	Value   fr.Element
	Counter uint32
}

type openingTranscriptStep uint8

const (
	stepU0 openingTranscriptStep = iota
	stepXi
	stepNu
	stepZ
	stepU1
	stepBeta
	stepU2
	stepKappa
	stepU3
	stepDelta
	stepComplete
)

// OpeningTranscript constructs and replays the canonical opening suffix
//
//	U0 -> xi -> nu -> z_ch -> U1 -> beta -> U2 -> kappa -> U3 -> delta.
//
// It is deliberately independent of prover state: a verifier can rebuild the
// same byte string using only the public context and proof messages.
type OpeningTranscript struct {
	outerDomain string
	bytes       []byte
	next        openingTranscriptStep
	zChallenge  fr.Element
}

// NewOpeningTranscript initializes a transcript with the standard DLinKZG
// domain separator.
func NewOpeningTranscript(context TranscriptContext) (*OpeningTranscript, error) {
	return newOpeningTranscript("", context)
}

// NewOpeningTranscriptWithDomain initializes a transcript with an explicit
// suite domain. This is useful when a caller composes DLinKZG into a larger
// protocol and needs an additional outer domain separator.
func NewOpeningTranscriptWithDomain(domain string, context TranscriptContext) (*OpeningTranscript, error) {
	if domain == "" {
		return nil, fmt.Errorf("%w: empty domain", ErrInvalidTranscriptContext)
	}
	return newOpeningTranscript(domain, context)
}

func newOpeningTranscript(outerDomain string, context TranscriptContext) (*OpeningTranscript, error) {
	if err := validateTranscriptContext(context); err != nil {
		return nil, err
	}

	t := &OpeningTranscript{outerDomain: outerDomain, next: stepU0}
	contextPayload := make([]byte, 0, 256)
	contextPayload = appendLengthPrefixed(contextPayload, []byte(OpeningTranscriptDomain))
	contextPayload = appendLengthPrefixed(contextPayload, []byte(outerDomain))
	contextPayload = appendLengthPrefixed(contextPayload, []byte(context.ProtocolVersion))
	contextPayload = appendLengthPrefixed(contextPayload, context.PublicStatementDigest[:])
	contextPayload = appendLengthPrefixed(contextPayload, context.SRSDigest[:])
	contextPayload = appendLengthPrefixed(contextPayload, context.IndexDigest[:])
	contextPayload = appendLengthPrefixed(contextPayload, context.PartyManifestDigest[:])
	contextPayload = appendLengthPrefixed(contextPayload, context.SessionNonce[:])
	contextPayload = appendLengthPrefixed(contextPayload, context.PriorTranscriptDigest[:])
	t.appendRecord("context", contextPayload)
	return t, nil
}

// AppendU0 appends the aggregate U0 record.
func (t *OpeningTranscript) AppendU0(message U0Message) error {
	if err := t.expect(stepU0); err != nil {
		return err
	}
	for i := range message.PartialCommitments {
		if err := validateTranscriptG1(&message.PartialCommitments[i]); err != nil {
			return fmt.Errorf("%w: U0 commitment %d", err, i)
		}
	}
	payload := make([]byte, 0, 3*bn254.SizeOfG1AffineCompressed)
	for i := range message.PartialCommitments {
		payload = appendG1(payload, &message.PartialCommitments[i])
	}
	t.appendRecord("U0", payload)
	t.next = stepXi
	return nil
}

// DeriveXi rejection-samples xi from F* and appends its ChallengeOut record.
func (t *OpeningTranscript) DeriveXi() (ChallengeOut, error) {
	return t.deriveAtStep(stepXi, ChallengeXi, []fr.Element{{}})
}

// DeriveNu rejection-samples nu from F* and appends its ChallengeOut record.
func (t *OpeningTranscript) DeriveNu() (ChallengeOut, error) {
	return t.deriveAtStep(stepNu, ChallengeNu, []fr.Element{{}})
}

// DeriveZChallenge samples the unrestricted source-link point z_ch and
// appends its ChallengeOut record.
func (t *OpeningTranscript) DeriveZChallenge() (ChallengeOut, error) {
	out, err := t.deriveAtStep(stepZ, ChallengeZ, nil)
	if err == nil {
		t.zChallenge = out.Value
	}
	return out, err
}

// AppendU1 appends the aggregate U1 record.
func (t *OpeningTranscript) AppendU1(message U1Message) error {
	if err := t.expect(stepU1); err != nil {
		return err
	}
	for i := range message.LinkEvaluations {
		if err := validateTranscriptField(&message.LinkEvaluations[i]); err != nil {
			return fmt.Errorf("%w: U1 field element %d", err, i)
		}
	}
	if err := validateTranscriptG1(&message.LinkCommitment); err != nil {
		return fmt.Errorf("%w: U1 link commitment", err)
	}
	if err := validateTranscriptG1(&message.LaurentCommitment); err != nil {
		return fmt.Errorf("%w: U1 Laurent commitment", err)
	}
	payload := make([]byte, 0, 3*fr.Bytes+2*bn254.SizeOfG1AffineCompressed)
	for i := range message.LinkEvaluations {
		payload = appendField(payload, &message.LinkEvaluations[i])
	}
	payload = appendG1(payload, &message.LinkCommitment)
	payload = appendG1(payload, &message.LaurentCommitment)
	t.appendRecord("U1", payload)
	t.next = stepBeta
	return nil
}

// DeriveBeta rejection-samples beta outside
// {0,1,-1,z_ch} union {z_ch^-1 when z_ch != 0} and appends its record.
func (t *OpeningTranscript) DeriveBeta() (ChallengeOut, error) {
	if err := t.expect(stepBeta); err != nil {
		return ChallengeOut{}, err
	}
	zero := fr.Element{}
	one := fr.NewElement(1)
	var minusOne fr.Element
	minusOne.Neg(&one)
	forbidden := []fr.Element{zero, one, minusOne, t.zChallenge}
	if !t.zChallenge.IsZero() {
		var inverse fr.Element
		inverse.Inverse(&t.zChallenge)
		forbidden = append(forbidden, inverse)
	}
	return t.deriveAtStep(stepBeta, ChallengeBeta, forbidden)
}

// AppendU2 appends the aggregate U2 record.
func (t *OpeningTranscript) AppendU2(message U2Message) error {
	if err := t.expect(stepU2); err != nil {
		return err
	}
	fieldArrays := [][]fr.Element{
		message.PartialAtBeta[:],
		message.PartialAtBetaInverse[:],
		message.BatchAtBeta[:],
		message.BatchAtBetaInverse[:],
	}
	for arrayIndex := range fieldArrays {
		for elementIndex := range fieldArrays[arrayIndex] {
			if err := validateTranscriptField(&fieldArrays[arrayIndex][elementIndex]); err != nil {
				return fmt.Errorf("%w: U2 field element %d:%d", err, arrayIndex, elementIndex)
			}
		}
	}
	payload := make([]byte, 0, 14*fr.Bytes)
	for i := range message.PartialAtBeta {
		payload = appendField(payload, &message.PartialAtBeta[i])
		payload = appendField(payload, &message.PartialAtBetaInverse[i])
	}
	for i := range message.BatchAtBeta {
		payload = appendField(payload, &message.BatchAtBeta[i])
		payload = appendField(payload, &message.BatchAtBetaInverse[i])
	}
	t.appendRecord("U2", payload)
	t.next = stepKappa
	return nil
}

// DeriveKappa rejection-samples kappa from F* and appends its record.
func (t *OpeningTranscript) DeriveKappa() (ChallengeOut, error) {
	return t.deriveAtStep(stepKappa, ChallengeKappa, []fr.Element{{}})
}

// AppendU3 appends the aggregate U3 record.
func (t *OpeningTranscript) AppendU3(message U3Message) error {
	if err := t.expect(stepU3); err != nil {
		return err
	}
	points := []*bn254.G1Affine{&message.WG, &message.WL, &message.PiZ, &message.PiY}
	for i := range points {
		if err := validateTranscriptG1(points[i]); err != nil {
			return fmt.Errorf("%w: U3 commitment %d", err, i)
		}
	}
	payload := make([]byte, 0, 4*bn254.SizeOfG1AffineCompressed)
	payload = appendG1(payload, &message.WG)
	payload = appendG1(payload, &message.WL)
	payload = appendG1(payload, &message.PiZ)
	payload = appendG1(payload, &message.PiY)
	t.appendRecord("U3", payload)
	t.next = stepDelta
	return nil
}

// DeriveDelta rejection-samples the verifier-only delta from F* and appends
// its record.
func (t *OpeningTranscript) DeriveDelta() (ChallengeOut, error) {
	return t.deriveAtStep(stepDelta, ChallengeDelta, []fr.Element{{}})
}

// Bytes returns a copy of the canonical transcript encoding accumulated so
// far.
func (t *OpeningTranscript) Bytes() []byte {
	if t == nil {
		return nil
	}
	return append([]byte(nil), t.bytes...)
}

// Digest returns the SHA-256 digest of the canonical transcript encoding
// accumulated so far.
func (t *OpeningTranscript) Digest() [sha256.Size]byte {
	if t == nil {
		return sha256.Sum256(nil)
	}
	return sha256.Sum256(t.bytes)
}

func validateTranscriptContext(context TranscriptContext) error {
	if context.ProtocolVersion == "" {
		return fmt.Errorf("%w: empty protocol version", ErrInvalidTranscriptContext)
	}
	requiredDigests := []struct {
		name  string
		value TranscriptDigest
	}{
		{"public statement", context.PublicStatementDigest},
		{"SRS identifier", context.SRSDigest},
		{"index digest", context.IndexDigest},
		{"party manifest", context.PartyManifestDigest},
		{"prior transcript", context.PriorTranscriptDigest},
	}
	for _, field := range requiredDigests {
		if field.value == (TranscriptDigest{}) {
			return fmt.Errorf("%w: zero %s digest", ErrInvalidTranscriptContext, field.name)
		}
	}
	if context.SessionNonce == ([32]byte{}) {
		return fmt.Errorf("%w: zero session nonce", ErrInvalidTranscriptContext)
	}
	return nil
}

func (t *OpeningTranscript) expect(want openingTranscriptStep) error {
	if t == nil {
		return fmt.Errorf("%w: nil transcript", ErrTranscriptOrder)
	}
	if t.next != want {
		return fmt.Errorf("%w: got step %d, want step %d", ErrTranscriptOrder, t.next, want)
	}
	return nil
}

func (t *OpeningTranscript) deriveAtStep(step openingTranscriptStep, id ChallengeID, forbidden []fr.Element) (ChallengeOut, error) {
	if err := t.expect(step); err != nil {
		return ChallengeOut{}, err
	}
	out, err := t.sampleChallenge(id, forbidden)
	if err != nil {
		return ChallengeOut{}, err
	}
	t.appendChallenge(out)
	t.next++
	return out, nil
}

// sampleChallenge implements unbiased hash-to-field followed by exclusion-
// set rejection. A digest is accepted only when its 256-bit integer is below
// the BN254 scalar modulus, so SetBytes performs no modular folding.
func (t *OpeningTranscript) sampleChallenge(id ChallengeID, forbidden []fr.Element) (ChallengeOut, error) {
	for counter := uint32(0); ; counter++ {
		candidate, inField := t.challengeCandidate(id, counter)
		if inField && !fieldElementInSet(candidate, forbidden) {
			return ChallengeOut{ID: id, Value: candidate, Counter: counter}, nil
		}
		if counter == math.MaxUint32 {
			break
		}
	}
	return ChallengeOut{}, ErrChallengeSampling
}

func (t *OpeningTranscript) challengeCandidate(id ChallengeID, counter uint32) (fr.Element, bool) {
	input := make([]byte, 0, len(t.bytes)+128)
	input = appendLengthPrefixed(input, []byte(challengeHashDomain))
	input = appendLengthPrefixed(input, []byte(OpeningTranscriptDomain))
	input = appendLengthPrefixed(input, []byte(t.outerDomain))
	input = appendLengthPrefixed(input, []byte(id))
	input = appendLengthPrefixed(input, t.bytes)
	var counterBytes [4]byte
	binary.BigEndian.PutUint32(counterBytes[:], counter)
	input = append(input, counterBytes[:]...)
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

func (t *OpeningTranscript) appendChallenge(out ChallengeOut) {
	payload := make([]byte, 0, len(out.ID)+4+fr.Bytes+4)
	payload = appendLengthPrefixed(payload, []byte(out.ID))
	var counter [4]byte
	binary.BigEndian.PutUint32(counter[:], out.Counter)
	payload = append(payload, counter[:]...)
	payload = appendField(payload, &out.Value)
	t.appendRecord("ChallengeOut", payload)
}

func (t *OpeningTranscript) appendRecord(label string, payload []byte) {
	t.bytes = appendLengthPrefixed(t.bytes, []byte(label))
	t.bytes = appendLengthPrefixed(t.bytes, payload)
}

func appendLengthPrefixed(dst, value []byte) []byte {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(value)))
	dst = append(dst, size[:]...)
	return append(dst, value...)
}

func appendField(dst []byte, value *fr.Element) []byte {
	encoded := value.Bytes()
	return append(dst, encoded[:]...)
}

func appendG1(dst []byte, value *bn254.G1Affine) []byte {
	encoded := value.Bytes()
	return append(dst, encoded[:]...)
}

func fieldElementInSet(value fr.Element, set []fr.Element) bool {
	for i := range set {
		if value.Equal(&set[i]) {
			return true
		}
	}
	return false
}

func validateTranscriptG1(value *bn254.G1Affine) error {
	if value == nil || !value.IsOnCurve() || !value.IsInSubGroup() {
		return ErrInvalidTranscriptMessage
	}
	return nil
}

func validateTranscriptField(value *fr.Element) error {
	if value == nil {
		return ErrInvalidTranscriptMessage
	}
	encoded := value.Bytes()
	var canonical fr.Element
	canonical.SetBytes(encoded[:])
	if !canonical.Equal(value) {
		return ErrInvalidTranscriptMessage
	}
	return nil
}
