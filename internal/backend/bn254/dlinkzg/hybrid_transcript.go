package dlinkzg

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// HybridOpeningTranscriptDomain separates the degree-aware hybrid terminal
	// compiler from the unified DLinKZG transcript retained for comparison.
	HybridOpeningTranscriptDomain = "DLinKZG/hybrid-opening-transcript/v1"
	hybridChallengeHashDomain     = "DLinKZG/hybrid-hash-to-field/sha256/v1"
)

const (
	ChallengeAlphaChallenge ChallengeID = "alpha_ch"
)

// HybridU0Message fixes the three semantic partial-evaluation commitments.
type HybridU0Message struct {
	PartialCommitments [3]bn254.G1Affine
}

// HybridU1Message fixes g_j(alpha_ch), the canonical h_gamma commitment,
// and the additive functional Laurent-witness commitment before beta.
type HybridU1Message struct {
	PartialAtChallenge   [3]fr.Element
	FunctionalCommitment bn254.G1Affine
	LaurentCommitment    bn254.G1Affine
}

// HybridU2Message fixes the two non-semantic cross-values of each g_j and
// the two-point Laurent batch values in order (h_gamma,H_0,H_1,S^fun).
type HybridU2Message struct {
	CircuitCrossValues   [3][2]fr.Element
	LaurentAtBeta        [4]fr.Element
	LaurentAtBetaInverse [4]fr.Element
}

// HybridU3Message fixes the two ordinary same-set quotients and the two
// directional source-link quotients before verifier-only delta.
type HybridU3Message struct {
	WCirc bn254.G1Affine
	WLaur bn254.G1Affine
	PiU   bn254.G1Affine
	PiV   bn254.G1Affine
}

type hybridOpeningTranscriptStep uint8

const (
	hybridStepU0 hybridOpeningTranscriptStep = iota
	hybridStepGamma
	hybridStepAlphaChallenge
	hybridStepU1
	hybridStepBeta
	hybridStepU2
	hybridStepKappa
	hybridStepU3
	hybridStepDelta
	hybridStepComplete
)

// HybridOpeningTranscript implements
// U0 -> gamma -> alpha_ch -> U1 -> beta -> U2 -> kappa -> U3 -> delta.
type HybridOpeningTranscript struct {
	outerDomain    string
	bytes          []byte
	next           hybridOpeningTranscriptStep
	alphaChallenge fr.Element
}

func NewHybridOpeningTranscript(context TranscriptContext) (*HybridOpeningTranscript, error) {
	if err := validateTranscriptContext(context); err != nil {
		return nil, err
	}
	t := &HybridOpeningTranscript{next: hybridStepU0}
	payload := make([]byte, 0, 256)
	payload = appendLengthPrefixed(payload, []byte(HybridOpeningTranscriptDomain))
	payload = appendLengthPrefixed(payload, []byte(context.ProtocolVersion))
	payload = appendLengthPrefixed(payload, context.PublicStatementDigest[:])
	payload = appendLengthPrefixed(payload, context.SRSDigest[:])
	payload = appendLengthPrefixed(payload, context.IndexDigest[:])
	payload = appendLengthPrefixed(payload, context.PartyManifestDigest[:])
	payload = appendLengthPrefixed(payload, context.SessionNonce[:])
	payload = appendLengthPrefixed(payload, context.PriorTranscriptDigest[:])
	t.appendRecord("context", payload)
	return t, nil
}

func (t *HybridOpeningTranscript) AppendU0(message HybridU0Message) error {
	if err := t.expect(hybridStepU0); err != nil {
		return err
	}
	payload := make([]byte, 0, 3*bn254.SizeOfG1AffineCompressed)
	for i := range message.PartialCommitments {
		if err := validateTranscriptG1(&message.PartialCommitments[i]); err != nil {
			return fmt.Errorf("%w: hybrid U0 commitment %d", err, i)
		}
		payload = appendG1(payload, &message.PartialCommitments[i])
	}
	t.appendRecord("H0", payload)
	t.next = hybridStepGamma
	return nil
}

func (t *HybridOpeningTranscript) DeriveGamma() (ChallengeOut, error) {
	return t.deriveAtStep(hybridStepGamma, ChallengeGamma, []fr.Element{{}})
}

func (t *HybridOpeningTranscript) DeriveAlphaChallenge(semanticPoints [3]fr.Element) (ChallengeOut, error) {
	out, err := t.deriveAtStep(hybridStepAlphaChallenge, ChallengeAlphaChallenge, semanticPoints[:])
	if err == nil {
		t.alphaChallenge = out.Value
	}
	return out, err
}

func (t *HybridOpeningTranscript) AppendU1(message HybridU1Message) error {
	if err := t.expect(hybridStepU1); err != nil {
		return err
	}
	payload := make([]byte, 0, 3*fr.Bytes+2*bn254.SizeOfG1AffineCompressed)
	for i := range message.PartialAtChallenge {
		if err := validateTranscriptField(&message.PartialAtChallenge[i]); err != nil {
			return fmt.Errorf("%w: hybrid U1 field %d", err, i)
		}
		payload = appendField(payload, &message.PartialAtChallenge[i])
	}
	for i, point := range []*bn254.G1Affine{&message.FunctionalCommitment, &message.LaurentCommitment} {
		if err := validateTranscriptG1(point); err != nil {
			return fmt.Errorf("%w: hybrid U1 commitment %d", err, i)
		}
		payload = appendG1(payload, point)
	}
	t.appendRecord("H1", payload)
	t.next = hybridStepBeta
	return nil
}

func (t *HybridOpeningTranscript) DeriveBeta() (ChallengeOut, error) {
	zero := fr.Element{}
	one := fr.One()
	minusOne := one
	minusOne.Neg(&minusOne)
	return t.deriveAtStep(
		hybridStepBeta,
		ChallengeBeta,
		[]fr.Element{zero, one, minusOne, t.alphaChallenge},
	)
}

func (t *HybridOpeningTranscript) AppendU2(message HybridU2Message) error {
	if err := t.expect(hybridStepU2); err != nil {
		return err
	}
	payload := make([]byte, 0, 14*fr.Bytes)
	for claim := range message.CircuitCrossValues {
		for cross := range message.CircuitCrossValues[claim] {
			value := &message.CircuitCrossValues[claim][cross]
			if err := validateTranscriptField(value); err != nil {
				return fmt.Errorf("%w: hybrid U2 circuit %d:%d", err, claim, cross)
			}
			payload = appendField(payload, value)
		}
	}
	for i := range message.LaurentAtBeta {
		if err := validateTranscriptField(&message.LaurentAtBeta[i]); err != nil {
			return fmt.Errorf("%w: hybrid U2 Laurent beta %d", err, i)
		}
		if err := validateTranscriptField(&message.LaurentAtBetaInverse[i]); err != nil {
			return fmt.Errorf("%w: hybrid U2 Laurent beta inverse %d", err, i)
		}
		payload = appendField(payload, &message.LaurentAtBeta[i])
		payload = appendField(payload, &message.LaurentAtBetaInverse[i])
	}
	t.appendRecord("H2", payload)
	t.next = hybridStepKappa
	return nil
}

func (t *HybridOpeningTranscript) DeriveKappa() (ChallengeOut, error) {
	return t.deriveAtStep(hybridStepKappa, ChallengeKappa, []fr.Element{{}})
}

func (t *HybridOpeningTranscript) AppendU3(message HybridU3Message) error {
	if err := t.expect(hybridStepU3); err != nil {
		return err
	}
	payload := make([]byte, 0, 4*bn254.SizeOfG1AffineCompressed)
	points := []*bn254.G1Affine{&message.WCirc, &message.WLaur, &message.PiU, &message.PiV}
	for i, point := range points {
		if err := validateTranscriptG1(point); err != nil {
			return fmt.Errorf("%w: hybrid U3 commitment %d", err, i)
		}
		payload = appendG1(payload, point)
	}
	t.appendRecord("H3", payload)
	t.next = hybridStepDelta
	return nil
}

func (t *HybridOpeningTranscript) DeriveDelta() (ChallengeOut, error) {
	return t.deriveAtStep(hybridStepDelta, ChallengeDelta, []fr.Element{{}})
}

func (t *HybridOpeningTranscript) Bytes() []byte {
	if t == nil {
		return nil
	}
	return append([]byte(nil), t.bytes...)
}

// Digest returns the canonical hash of the transcript prefix accumulated so
// far. It is diagnostic state and never enters a proof.
func (t *HybridOpeningTranscript) Digest() [32]byte {
	return sha256.Sum256(t.Bytes())
}

func (t *HybridOpeningTranscript) expect(want hybridOpeningTranscriptStep) error {
	if t == nil || t.next != want {
		return fmt.Errorf("%w: hybrid got step %d, want %d", ErrTranscriptOrder, t.next, want)
	}
	return nil
}

func (t *HybridOpeningTranscript) deriveAtStep(step hybridOpeningTranscriptStep, id ChallengeID, forbidden []fr.Element) (ChallengeOut, error) {
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

func (t *HybridOpeningTranscript) sampleChallenge(id ChallengeID, forbidden []fr.Element) (ChallengeOut, error) {
	for counter := uint32(0); ; counter++ {
		candidate, inField := t.challengeCandidate(id, counter)
		if inField && !fieldElementInSet(candidate, forbidden) {
			return ChallengeOut{ID: id, Value: candidate, Counter: counter}, nil
		}
		if counter == math.MaxUint32 {
			return ChallengeOut{}, ErrChallengeSampling
		}
	}
}

func (t *HybridOpeningTranscript) challengeCandidate(id ChallengeID, counter uint32) (fr.Element, bool) {
	input := make([]byte, 0, len(t.bytes)+128)
	input = appendLengthPrefixed(input, []byte(hybridChallengeHashDomain))
	input = appendLengthPrefixed(input, []byte(HybridOpeningTranscriptDomain))
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

func (t *HybridOpeningTranscript) appendChallenge(out ChallengeOut) {
	payload := make([]byte, 0, len(out.ID)+4+fr.Bytes+4)
	payload = appendLengthPrefixed(payload, []byte(out.ID))
	var counter [4]byte
	binary.BigEndian.PutUint32(counter[:], out.Counter)
	payload = append(payload, counter[:]...)
	payload = appendField(payload, &out.Value)
	t.appendRecord("ChallengeOut", payload)
}

func (t *HybridOpeningTranscript) appendRecord(label string, payload []byte) {
	t.bytes = appendLengthPrefixed(t.bytes, []byte(label))
	t.bytes = appendLengthPrefixed(t.bytes, payload)
}
