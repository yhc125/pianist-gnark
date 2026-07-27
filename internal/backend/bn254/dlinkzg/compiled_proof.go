package dlinkzg

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	compiledProofMagic        = "DLKZGPRF"
	compiledProofVersion      = uint16(2)
	compiledProofHeaderSize   = 20
	compiledProofG1Count      = 17
	compiledProofFixedFrCount = 43
)

var (
	// ErrInvalidCompiledProof reports a proof with the wrong public inventory
	// or a message that the corresponding transcript would reject.
	ErrInvalidCompiledProof = errors.New("dlinkzg: invalid compiled proof")
	// ErrCompiledProofEncoding reports a malformed or non-canonical binary
	// representation of a compiled proof.
	ErrCompiledProofEncoding = errors.New("dlinkzg: invalid compiled proof encoding")
)

// CompiledProof is the complete accepted-path public proof. Fiat--Shamir
// challenges and their rejection counters are recomputed by the verifier and
// therefore are not serialized. Retained W3 records, clear party tables, and
// all prover-only state are likewise absent.
//
// For M partitions it contains exactly 17 compressed G1 elements and
// 6*log2(M)+43 scalar-field elements.
type CompiledProof struct {
	W0               OuterW0Message
	W1               OuterW1Message
	W2               OuterW2Message
	ProductCheck     OuterProductCheckCommitmentsMessage
	SumCheckRounds   []OuterSumCheckRoundMessage
	FinalEvaluations OuterFinalEvaluationsMessage
	U0               U0Message
	U1               U1Message
	U2               U2Message
	U3               U3Message
}

// CompiledProofElementCounts returns the exact accepted-path proof inventory
// for a power-of-two partition count M >= 2.
func CompiledProofElementCounts(partitions uint64) (g1Elements, fieldElements int, err error) {
	rounds, err := compiledProofRoundCount(partitions)
	if err != nil {
		return 0, 0, err
	}
	return compiledProofG1Count, compiledProofFixedFrCount + SumCheckRoundCoefficients*rounds, nil
}

// CompiledProofEncodedSize returns the canonical byte length, including the
// fixed magic/version/partition header.
func CompiledProofEncodedSize(partitions uint64) (int, error) {
	g1Elements, fieldElements, err := CompiledProofElementCounts(partitions)
	if err != nil {
		return 0, err
	}
	return compiledProofHeaderSize + g1Elements*bn254.SizeOfG1AffineCompressed + fieldElements*fr.Bytes, nil
}

// MarshalBinary serializes the proof in protocol order. The partition count
// is included in the fixed header as well as supplied to decoding, preventing
// the same byte string from being interpreted with a different SumCheck shape.
func (proof *CompiledProof) MarshalBinary(partitions uint64) ([]byte, error) {
	if proof == nil {
		return nil, fmt.Errorf("%w: nil proof", ErrInvalidCompiledProof)
	}
	rounds, err := compiledProofRoundCount(partitions)
	if err != nil {
		return nil, err
	}
	if len(proof.SumCheckRounds) != rounds {
		return nil, fmt.Errorf("%w: got %d SumCheck rounds, want %d", ErrInvalidCompiledProof, len(proof.SumCheckRounds), rounds)
	}
	if err := validateCompiledProof(proof); err != nil {
		return nil, err
	}

	encodedSize, err := CompiledProofEncodedSize(partitions)
	if err != nil {
		return nil, err
	}
	encoded := make([]byte, compiledProofHeaderSize, encodedSize)
	copy(encoded[:8], compiledProofMagic)
	binary.BigEndian.PutUint16(encoded[8:10], compiledProofVersion)
	// Bytes 10:12 are reserved and canonically zero.
	binary.BigEndian.PutUint64(encoded[12:20], partitions)

	for i := range proof.W0.WitnessCommitments {
		encoded = appendCompiledProofG1(encoded, &proof.W0.WitnessCommitments[i])
	}
	encoded = appendCompiledProofG1(encoded, &proof.W1.AccumulatorCommitment)
	for i := range proof.W2.QuotientCommitments {
		encoded = appendCompiledProofG1(encoded, &proof.W2.QuotientCommitments[i])
	}
	for i := range proof.ProductCheck.Commitments {
		encoded = appendCompiledProofG1(encoded, &proof.ProductCheck.Commitments[i])
	}
	for round := range proof.SumCheckRounds {
		for coefficient := range proof.SumCheckRounds[round].Coefficients {
			encoded = appendCompiledProofField(encoded, &proof.SumCheckRounds[round].Coefficients[coefficient])
		}
	}
	for i := range proof.FinalEvaluations.Terminal {
		encoded = appendCompiledProofField(encoded, &proof.FinalEvaluations.Terminal[i])
	}
	for i := range proof.FinalEvaluations.ProductCheck {
		encoded = appendCompiledProofField(encoded, &proof.FinalEvaluations.ProductCheck[i])
	}
	for i := range proof.U0.PartialCommitments {
		encoded = appendCompiledProofG1(encoded, &proof.U0.PartialCommitments[i])
	}
	for i := range proof.U1.LinkEvaluations {
		encoded = appendCompiledProofField(encoded, &proof.U1.LinkEvaluations[i])
	}
	encoded = appendCompiledProofG1(encoded, &proof.U1.LinkCommitment)
	encoded = appendCompiledProofG1(encoded, &proof.U1.LaurentCommitment)
	for i := range proof.U2.PartialAtBeta {
		encoded = appendCompiledProofField(encoded, &proof.U2.PartialAtBeta[i])
		encoded = appendCompiledProofField(encoded, &proof.U2.PartialAtBetaInverse[i])
	}
	for i := range proof.U2.BatchAtBeta {
		encoded = appendCompiledProofField(encoded, &proof.U2.BatchAtBeta[i])
		encoded = appendCompiledProofField(encoded, &proof.U2.BatchAtBetaInverse[i])
	}
	encoded = appendCompiledProofG1(encoded, &proof.U3.WN)
	encoded = appendCompiledProofG1(encoded, &proof.U3.PiZ)
	encoded = appendCompiledProofG1(encoded, &proof.U3.PiY)

	if len(encoded) != encodedSize {
		return nil, fmt.Errorf("%w: internal size mismatch %d != %d", ErrCompiledProofEncoding, len(encoded), encodedSize)
	}
	return encoded, nil
}

// DecodeCompiledProof parses one canonical proof for the supplied partition
// count. It rejects truncation, trailing data, alternate field encodings, and
// alternate or invalid compressed-point encodings.
func DecodeCompiledProof(encoded []byte, partitions uint64) (*CompiledProof, error) {
	rounds, err := compiledProofRoundCount(partitions)
	if err != nil {
		return nil, err
	}
	expectedSize, err := CompiledProofEncodedSize(partitions)
	if err != nil {
		return nil, err
	}
	if len(encoded) != expectedSize {
		return nil, fmt.Errorf("%w: encoded length %d, want %d", ErrCompiledProofEncoding, len(encoded), expectedSize)
	}
	if !bytes.Equal(encoded[:8], []byte(compiledProofMagic)) {
		return nil, fmt.Errorf("%w: bad magic", ErrCompiledProofEncoding)
	}
	if version := binary.BigEndian.Uint16(encoded[8:10]); version != compiledProofVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCompiledProofEncoding, version)
	}
	if encoded[10] != 0 || encoded[11] != 0 {
		return nil, fmt.Errorf("%w: nonzero reserved header", ErrCompiledProofEncoding)
	}
	if encodedPartitions := binary.BigEndian.Uint64(encoded[12:20]); encodedPartitions != partitions {
		return nil, fmt.Errorf("%w: encoded partition count %d, want %d", ErrCompiledProofEncoding, encodedPartitions, partitions)
	}

	decoder := compiledProofDecoder{encoded: encoded, offset: compiledProofHeaderSize}
	proof := &CompiledProof{SumCheckRounds: make([]OuterSumCheckRoundMessage, rounds)}
	for i := range proof.W0.WitnessCommitments {
		if err := decoder.g1(&proof.W0.WitnessCommitments[i], "W0", i); err != nil {
			return nil, err
		}
	}
	if err := decoder.g1(&proof.W1.AccumulatorCommitment, "W1", 0); err != nil {
		return nil, err
	}
	for i := range proof.W2.QuotientCommitments {
		if err := decoder.g1(&proof.W2.QuotientCommitments[i], "W2", i); err != nil {
			return nil, err
		}
	}
	for i := range proof.ProductCheck.Commitments {
		if err := decoder.g1(&proof.ProductCheck.Commitments[i], "ProductCheck", i); err != nil {
			return nil, err
		}
	}
	for round := range proof.SumCheckRounds {
		for coefficient := range proof.SumCheckRounds[round].Coefficients {
			if err := decoder.field(&proof.SumCheckRounds[round].Coefficients[coefficient], "SumCheck", round*SumCheckRoundCoefficients+coefficient); err != nil {
				return nil, err
			}
		}
	}
	for i := range proof.FinalEvaluations.Terminal {
		if err := decoder.field(&proof.FinalEvaluations.Terminal[i], "final terminal", i); err != nil {
			return nil, err
		}
	}
	for i := range proof.FinalEvaluations.ProductCheck {
		if err := decoder.field(&proof.FinalEvaluations.ProductCheck[i], "final ProductCheck", i); err != nil {
			return nil, err
		}
	}
	for i := range proof.U0.PartialCommitments {
		if err := decoder.g1(&proof.U0.PartialCommitments[i], "U0", i); err != nil {
			return nil, err
		}
	}
	for i := range proof.U1.LinkEvaluations {
		if err := decoder.field(&proof.U1.LinkEvaluations[i], "U1", i); err != nil {
			return nil, err
		}
	}
	if err := decoder.g1(&proof.U1.LinkCommitment, "U1 link", 0); err != nil {
		return nil, err
	}
	if err := decoder.g1(&proof.U1.LaurentCommitment, "U1 Laurent", 0); err != nil {
		return nil, err
	}
	for i := range proof.U2.PartialAtBeta {
		if err := decoder.field(&proof.U2.PartialAtBeta[i], "U2 partial beta", i); err != nil {
			return nil, err
		}
		if err := decoder.field(&proof.U2.PartialAtBetaInverse[i], "U2 partial beta inverse", i); err != nil {
			return nil, err
		}
	}
	for i := range proof.U2.BatchAtBeta {
		if err := decoder.field(&proof.U2.BatchAtBeta[i], "U2 batch beta", i); err != nil {
			return nil, err
		}
		if err := decoder.field(&proof.U2.BatchAtBetaInverse[i], "U2 batch beta inverse", i); err != nil {
			return nil, err
		}
	}
	if err := decoder.g1(&proof.U3.WN, "U3", 0); err != nil {
		return nil, err
	}
	if err := decoder.g1(&proof.U3.PiZ, "U3", 1); err != nil {
		return nil, err
	}
	if err := decoder.g1(&proof.U3.PiY, "U3", 2); err != nil {
		return nil, err
	}
	if decoder.offset != len(encoded) {
		return nil, fmt.Errorf("%w: trailing data", ErrCompiledProofEncoding)
	}
	if err := validateCompiledProof(proof); err != nil {
		return nil, err
	}
	return proof, nil
}

// UnmarshalCompiledProof is the value-returning counterpart of
// DecodeCompiledProof.
func UnmarshalCompiledProof(encoded []byte, partitions uint64) (CompiledProof, error) {
	proof, err := DecodeCompiledProof(encoded, partitions)
	if err != nil {
		return CompiledProof{}, err
	}
	return *proof, nil
}

// UnmarshalBinary replaces proof only after a complete canonical decode.
func (proof *CompiledProof) UnmarshalBinary(encoded []byte, partitions uint64) error {
	if proof == nil {
		return fmt.Errorf("%w: nil destination", ErrInvalidCompiledProof)
	}
	decoded, err := DecodeCompiledProof(encoded, partitions)
	if err != nil {
		return err
	}
	*proof = *decoded
	return nil
}

func compiledProofRoundCount(partitions uint64) (int, error) {
	if partitions < 2 || !isPowerOfTwo64(partitions) {
		return 0, fmt.Errorf("%w: partition count %d is not a power of two >= 2", ErrInvalidCompiledProof, partitions)
	}
	return bits.Len64(partitions) - 1, nil
}

func validateCompiledProof(proof *CompiledProof) error {
	validatePoint := func(name string, value *bn254.G1Affine) error {
		if err := validateTranscriptG1(value); err != nil {
			return fmt.Errorf("%w: %s: %v", ErrInvalidCompiledProof, name, err)
		}
		return nil
	}
	for i := range proof.W0.WitnessCommitments {
		if err := validatePoint(fmt.Sprintf("W0[%d]", i), &proof.W0.WitnessCommitments[i]); err != nil {
			return err
		}
	}
	if err := validatePoint("W1", &proof.W1.AccumulatorCommitment); err != nil {
		return err
	}
	for i := range proof.W2.QuotientCommitments {
		if err := validatePoint(fmt.Sprintf("W2[%d]", i), &proof.W2.QuotientCommitments[i]); err != nil {
			return err
		}
	}
	for i := range proof.ProductCheck.Commitments {
		if err := validatePoint(fmt.Sprintf("ProductCheck[%d]", i), &proof.ProductCheck.Commitments[i]); err != nil {
			return err
		}
	}
	for i := range proof.U0.PartialCommitments {
		if err := validatePoint(fmt.Sprintf("U0[%d]", i), &proof.U0.PartialCommitments[i]); err != nil {
			return err
		}
	}
	remainingPoints := []struct {
		name  string
		value *bn254.G1Affine
	}{
		{"U1.link", &proof.U1.LinkCommitment},
		{"U1.Laurent", &proof.U1.LaurentCommitment},
		{"U3.WN", &proof.U3.WN},
		{"U3.PiZ", &proof.U3.PiZ},
		{"U3.PiY", &proof.U3.PiY},
	}
	for i := range remainingPoints {
		if err := validatePoint(remainingPoints[i].name, remainingPoints[i].value); err != nil {
			return err
		}
	}

	for round := range proof.SumCheckRounds {
		for coefficient := range proof.SumCheckRounds[round].Coefficients {
			if err := validateTranscriptField(&proof.SumCheckRounds[round].Coefficients[coefficient]); err != nil {
				return fmt.Errorf("%w: SumCheck[%d][%d]: %v", ErrInvalidCompiledProof, round, coefficient, err)
			}
		}
	}
	fieldGroups := []struct {
		name   string
		values []fr.Element
	}{
		{"final terminal", proof.FinalEvaluations.Terminal[:]},
		{"final ProductCheck", proof.FinalEvaluations.ProductCheck[:]},
		{"U1 link", proof.U1.LinkEvaluations[:]},
		{"U2 partial beta", proof.U2.PartialAtBeta[:]},
		{"U2 partial beta inverse", proof.U2.PartialAtBetaInverse[:]},
		{"U2 batch beta", proof.U2.BatchAtBeta[:]},
		{"U2 batch beta inverse", proof.U2.BatchAtBetaInverse[:]},
	}
	for group := range fieldGroups {
		for i := range fieldGroups[group].values {
			if err := validateTranscriptField(&fieldGroups[group].values[i]); err != nil {
				return fmt.Errorf("%w: %s field element %d: %v", ErrInvalidCompiledProof, fieldGroups[group].name, i, err)
			}
		}
	}
	return nil
}

func appendCompiledProofG1(destination []byte, value *bn254.G1Affine) []byte {
	encoded := value.Bytes()
	return append(destination, encoded[:]...)
}

func appendCompiledProofField(destination []byte, value *fr.Element) []byte {
	encoded := value.Bytes()
	return append(destination, encoded[:]...)
}

type compiledProofDecoder struct {
	encoded []byte
	offset  int
}

func (decoder *compiledProofDecoder) field(destination *fr.Element, phase string, index int) error {
	fieldBytes := decoder.encoded[decoder.offset : decoder.offset+fr.Bytes]
	destination.SetBytes(fieldBytes)
	canonical := destination.Bytes()
	if !bytes.Equal(canonical[:], fieldBytes) {
		return fmt.Errorf("%w: non-canonical %s field element %d", ErrCompiledProofEncoding, phase, index)
	}
	decoder.offset += fr.Bytes
	return nil
}

func (decoder *compiledProofDecoder) g1(destination *bn254.G1Affine, phase string, index int) error {
	pointBytes := decoder.encoded[decoder.offset : decoder.offset+bn254.SizeOfG1AffineCompressed]
	read, err := destination.SetBytes(pointBytes)
	if err != nil || read != bn254.SizeOfG1AffineCompressed {
		return fmt.Errorf("%w: invalid %s G1 element %d", ErrCompiledProofEncoding, phase, index)
	}
	canonical := destination.Bytes()
	if !bytes.Equal(canonical[:], pointBytes) {
		return fmt.Errorf("%w: non-canonical %s G1 element %d", ErrCompiledProofEncoding, phase, index)
	}
	if err := validateTranscriptG1(destination); err != nil {
		return fmt.Errorf("%w: transcript-invalid %s G1 element %d", ErrCompiledProofEncoding, phase, index)
	}
	decoder.offset += bn254.SizeOfG1AffineCompressed
	return nil
}
