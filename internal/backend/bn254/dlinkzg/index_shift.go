package dlinkzg

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// IndexShiftHashDomain fixes the benchmark suite's SHA-256 hash-to-field
	// domain. It is part of the canonical index-shift format.
	IndexShiftHashDomain = "DLinKZG/index-shift/hash-to-field/sha256/v1"

	indexShiftLabel = "DLinKZG/sigma"

	// IndexShiftSuiteNotice distinguishes this executable benchmark suite from
	// the paper's abstract HashToField interface.
	IndexShiftSuiteNotice = "BN254/SHA-256 benchmark instantiation of the paper's abstract HashToField(pref_index, DLinKZG/sigma, c) index-shift suite"
)

var (
	ErrInvalidIndexShiftInput = errors.New("dlinkzg: invalid index-shift input")
	ErrIndexShiftSampling     = errors.New("dlinkzg: index-shift rejection counter exhausted")
	ErrIndexShiftMismatch     = errors.New("dlinkzg: supplied index shift does not match its counter")
	ErrIndexShiftNonMinimal   = errors.New("dlinkzg: supplied index-shift counter is not minimal")
)

// IndexShift is the canonical public output (c_sigma,sigma) of the concrete
// BN254/SHA-256 setup suite.
type IndexShift struct {
	Counter uint32
	Sigma   fr.Element
}

// DeriveIndexShift returns the first counter whose unbiased SHA-256 field
// candidate sigma satisfies x_star-sigma notin B_T, where
// B_T=<omega>\{1} and x_star=omega^{-1}.
func DeriveIndexShift(prefIndexDigest TranscriptDigest, localDomainSize uint64, omega fr.Element) (IndexShift, error) {
	if err := validateIndexShiftInputs(prefIndexDigest, localDomainSize, omega); err != nil {
		return IndexShift{}, err
	}
	return deriveIndexShiftFromCounter(prefIndexDigest, localDomainSize, omega, 0)
}

// VerifyIndexShift derives the first admissible pair independently, then
// checks the supplied pair against it. Thus verification establishes exact
// candidate decoding and rejection of every earlier counter without making
// its running time depend on an attacker-controlled cSigma.
func VerifyIndexShift(
	prefIndexDigest TranscriptDigest,
	localDomainSize uint64,
	omega fr.Element,
	cSigma uint32,
	sigma fr.Element,
) error {
	if err := validateIndexShiftInputs(prefIndexDigest, localDomainSize, omega); err != nil {
		return err
	}
	if err := validateTranscriptField(&sigma); err != nil {
		return fmt.Errorf("%w: noncanonical sigma", ErrInvalidIndexShiftInput)
	}
	expected, err := DeriveIndexShift(prefIndexDigest, localDomainSize, omega)
	if err != nil {
		return err
	}
	if expected.Counter < cSigma {
		return fmt.Errorf("%w: counter %d was already admissible", ErrIndexShiftNonMinimal, expected.Counter)
	}
	if expected.Counter != cSigma || !expected.Sigma.Equal(&sigma) {
		return ErrIndexShiftMismatch
	}
	return nil
}

func deriveIndexShiftFromCounter(
	prefIndexDigest TranscriptDigest,
	localDomainSize uint64,
	omega fr.Element,
	start uint32,
) (IndexShift, error) {
	xStar := omega
	xStar.Inverse(&xStar)
	for counter := start; ; counter++ {
		candidate, inField := indexShiftCandidate(prefIndexDigest, counter)
		if inField && indexShiftAdmissible(candidate, xStar, localDomainSize) {
			return IndexShift{Counter: counter, Sigma: candidate}, nil
		}
		if counter == math.MaxUint32 {
			break
		}
	}
	return IndexShift{}, ErrIndexShiftSampling
}

func validateIndexShiftInputs(prefIndexDigest TranscriptDigest, localDomainSize uint64, omega fr.Element) error {
	if prefIndexDigest == (TranscriptDigest{}) {
		return fmt.Errorf("%w: zero pref_index digest", ErrInvalidIndexShiftInput)
	}
	if localDomainSize < 4 || !isPowerOfTwo64(localDomainSize) {
		return fmt.Errorf("%w: T must be a power of two at least four", ErrInvalidIndexShiftInput)
	}
	if localDomainSize > (math.MaxUint64-5)/3 {
		return fmt.Errorf("%w: T overflows the field-size bound", ErrInvalidIndexShiftInput)
	}
	fieldRequirement := new(big.Int).SetUint64(3*localDomainSize + 5)
	if fr.Modulus().Cmp(fieldRequirement) <= 0 {
		return fmt.Errorf("%w: |F| must exceed 3T+5", ErrInvalidIndexShiftInput)
	}
	if err := validateTranscriptField(&omega); err != nil || omega.IsZero() {
		return fmt.Errorf("%w: invalid omega", ErrInvalidIndexShiftInput)
	}
	one := fr.One()
	fullOrder := outerPower(omega, localDomainSize)
	halfOrder := outerPower(omega, localDomainSize/2)
	if !fullOrder.Equal(&one) || halfOrder.Equal(&one) {
		return fmt.Errorf("%w: omega does not have exact order T", ErrInvalidIndexShiftInput)
	}
	return nil
}

func indexShiftCandidate(prefIndexDigest TranscriptDigest, counter uint32) (fr.Element, bool) {
	return indexShiftCandidateWithDomain(IndexShiftHashDomain, prefIndexDigest, counter)
}

func indexShiftCandidateWithDomain(domain string, prefIndexDigest TranscriptDigest, counter uint32) (fr.Element, bool) {
	input := make([]byte, 0, 128)
	input = appendLengthPrefixed(input, []byte(domain))
	input = appendLengthPrefixed(input, []byte(indexShiftLabel))
	input = appendLengthPrefixed(input, prefIndexDigest[:])
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

func indexShiftAdmissible(candidate, xStar fr.Element, localDomainSize uint64) bool {
	translated := xStar
	translated.Sub(&translated, &candidate)
	return !outerInBoundarySet(translated, localDomainSize)
}
