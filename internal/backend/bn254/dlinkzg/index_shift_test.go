package dlinkzg

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func TestIndexShiftDeterministicGoldenAndVerify(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-golden-prefix"))
	domain := fft.NewDomain(4)
	first, err := DeriveIndexShift(pref, 4, domain.Generator)
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveIndexShift(pref, 4, domain.Generator)
	if err != nil {
		t.Fatal(err)
	}
	if first.Counter != second.Counter || !first.Sigma.Equal(&second.Sigma) {
		t.Fatal("index-shift derivation is not deterministic")
	}
	if err := VerifyIndexShift(pref, 4, domain.Generator, first.Counter, first.Sigma); err != nil {
		t.Fatalf("verify derived shift: %v", err)
	}

	encoded := first.Sigma.Bytes()
	const expectedCounter = 10
	const expectedSigma = "27da6d980914029de9ff28ac9968f43c702e400f3900e701d364a3451d40d79f"
	if first.Counter != expectedCounter || hex.EncodeToString(encoded[:]) != expectedSigma {
		t.Fatalf("golden shift mismatch: counter=%d sigma=%s", first.Counter, hex.EncodeToString(encoded[:]))
	}
	if IndexShiftSuiteNotice == "" {
		t.Fatal("benchmark-suite notice must remain explicit")
	}
}

func TestIndexShiftRejectsTamperedShift(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-tamper-prefix"))
	domain := fft.NewDomain(4)
	shift, err := DeriveIndexShift(pref, 4, domain.Generator)
	if err != nil {
		t.Fatal(err)
	}
	tampered := shift.Sigma
	tampered.Add(&tampered, indexShiftFieldPointer(1))
	if err := VerifyIndexShift(pref, 4, domain.Generator, shift.Counter, tampered); !errors.Is(err, ErrIndexShiftMismatch) {
		t.Fatalf("tampered sigma: got %v", err)
	}

	malformed := fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	if err := VerifyIndexShift(pref, 4, domain.Generator, shift.Counter, malformed); !errors.Is(err, ErrInvalidIndexShiftInput) {
		t.Fatalf("noncanonical sigma: got %v", err)
	}
}

func TestIndexShiftRejectsNonminimalCounter(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-nonminimal-prefix"))
	domain := fft.NewDomain(4)
	minimal, err := DeriveIndexShift(pref, 4, domain.Generator)
	if err != nil {
		t.Fatal(err)
	}
	var xStar fr.Element
	xStar.Inverse(&domain.Generator)

	var laterCounter uint32
	var laterSigma fr.Element
	found := false
	for counter := minimal.Counter + 1; counter < minimal.Counter+1024; counter++ {
		candidate, inField := indexShiftCandidate(pref, counter)
		if inField && indexShiftAdmissible(candidate, xStar, 4) {
			laterCounter = counter
			laterSigma = candidate
			found = true
			break
		}
	}
	if !found {
		t.Fatal("failed to find deterministic later admissible counter fixture")
	}
	if err := VerifyIndexShift(pref, 4, domain.Generator, laterCounter, laterSigma); !errors.Is(err, ErrIndexShiftNonMinimal) {
		t.Fatalf("nonminimal counter: got %v", err)
	}
}

func TestIndexShiftHostileMaximumCounterDoesNotDriveVerification(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-hostile-maximum-counter"))
	domain := fft.NewDomain(4)
	canonical, err := DeriveIndexShift(pref, 4, domain.Generator)
	if err != nil {
		t.Fatal(err)
	}
	if canonical.Counter == math.MaxUint32 {
		t.Fatal("unexpected maximum canonical counter fixture")
	}
	if err := VerifyIndexShift(pref, 4, domain.Generator, math.MaxUint32, fr.Element{}); !errors.Is(err, ErrIndexShiftNonMinimal) {
		t.Fatalf("hostile maximum counter: got %v", err)
	}
}

func TestIndexShiftCounterZeroAndBoundaryFixtures(t *testing.T) {
	domain := fft.NewDomain(4)
	var zeroFixture TranscriptDigest
	var zeroShift IndexShift
	found := false
	for nonce := 0; nonce < 1024; nonce++ {
		pref := NewTranscriptDigest([]byte(fmt.Sprintf("index-shift-counter-zero-%d", nonce)))
		shift, err := DeriveIndexShift(pref, 4, domain.Generator)
		if err != nil {
			t.Fatal(err)
		}
		if shift.Counter == 0 {
			zeroFixture = pref
			zeroShift = shift
			found = true
			break
		}
	}
	if !found {
		t.Fatal("failed to find deterministic c_sigma=0 fixture")
	}
	if err := VerifyIndexShift(zeroFixture, 4, domain.Generator, 0, zeroShift.Sigma); err != nil {
		t.Fatalf("verify c_sigma=0 fixture: %v", err)
	}

	var xStar fr.Element
	xStar.Inverse(&domain.Generator)
	boundaryCandidate := xStar
	boundaryCandidate.Sub(&boundaryCandidate, &domain.Generator)
	if indexShiftAdmissible(boundaryCandidate, xStar, 4) {
		t.Fatal("candidate with x_star-sigma in B_T was accepted")
	}
	allowedAtOne := xStar
	allowedAtOne.Sub(&allowedAtOne, indexShiftFieldPointer(1))
	if !indexShiftAdmissible(allowedAtOne, xStar, 4) {
		t.Fatal("candidate with x_star-sigma=1 was rejected even though 1 is excluded from B_T")
	}
}

func TestIndexShiftRejectsInvalidDomainInputs(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-invalid-input"))
	domain := fft.NewDomain(4)
	tests := []struct {
		name  string
		pref  TranscriptDigest
		t     uint64
		omega fr.Element
	}{
		{name: "zero prefix", pref: TranscriptDigest{}, t: 4, omega: domain.Generator},
		{name: "T below four", pref: pref, t: 2, omega: domain.Generator},
		{name: "T not power of two", pref: pref, t: 6, omega: domain.Generator},
		{name: "T overflow", pref: pref, t: 1 << 63, omega: domain.Generator},
		{name: "zero omega", pref: pref, t: 4, omega: fr.Element{}},
		{name: "wrong generator order", pref: pref, t: 4, omega: fr.One()},
		{name: "noncanonical omega", pref: pref, t: 4, omega: fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := DeriveIndexShift(test.pref, test.t, test.omega); !errors.Is(err, ErrInvalidIndexShiftInput) {
				t.Fatalf("derive: got %v", err)
			}
			if err := VerifyIndexShift(test.pref, test.t, test.omega, 0, fr.Element{}); !errors.Is(err, ErrInvalidIndexShiftInput) {
				t.Fatalf("verify: got %v", err)
			}
		})
	}
}

func TestIndexShiftDomainSeparatorSensitivity(t *testing.T) {
	pref := NewTranscriptDigest([]byte("index-shift-domain-separation"))
	for counter := uint32(0); counter < 1024; counter++ {
		standard, standardInField := indexShiftCandidateWithDomain(IndexShiftHashDomain, pref, counter)
		alternate, alternateInField := indexShiftCandidateWithDomain("DLinKZG/index-shift/hash-to-field/sha256/alternate", pref, counter)
		if standardInField != alternateInField || standardInField && !standard.Equal(&alternate) {
			return
		}
	}
	t.Fatal("changing the suite domain did not change any candidate fixture")
}

func TestIndexShiftCounterExhaustionGuard(t *testing.T) {
	domain := fft.NewDomain(4)
	var pref TranscriptDigest
	found := false
	for nonce := 0; nonce < 1024; nonce++ {
		candidatePref := NewTranscriptDigest([]byte(fmt.Sprintf("index-shift-exhaustion-%d", nonce)))
		candidate, inField := indexShiftCandidate(candidatePref, math.MaxUint32)
		var xStar fr.Element
		xStar.Inverse(&domain.Generator)
		if !inField || !indexShiftAdmissible(candidate, xStar, 4) {
			pref = candidatePref
			found = true
			break
		}
	}
	if !found {
		t.Fatal("failed to find deterministic exhaustion fixture")
	}
	if _, err := deriveIndexShiftFromCounter(pref, 4, domain.Generator, math.MaxUint32); !errors.Is(err, ErrIndexShiftSampling) {
		t.Fatalf("counter exhaustion: got %v", err)
	}
}

func TestIndexShiftLargeDomainUsesPowerMembership(t *testing.T) {
	const localDomainSize = uint64(1 << 20)
	pref := NewTranscriptDigest([]byte("index-shift-large-domain"))
	omega := testOuterRootOfUnity(localDomainSize)
	shift, err := DeriveIndexShift(pref, localDomainSize, omega)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyIndexShift(pref, localDomainSize, omega, shift.Counter, shift.Sigma); err != nil {
		t.Fatal(err)
	}
	if !outerInBoundarySet(omega, localDomainSize) {
		t.Fatal("large-domain nonidentity root was not recognized as a B_T member")
	}
}

func indexShiftFieldPointer(value uint64) *fr.Element {
	element := fr.NewElement(value)
	return &element
}
