package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestProductCheckTableHonestAndDeterministic(t *testing.T) {
	factors := []fr.Element{fieldElement(2), fieldElement(3), fieldElement(5)}
	var inverseThirty fr.Element
	inverseThirty.Inverse(fieldElementPointer(30))
	factors = append(factors, inverseThirty)

	table, err := BuildProductCheckTable(factors)
	if err != nil {
		t.Fatalf("build ProductCheck table: %v", err)
	}
	if err := VerifyProductCheckTable(table); err != nil {
		t.Fatalf("verify honest ProductCheck table: %v", err)
	}

	var inverseSix fr.Element
	inverseSix.Inverse(fieldElementPointer(6))
	want := []fr.Element{
		fieldElement(2), fieldElement(3), fieldElement(5), inverseThirty,
		fieldElement(6), inverseSix, fieldElement(1), fieldElement(0),
	}
	assertFieldSliceEqual(t, table, want)

	second, err := BuildProductCheckTable(factors)
	if err != nil {
		t.Fatalf("rebuild ProductCheck table: %v", err)
	}
	assertFieldSliceEqual(t, second, table)

	t0, t1, err := SplitProductCheckTable(table)
	if err != nil {
		t.Fatalf("split ProductCheck table: %v", err)
	}
	assertFieldSliceEqual(t, t0, want[:4])
	assertFieldSliceEqual(t, t1, want[4:])
}

func TestProductCheckTableRejectsTampering(t *testing.T) {
	factors := []fr.Element{fieldElement(2), fieldElement(3)}
	var inverseSix fr.Element
	inverseSix.Inverse(fieldElementPointer(6))
	factors = append(factors, fieldElement(1), inverseSix)

	table, err := BuildProductCheckTable(factors)
	if err != nil {
		t.Fatalf("build ProductCheck table: %v", err)
	}

	internalTamper := append([]fr.Element(nil), table...)
	internalTamper[4].Add(&internalTamper[4], fieldElementPointer(1))
	if err := VerifyProductCheckTable(internalTamper); !errors.Is(err, ErrInvalidProductCheckTable) {
		t.Fatalf("internal-node tampering: got %v, want ErrInvalidProductCheckTable", err)
	}

	rootTamper := append([]fr.Element(nil), table...)
	rootTamper[6].SetUint64(9)
	if err := VerifyProductCheckTable(rootTamper); !errors.Is(err, ErrInvalidProductCheckTable) {
		t.Fatalf("root tampering: got %v, want ErrInvalidProductCheckTable", err)
	}

	// The final entry is an honest canonical filler, not an accepted-transcript
	// constraint in the paper.
	fillerTamper := append([]fr.Element(nil), table...)
	fillerTamper[7].SetUint64(9)
	if err := VerifyProductCheckTable(fillerTamper); err != nil {
		t.Fatalf("unused filler must not affect verification: %v", err)
	}
}

func TestProductCheckNeutralPadding(t *testing.T) {
	logical := []fr.Element{fieldElement(2), fieldElement(3)}
	var inverseSix fr.Element
	inverseSix.Inverse(fieldElementPointer(6))
	logical = append(logical, inverseSix)

	padded, err := PadBoundaryFactors(logical)
	if err != nil {
		t.Fatalf("pad boundary factors: %v", err)
	}
	if len(padded) != 4 {
		t.Fatalf("padded length: got %d, want 4", len(padded))
	}
	assertFieldSliceEqual(t, padded[:3], logical)
	if !padded[3].IsOne() {
		t.Fatal("neutral padding entry is not one")
	}

	table, err := BuildProductCheckTable(padded)
	if err != nil {
		t.Fatalf("build padded ProductCheck table: %v", err)
	}
	if err := VerifyProductCheckTable(table); err != nil {
		t.Fatalf("verify padded ProductCheck table: %v", err)
	}
	if !table[len(table)-1].IsZero() {
		t.Fatal("honest canonical filler is not zero")
	}

	if _, err := BuildProductCheckTable(logical); !errors.Is(err, ErrInvalidProductCheckSize) {
		t.Fatalf("un-padded non-power-of-two input: got %v, want ErrInvalidProductCheckSize", err)
	}
}

func fieldElement(value uint64) fr.Element {
	return fr.NewElement(value)
}

func fieldElementPointer(value uint64) *fr.Element {
	element := fr.NewElement(value)
	return &element
}

func assertFieldSliceEqual(t *testing.T, got, want []fr.Element) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("field slice length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !got[i].Equal(&want[i]) {
			t.Fatalf("field slice mismatch at index %d: got %s, want %s", i, got[i].String(), want[i].String())
		}
	}
}
