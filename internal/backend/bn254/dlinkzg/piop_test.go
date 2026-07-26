package dlinkzg

import (
	"errors"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func TestLocalPIOPExactAlgebra(t *testing.T) {
	table, index, challenges := localPIOPTestFixture(t)
	relation, err := BuildLocalPIOPRelation(table, index, challenges)
	if err != nil {
		t.Fatalf("build local relation: %v", err)
	}
	if err := VerifyLocalPIOPRelation(relation); err != nil {
		t.Fatalf("verify local relation: %v", err)
	}

	if LocalPIOPAlgebraNotice == "" {
		t.Fatal("the implementation scope notice must remain explicit")
	}
	if relation.DomainSize != 4 {
		t.Fatalf("domain size = %d, want 4", relation.DomainSize)
	}
	if relation.EndpointCorrection.IsZero() {
		t.Fatal("fixture must exercise a nonzero endpoint correction")
	}
	if !relation.BlockFactor.Equal(localPIOPFieldPointer(6)) {
		t.Fatalf("block factor = %s, want 6", relation.BlockFactor.String())
	}
	if err := VerifyLocalBlockFactor(relation.BlockFactor, relation.BoundaryDenominator, relation.BoundaryNumerator); err != nil {
		t.Fatalf("division-free block relation: %v", err)
	}

	// The fixture changes the destination product by 2 on the first row and
	// by 3 on the final row. Hence z=(1,2,2,2), rho=6, and the uncorrected
	// final transition is exactly the nonzero endpoint correction.
	wantZ := []fr.Element{fr.NewElement(1), fr.NewElement(2), fr.NewElement(2), fr.NewElement(2)}
	f := multiplyThreeLocalPolynomials(relation.Tags[0], relation.Tags[1], relation.Tags[2])
	fPrime := multiplyThreeLocalPolynomials(relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2])
	x := fr.One()
	for row := 0; row < relation.DomainSize; row++ {
		zAtRow := evaluateLocalPolynomial(relation.Accumulator, x)
		if !zAtRow.Equal(&wantZ[row]) {
			t.Fatalf("z(omega^%d) = %s, want %s", row, zAtRow.String(), wantZ[row].String())
		}
		var nextPoint fr.Element
		nextPoint.Mul(&relation.Omega, &x)
		zAtNext := evaluateLocalPolynomial(relation.Accumulator, nextPoint)
		fAtRow := evaluateLocalPolynomial(f, x)
		fPrimeAtRow := evaluateLocalPolynomial(fPrime, x)
		var transition, term fr.Element
		transition.Mul(&zAtRow, &fAtRow)
		term.Mul(&zAtNext, &fPrimeAtRow)
		transition.Sub(&transition, &term)

		if row < relation.DomainSize-1 {
			if !transition.IsZero() {
				t.Fatalf("nonfinal transition %d is nonzero", row)
			}
		} else if !transition.Equal(&relation.EndpointCorrection) {
			t.Fatalf("final transition does not equal endpoint correction")
		}
		x = nextPoint
	}

	var fiveTimesDenominator fr.Element
	fiveTimesDenominator.Mul(&relation.BoundaryDenominator, localPIOPFieldPointer(5))
	if !fiveTimesDenominator.Equal(&relation.EndpointCorrection) {
		t.Fatal("endpoint correction must equal (rho-1)v in the fixture")
	}

	alpha := fr.NewElement(5)
	residual, err := relation.QuotientResidualAt(alpha)
	if err != nil {
		t.Fatalf("evaluate local quotient residual: %v", err)
	}
	if !residual.IsZero() {
		t.Fatalf("Delta(alpha) = %s, want zero", residual.String())
	}

	// Check the three-chunk representation independently at alpha.
	core := deriveLocalPIOPCore(relation)
	numeratorAtAlpha := evaluateLocalPolynomial(core.numerator, alpha)
	hAlpha := combineLocalQuotient(relation.Quotient, alpha, relation.DomainSize)
	hAtAlpha := evaluateLocalPolynomial(hAlpha, alpha)
	alphaT := powerLocalField(alpha, relation.DomainSize)
	var vanishing, reconstructed fr.Element
	vanishing.Sub(&alphaT, localPIOPFieldPointer(1))
	reconstructed.Mul(&vanishing, &hAtAlpha)
	if !numeratorAtAlpha.Equal(&reconstructed) {
		t.Fatal("three quotient chunks do not reconstruct the corrected numerator")
	}

	// This fixture has a nonconstant z and full-width tag interpolation; its
	// third quotient chunk is intentionally nonzero.
	if localPolynomialIsZero(relation.Quotient[2]) {
		t.Fatal("fixture did not exercise the third quotient chunk")
	}
}

func TestLocalPIOPTerminalOrderAndSourceCompression(t *testing.T) {
	table, index, challenges := localPIOPTestFixture(t)
	relation, err := BuildLocalPIOPRelation(table, index, challenges)
	if err != nil {
		t.Fatalf("build local relation: %v", err)
	}
	alpha := fr.NewElement(5)
	terminals, err := relation.TerminalEvaluations(alpha)
	if err != nil {
		t.Fatalf("terminal evaluations: %v", err)
	}

	wantAlphaOrder := []LocalTerminalFamily{
		LocalTerminalPhiA, LocalTerminalPhiB, LocalTerminalPhiC,
		LocalTerminalPhiPrimeA, LocalTerminalPhiPrimeB, LocalTerminalPhiPrimeC,
		LocalTerminalZ,
		LocalTerminalQM, LocalTerminalQL, LocalTerminalQR, LocalTerminalQO, LocalTerminalQC,
		LocalTerminalHAlpha,
	}
	wantOmegaOrder := []LocalTerminalFamily{LocalTerminalZ}
	wantXStarOrder := []LocalTerminalFamily{
		LocalTerminalPhiA, LocalTerminalPhiB, LocalTerminalPhiC,
		LocalTerminalPhiPrimeA, LocalTerminalPhiPrimeB, LocalTerminalPhiPrimeC,
		LocalTerminalZ,
	}
	if got := LocalTerminalFamilyOrder(LocalTerminalAtAlpha); !reflect.DeepEqual(got, wantAlphaOrder) {
		t.Fatalf("alpha family order = %v", got)
	}
	if got := LocalTerminalFamilyOrder(LocalTerminalAtOmegaAlpha); !reflect.DeepEqual(got, wantOmegaOrder) {
		t.Fatalf("omega-alpha family order = %v", got)
	}
	if got := LocalTerminalFamilyOrder(LocalTerminalAtXStar); !reflect.DeepEqual(got, wantXStarOrder) {
		t.Fatalf("x-star family order = %v", got)
	}
	if got := len(terminals.Flatten()); got != LocalTerminalTotalCount {
		t.Fatalf("flattened terminal count = %d, want %d", got, LocalTerminalTotalCount)
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		assertLocalFieldEqual(t, terminals.Alpha[int(AlphaPhiA)+wire], evaluateLocalPolynomial(relation.Tags[wire], alpha), "alpha destination tag")
		assertLocalFieldEqual(t, terminals.Alpha[int(AlphaPhiPrimeA)+wire], evaluateLocalPolynomial(relation.IdentityTags[wire], alpha), "alpha identity tag")
		assertLocalFieldEqual(t, terminals.XStar[int(XStarPhiA)+wire], evaluateLocalPolynomial(relation.Tags[wire], relation.XStar), "x-star destination tag")
		assertLocalFieldEqual(t, terminals.XStar[int(XStarPhiPrimeA)+wire], evaluateLocalPolynomial(relation.IdentityTags[wire], relation.XStar), "x-star identity tag")
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		assertLocalFieldEqual(t, terminals.Alpha[int(AlphaQM)+selector], evaluateLocalPolynomial(relation.Selectors[selector], alpha), "selector")
	}
	var omegaAlpha fr.Element
	omegaAlpha.Mul(&relation.Omega, &alpha)
	assertLocalFieldEqual(t, terminals.Alpha[AlphaZ], evaluateLocalPolynomial(relation.Accumulator, alpha), "z(alpha)")
	assertLocalFieldEqual(t, terminals.OmegaAlpha[0], evaluateLocalPolynomial(relation.Accumulator, omegaAlpha), "z(omega alpha)")
	assertLocalFieldEqual(t, terminals.XStar[XStarZ], evaluateLocalPolynomial(relation.Accumulator, relation.XStar), "z(x-star)")

	mus := [LocalCompressedSourceCount]fr.Element{fr.NewElement(2), fr.NewElement(3), fr.NewElement(5)}
	compression, err := CompressLocalTerminalSources(relation, alpha, mus)
	if err != nil {
		t.Fatalf("compress terminal sources: %v", err)
	}
	if err := VerifyLocalTerminalSourceCompression(relation, alpha, mus, compression); err != nil {
		t.Fatalf("verify terminal source compression: %v", err)
	}

	// Rebuild every group from an independently written ordered list. This
	// catches accidental movement of h^(alpha), selectors, or z.
	hAlpha := combineLocalQuotient(relation.Quotient, alpha, relation.DomainSize)
	wantGroups := [LocalCompressedSourceCount][][]fr.Element{
		{
			relation.Tags[0], relation.Tags[1], relation.Tags[2],
			relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2],
			relation.Accumulator,
			relation.Selectors[0], relation.Selectors[1], relation.Selectors[2], relation.Selectors[3], relation.Selectors[4],
			hAlpha,
		},
		{relation.Accumulator},
		{
			relation.Tags[0], relation.Tags[1], relation.Tags[2],
			relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2],
			relation.Accumulator,
		},
	}
	for group := 0; group < LocalCompressedSourceCount; group++ {
		wantPolynomial := make([]fr.Element, relation.DomainSize)
		power := fr.One()
		for family := range wantGroups[group] {
			addScaledLocalPolynomial(wantPolynomial, wantGroups[group][family], power)
			power.Mul(&power, &mus[group])
		}
		if !equalLocalPolynomials(wantPolynomial, compression.Polynomials[group]) {
			t.Fatalf("compressed source group %d has the wrong polynomial order", group)
		}
		opened := evaluateLocalPolynomial(compression.Polynomials[group], compression.Points[group])
		if !opened.Equal(&compression.Values[group]) {
			t.Fatalf("compressed source group %d has the wrong value", group)
		}
	}
}

func TestLocalPIOPTamperRejection(t *testing.T) {
	tests := []struct {
		name   string
		tamper func(*LocalPIOPRelation)
	}{
		{
			name: "destination tag",
			tamper: func(relation *LocalPIOPRelation) {
				relation.Tags[0][0].Add(&relation.Tags[0][0], localPIOPFieldPointer(1))
			},
		},
		{
			name: "identity tag",
			tamper: func(relation *LocalPIOPRelation) {
				relation.IdentityTags[1][1].Add(&relation.IdentityTags[1][1], localPIOPFieldPointer(1))
			},
		},
		{
			name: "accumulator",
			tamper: func(relation *LocalPIOPRelation) {
				relation.Accumulator[1].Add(&relation.Accumulator[1], localPIOPFieldPointer(1))
			},
		},
		{
			name: "endpoint correction",
			tamper: func(relation *LocalPIOPRelation) {
				relation.EndpointCorrection.Add(&relation.EndpointCorrection, localPIOPFieldPointer(1))
			},
		},
		{
			name: "block factor",
			tamper: func(relation *LocalPIOPRelation) {
				relation.BlockFactor.Add(&relation.BlockFactor, localPIOPFieldPointer(1))
			},
		},
		{
			name: "boundary numerator",
			tamper: func(relation *LocalPIOPRelation) {
				relation.BoundaryNumerator.Add(&relation.BoundaryNumerator, localPIOPFieldPointer(1))
			},
		},
		{
			name: "quotient chunk",
			tamper: func(relation *LocalPIOPRelation) {
				relation.Quotient[2][0].Add(&relation.Quotient[2][0], localPIOPFieldPointer(1))
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			table, index, challenges := localPIOPTestFixture(t)
			relation, err := BuildLocalPIOPRelation(table, index, challenges)
			if err != nil {
				t.Fatalf("build local relation: %v", err)
			}
			test.tamper(relation)
			if err := VerifyLocalPIOPRelation(relation); err == nil {
				t.Fatal("tampered relation was accepted")
			}
		})
	}
}

func TestLocalPIOPTerminalAndCompressionTamper(t *testing.T) {
	table, index, challenges := localPIOPTestFixture(t)
	relation, err := BuildLocalPIOPRelation(table, index, challenges)
	if err != nil {
		t.Fatalf("build local relation: %v", err)
	}
	alpha := fr.NewElement(5)
	terminals, err := relation.TerminalEvaluations(alpha)
	if err != nil {
		t.Fatalf("terminal evaluations: %v", err)
	}
	context := LocalQuotientEvaluationContext{
		DomainSize:         relation.DomainSize,
		Omega:              relation.Omega,
		XStar:              relation.XStar,
		Index:              relation.Index,
		Challenges:         relation.Challenges,
		PublicInputAtAlpha: evaluateLocalPolynomial(relation.PublicInput, alpha),
	}

	terminals.Alpha[AlphaQC].Add(&terminals.Alpha[AlphaQC], localPIOPFieldPointer(1))
	residual, err := EvaluateLocalQuotientResidual(terminals, alpha, context)
	if err != nil {
		t.Fatalf("evaluate tampered residual: %v", err)
	}
	if residual.IsZero() {
		t.Fatal("tampered q_C terminal value left Delta(alpha) equal to zero")
	}

	mus := [LocalCompressedSourceCount]fr.Element{fr.NewElement(2), fr.NewElement(3), fr.NewElement(5)}
	compression, err := CompressLocalTerminalSources(relation, alpha, mus)
	if err != nil {
		t.Fatalf("compress terminal sources: %v", err)
	}
	compression.Values[0].Add(&compression.Values[0], localPIOPFieldPointer(1))
	if err := VerifyLocalTerminalSourceCompression(relation, alpha, mus, compression); err == nil {
		t.Fatal("tampered compressed value was accepted")
	}

	compression, err = CompressLocalTerminalSources(relation, alpha, mus)
	if err != nil {
		t.Fatalf("compress terminal sources: %v", err)
	}
	compression.Polynomials[2][0].Add(&compression.Polynomials[2][0], localPIOPFieldPointer(1))
	if err := VerifyLocalTerminalSourceCompression(relation, alpha, mus, compression); err == nil {
		t.Fatal("tampered compressed polynomial was accepted")
	}
}

func TestLocalPIOPInvalidInputs(t *testing.T) {
	t.Run("unsatisfied gate is not divisible", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		table.PublicInput[0].SetOne()
		_, err := BuildLocalPIOPRelation(table, index, challenges)
		if !errors.Is(err, ErrLocalQuotientNotDivisible) {
			t.Fatalf("error = %v, want ErrLocalQuotientNotDivisible", err)
		}
	})

	t.Run("zero identity-tag denominator aborts honest construction", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		var identityConstant fr.Element
		identityConstant.Mul(&challenges.EtaPart, &index.SlotLabel)
		var identityLinear fr.Element
		identityLinear.Mul(&challenges.EtaX, &index.WireCosets[LocalWireA])
		identityConstant.Add(&identityConstant, &identityLinear).Add(&identityConstant, &challenges.Gamma)
		table.Wires[LocalWireA][0].Neg(&identityConstant)
		_, err := BuildLocalPIOPRelation(table, index, challenges)
		if !errors.Is(err, ErrZeroLocalTagDenominator) {
			t.Fatalf("error = %v, want ErrZeroLocalTagDenominator", err)
		}
	})

	t.Run("malformed table", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		table.Selectors[LocalSelectorC] = table.Selectors[LocalSelectorC][:3]
		_, err := BuildLocalPIOPRelation(table, index, challenges)
		if !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("error = %v, want ErrInvalidLocalPIOPTable", err)
		}
	})

	t.Run("duplicate wire coset", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		index.WireCosets[2] = index.WireCosets[1]
		_, err := BuildLocalPIOPRelation(table, index, challenges)
		if !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("error = %v, want ErrInvalidLocalPIOPTable", err)
		}
	})

	t.Run("distinct representatives of the same wire coset", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		index.WireCosets[2].Neg(&index.WireCosets[1]) // -1 belongs to H_X for T=4.
		_, err := BuildLocalPIOPRelation(table, index, challenges)
		if !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("error = %v, want ErrInvalidLocalPIOPTable", err)
		}
	})

	t.Run("terminal point rejection", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		relation, err := BuildLocalPIOPRelation(table, index, challenges)
		if err != nil {
			t.Fatalf("build local relation: %v", err)
		}
		if _, err := relation.TerminalEvaluations(fr.Element{}); !errors.Is(err, ErrInvalidLocalTerminalPoint) {
			t.Fatalf("zero alpha error = %v", err)
		}
		if _, err := relation.TerminalEvaluations(fr.One()); !errors.Is(err, ErrInvalidLocalTerminalPoint) {
			t.Fatalf("domain alpha error = %v", err)
		}
	})
}

func TestDivisionFreeBlockFactorRelation(t *testing.T) {
	// The relation itself contains no inverse: on the exceptional 0/0 branch
	// every rho satisfies rho*0=0. Honest construction rejects this branch,
	// while the PIOP equation remains a polynomial identity.
	if err := VerifyLocalBlockFactor(fr.NewElement(19), fr.Element{}, fr.Element{}); err != nil {
		t.Fatalf("division-free zero branch: %v", err)
	}
	if err := VerifyLocalBlockFactor(fr.NewElement(19), fr.Element{}, fr.One()); !errors.Is(err, ErrInvalidLocalBlockFactor) {
		t.Fatalf("error = %v, want ErrInvalidLocalBlockFactor", err)
	}
}

func localPIOPTestFixture(t *testing.T) (LocalPIOPTable, LocalPIOPIndex, LocalPIOPChallenges) {
	t.Helper()
	const size = 4
	domain := fft.NewDomain(size)
	index := LocalPIOPIndex{
		SlotLabel: fr.NewElement(9),
		WireCosets: [LocalWireCount]fr.Element{
			fr.NewElement(2), fr.NewElement(3), fr.NewElement(4),
		},
	}
	challenges := LocalPIOPChallenges{
		EtaPart: fr.NewElement(1),
		EtaX:    fr.NewElement(1),
		Gamma:   fr.NewElement(7),
		Lambda:  fr.NewElement(11),
	}
	var table LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		table.Wires[wire] = make([]fr.Element, size)
		table.SigmaX[wire] = make([]fr.Element, size)
		table.SigmaPart[wire] = make([]fr.Element, size)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		table.Selectors[selector] = make([]fr.Element, size)
	}
	table.PublicInput = make([]fr.Element, size)

	x := fr.One()
	for row := 0; row < size; row++ {
		table.Wires[LocalWireA][row] = powerLocalField(x, size-1)
		table.Wires[LocalWireB][row] = x
		table.Wires[LocalWireC][row].SetOne()
		table.Selectors[LocalSelectorM][row].SetOne()
		table.Selectors[LocalSelectorO][row].SetOne().Neg(&table.Selectors[LocalSelectorO][row])
		for wire := 0; wire < LocalWireCount; wire++ {
			table.SigmaX[wire][row].Mul(&index.WireCosets[wire], &x)
			table.SigmaPart[wire][row] = index.SlotLabel
		}
		x.Mul(&x, &domain.Generator)
	}

	// q_M*a*b+q_O*c = X^T-1: it vanishes on the table but gives a
	// nonzero quotient. Modify phi_a/f'_a by factors 2 and 3 on the first
	// and final rows, respectively. This yields z=(1,2,2,2), rho=6, and a
	// nonzero endpoint correction while retaining all nonfinal recurrences.
	for _, adjustment := range []struct {
		wire  LocalWire
		row   int
		extra fr.Element
	}{
		{wire: LocalWireA, row: 0, extra: fr.NewElement(1)},
		{wire: LocalWireA, row: size - 1, extra: fr.NewElement(2)},
	} {
		identityTag := table.Wires[adjustment.wire][adjustment.row]
		var term fr.Element
		term.Mul(&challenges.EtaPart, &index.SlotLabel)
		identityTag.Add(&identityTag, &term)
		term.Mul(&challenges.EtaX, &table.SigmaX[adjustment.wire][adjustment.row])
		identityTag.Add(&identityTag, &term).Add(&identityTag, &challenges.Gamma)
		term.Mul(&identityTag, &adjustment.extra)
		table.SigmaPart[adjustment.wire][adjustment.row].Add(&table.SigmaPart[adjustment.wire][adjustment.row], &term)
	}

	// Give phi_b and phi_c full-width interpolants without changing their
	// product on the domain: at row 1 multiply them by 2 and 1/2. This leaves
	// the accumulator and rho above unchanged while exercising h_2.
	half := fr.NewElement(2)
	half.Inverse(&half)
	minusHalf := half
	minusHalf.Neg(&minusHalf)
	for _, adjustment := range []struct {
		wire  LocalWire
		extra fr.Element
	}{
		{wire: LocalWireB, extra: fr.NewElement(1)},
		{wire: LocalWireC, extra: minusHalf},
	} {
		const row = 1
		identityTag := table.Wires[adjustment.wire][row]
		var term fr.Element
		term.Mul(&challenges.EtaPart, &index.SlotLabel)
		identityTag.Add(&identityTag, &term)
		term.Mul(&challenges.EtaX, &table.SigmaX[adjustment.wire][row])
		identityTag.Add(&identityTag, &term).Add(&identityTag, &challenges.Gamma)
		term.Mul(&identityTag, &adjustment.extra)
		table.SigmaPart[adjustment.wire][row].Add(&table.SigmaPart[adjustment.wire][row], &term)
	}

	// Guard the fixture itself: every identity product must be nonzero.
	for row := 0; row < size; row++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			identityTag := table.Wires[wire][row]
			var term fr.Element
			term.Mul(&challenges.EtaPart, &index.SlotLabel)
			identityTag.Add(&identityTag, &term)
			term.Mul(&challenges.EtaX, &table.SigmaX[wire][row])
			identityTag.Add(&identityTag, &term).Add(&identityTag, &challenges.Gamma)
			if identityTag.IsZero() {
				t.Fatalf("fixture identity tag is zero at row %d wire %d", row, wire)
			}
		}
	}
	return table, index, challenges
}

func assertLocalFieldEqual(t *testing.T, got, want fr.Element, label string) {
	t.Helper()
	if !got.Equal(&want) {
		t.Fatalf("%s = %s, want %s", label, got.String(), want.String())
	}
}

func localPolynomialIsZero(polynomial []fr.Element) bool {
	for coefficient := range polynomial {
		if !polynomial[coefficient].IsZero() {
			return false
		}
	}
	return true
}

func localPIOPFieldPointer(value uint64) *fr.Element {
	element := fr.NewElement(value)
	return &element
}
