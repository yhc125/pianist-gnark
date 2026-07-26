package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

func TestCorrectnessPrototypeRunsEndToEnd(t *testing.T) {
	input, challenges, srs := prototypeFixture(t)
	result, err := Run(input, challenges, srs)
	if err != nil {
		t.Fatalf("run correctness prototype: %v", err)
	}
	if !result.Verified {
		t.Fatal("honest correctness prototype did not verify")
	}
	if result.ProductionReady {
		t.Fatal("correctness prototype was incorrectly marked production-ready")
	}
	if result.Notice != CorrectnessPrototypeNotice {
		t.Fatalf("unexpected prototype notice: %q", result.Notice)
	}
	if result.Timings.Prove <= 0 || result.Timings.Verify <= 0 {
		t.Fatal("run did not record positive prove and verify timings")
	}
}

func TestCorrectnessPrototypeRunsFourPartitionsWithTrimmedVerifierKey(t *testing.T) {
	input := PrototypeInput{
		T:                     8,
		CircuitShiftedQueries: [terminalCircuitClaims]fr.Element{fieldElement(2), fieldElement(3), fieldElement(5)},
		LocalResiduals:        make([]fr.Element, 4),
		BoundaryFactors:       prototypeElements(2, 3, 5, 0),
		BoundaryNumerators:    make([]fr.Element, 4),
		BoundaryDenominators:  prototypeElements(7, 11, 13, 17),
	}
	product := fieldElement(30)
	input.BoundaryFactors[3].Inverse(&product)
	for j := 0; j < terminalCircuitClaims; j++ {
		input.Sources[j] = make([][]fr.Element, 4)
		for i := range input.Sources[j] {
			input.Sources[j][i] = make([]fr.Element, input.T)
			for coefficient := range input.Sources[j][i] {
				input.Sources[j][i][coefficient].SetUint64(uint64(1 + 37*j + 11*i + coefficient))
			}
		}
	}
	for i := range input.BoundaryFactors {
		input.BoundaryNumerators[i].Mul(&input.BoundaryFactors[i], &input.BoundaryDenominators[i])
	}
	challenges := PrototypeChallenges{
		Theta:      prototypeElements(19, 23),
		SumCheck:   prototypeElements(29, 31),
		Zeta:       fieldElement(37),
		Xi:         fieldElement(41),
		Nu:         fieldElement(43),
		ZChallenge: fieldElement(47),
		Beta:       fieldElement(53),
		Kappa:      fieldElement(59),
		Delta:      fieldElement(61),
	}
	srs, err := cryptodlinkzg.NewMonomialSRS(4, 8, fieldElement(67), fieldElement(71))
	if err != nil {
		t.Fatalf("construct four-partition SRS: %v", err)
	}
	// The paper's verifier key only needs tau_Z^0,...,tau_Z^3 in G2:
	// three G-batch points induce the largest (degree-three) vanishing
	// polynomial. G1 still retains all T local-row powers.
	srs.G2Z = srs.G2Z[:4]
	result, err := Run(input, challenges, srs)
	if err != nil {
		t.Fatalf("run four-partition correctness prototype: %v", err)
	}
	if !result.Verified {
		t.Fatal("four-partition correctness prototype did not verify")
	}
}

func TestCorrectnessPrototypeRejectsTampering(t *testing.T) {
	input, challenges, srs := prototypeFixture(t)
	statement, proof, err := Prove(input, challenges, srs)
	if err != nil {
		t.Fatalf("prove correctness prototype: %v", err)
	}
	if err := Verify(statement, proof, challenges, srs); err != nil {
		t.Fatalf("verify honest correctness prototype: %v", err)
	}

	tamperedSumCheck := proof
	tamperedSumCheck.SumCheck = cloneSumCheckTranscript(proof.SumCheck)
	tamperedSumCheck.SumCheck.Rounds[0][0].Add(
		&tamperedSumCheck.SumCheck.Rounds[0][0],
		fieldElementPointer(1),
	)
	if err := Verify(statement, tamperedSumCheck, challenges, srs); !errors.Is(err, ErrPrototypePIOP) {
		t.Fatalf("tampered SumCheck: got %v, want ErrPrototypePIOP", err)
	}

	tamperedClaim := statement
	tamperedClaim.CircuitClaims[0].Add(&tamperedClaim.CircuitClaims[0], fieldElementPointer(1))
	if err := Verify(tamperedClaim, proof, challenges, srs); !errors.Is(err, ErrPrototypeOpening) {
		t.Fatalf("tampered functional claim: got %v, want ErrPrototypeOpening", err)
	}

	tamperedQuotient := proof
	tamperedQuotient.WG = statement.SourceCommitments[0]
	if err := Verify(statement, tamperedQuotient, challenges, srs); !errors.Is(err, ErrPrototypeOpening) {
		t.Fatalf("tampered quotient commitment: got %v, want ErrPrototypeOpening", err)
	}
}

func TestCorrectnessPrototypeTreePointsAndLaurentConstantReference(t *testing.T) {
	input, challenges, srs := prototypeFixture(t)
	statement, proof, err := Prove(input, challenges, srs)
	if err != nil {
		t.Fatalf("prove correctness prototype: %v", err)
	}

	table, err := BuildProductCheckTable(input.BoundaryFactors)
	if err != nil {
		t.Fatalf("build ProductCheck table: %v", err)
	}
	r := challenges.SumCheck
	fullPoints := [][]fr.Element{
		{r[0], fieldElement(0)},
		{r[0], fieldElement(1)},
		{fieldElement(0), r[0]},
		{fieldElement(1), r[0]},
		{fieldElement(0), fieldElement(1)}, // bitrepr_2(2M-2)=bitrepr_2(2)
	}
	for c := range fullPoints {
		want, evalErr := cryptodlinkzg.MLEEval(table, fullPoints[c])
		if evalErr != nil {
			t.Fatalf("evaluate direct tree point %d: %v", c, evalErr)
		}
		if !want.Equal(&statement.TreeClaims[c]) {
			t.Fatalf("tree point %d differs from direct length-2M MLE", c)
		}
	}

	t0, t1, err := SplitProductCheckTable(table)
	if err != nil {
		t.Fatalf("split ProductCheck table: %v", err)
	}
	rWeights := cryptodlinkzg.EqualityWeights(r)
	var coefficientConstant, transcriptConstant, term fr.Element
	xiPower := fr.One()
	for j := 0; j < terminalCircuitClaims; j++ {
		query, scale, queryErr := referenceTranslatedQuery(input.CircuitShiftedQueries[j], 2)
		if queryErr != nil {
			t.Fatalf("translate query %d: %v", j, queryErr)
		}
		partial := make([]fr.Element, input.T)
		for i := range input.Sources[j] {
			addScaledReference(partial, input.Sources[j][i], rWeights[i])
		}
		queryWeights := cryptodlinkzg.EqualityWeights(query)
		diagonal := dotReference(partial, queryWeights)
		diagonal.Mul(&diagonal, &scale)
		term.Mul(&xiPower, &diagonal)
		coefficientConstant.Add(&coefficientConstant, &term)

		term.Mul(&xiPower, &statement.CircuitClaims[j])
		transcriptConstant.Add(&transcriptConstant, &term)
		linkValue := cryptodlinkzg.Eval(partial, challenges.ZChallenge)
		term.Mul(&xiPower, &linkValue).Mul(&term, &challenges.Nu)
		coefficientConstant.Add(&coefficientConstant, &term)
		term.Mul(&xiPower, &proof.PartialValues[j][0]).Mul(&term, &challenges.Nu)
		transcriptConstant.Add(&transcriptConstant, &term)
		xiPower.Mul(&xiPower, &challenges.Xi)
	}

	aWeights := make([]fr.Element, len(t0))
	bWeights := make([]fr.Element, len(t1))
	for c := range fullPoints {
		partition := fullPoints[c][:1]
		selector := fullPoints[c][1]
		weights := cryptodlinkzg.EqualityWeights(partition)
		oneMinus := fr.One()
		oneMinus.Sub(&oneMinus, &selector)
		var scaleA, scaleB fr.Element
		scaleA.Mul(&xiPower, &oneMinus)
		scaleB.Mul(&xiPower, &selector)
		addScaledReference(aWeights, weights, scaleA)
		addScaledReference(bWeights, weights, scaleB)

		term.Mul(&xiPower, &statement.TreeClaims[c])
		transcriptConstant.Add(&transcriptConstant, &term)
		xiPower.Mul(&xiPower, &challenges.Xi)
	}
	term = dotReference(t0, aWeights)
	coefficientConstant.Add(&coefficientConstant, &term)
	term = dotReference(t1, bWeights)
	coefficientConstant.Add(&coefficientConstant, &term)
	coefficientConstant.Double(&coefficientConstant)
	transcriptConstant.Double(&transcriptConstant)
	if !coefficientConstant.Equal(&transcriptConstant) {
		t.Fatal("Laurent constant from coefficient inner products differs from 2*a_lin")
	}
}

func TestCorrectnessPrototypeRejectsNonUnitBoundaryProduct(t *testing.T) {
	input, challenges, srs := prototypeFixture(t)
	input.BoundaryFactors[1].SetUint64(3)
	input.BoundaryNumerators[1].Mul(&input.BoundaryFactors[1], &input.BoundaryDenominators[1])
	if _, _, err := Prove(input, challenges, srs); !errors.Is(err, ErrPrototypePIOP) {
		t.Fatalf("non-unit boundary product: got %v, want ErrPrototypePIOP", err)
	}
}

func BenchmarkCorrectnessPrototype(b *testing.B) {
	input, challenges, srs := prototypeFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := Run(input, challenges, srs)
		if err != nil {
			b.Fatal(err)
		}
		if !result.Verified {
			b.Fatal("correctness prototype did not verify")
		}
	}
}

func prototypeFixture(t testing.TB) (PrototypeInput, PrototypeChallenges, *cryptodlinkzg.MonomialSRS) {
	t.Helper()
	input := PrototypeInput{
		T: 4,
		Sources: [terminalCircuitClaims][][]fr.Element{
			{prototypeElements(1, 2, 3, 4), prototypeElements(5, 6, 7, 8)},
			{prototypeElements(2, 3, 5, 7), prototypeElements(11, 13, 17, 19)},
			{prototypeElements(23, 29, 31, 37), prototypeElements(41, 43, 47, 53)},
		},
		CircuitShiftedQueries: [terminalCircuitClaims]fr.Element{fieldElement(2), fieldElement(3), fieldElement(5)},
		LocalResiduals:        prototypeElements(0, 0),
		BoundaryFactors:       prototypeElements(2, 0),
		BoundaryNumerators:    prototypeElements(6, 0),
		BoundaryDenominators:  prototypeElements(3, 5),
	}
	input.BoundaryFactors[1].Inverse(&input.BoundaryFactors[0])
	input.BoundaryNumerators[1].Mul(&input.BoundaryFactors[1], &input.BoundaryDenominators[1])

	challenges := PrototypeChallenges{
		Theta:      prototypeElements(7),
		SumCheck:   prototypeElements(11),
		Zeta:       fieldElement(13),
		Xi:         fieldElement(17),
		Nu:         fieldElement(19),
		ZChallenge: fieldElement(23),
		Beta:       fieldElement(29),
		Kappa:      fieldElement(31),
		Delta:      fieldElement(37),
	}
	srs, err := cryptodlinkzg.NewMonomialSRS(2, 4, fieldElement(41), fieldElement(43))
	if err != nil {
		t.Fatalf("construct prototype SRS: %v", err)
	}
	return input, challenges, srs
}

func prototypeElements(values ...uint64) []fr.Element {
	result := make([]fr.Element, len(values))
	for i := range values {
		result[i].SetUint64(values[i])
	}
	return result
}

func referenceTranslatedQuery(shiftedQuery fr.Element, logT int) ([]fr.Element, fr.Element, error) {
	point := make([]fr.Element, logT)
	scale := fr.One()
	power := shiftedQuery
	for k := 0; k < logT; k++ {
		denominator := fr.One()
		denominator.Add(&denominator, &power)
		if denominator.IsZero() {
			return nil, fr.Element{}, ErrInvalidPrototypeChallenge
		}
		point[k].Div(&power, &denominator)
		scale.Mul(&scale, &denominator)
		power.Square(&power)
	}
	return point, scale, nil
}

func addScaledReference(destination, source []fr.Element, scale fr.Element) {
	for i := range source {
		var term fr.Element
		term.Mul(&source[i], &scale)
		destination[i].Add(&destination[i], &term)
	}
}

func dotReference(left, right []fr.Element) fr.Element {
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}
	var result fr.Element
	for i := 0; i < limit; i++ {
		var term fr.Element
		term.Mul(&left[i], &right[i])
		result.Add(&result, &term)
	}
	return result
}
