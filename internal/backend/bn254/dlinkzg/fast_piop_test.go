package dlinkzg

import (
	"bytes"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func TestFastLocalPIOPMatchesSlowReference(t *testing.T) {
	table, index, challenges := localPIOPTestFixture(t)
	want, err := BuildLocalPIOPRelation(table, index, challenges)
	if err != nil {
		t.Fatalf("slow reference: %v", err)
	}
	got, _, _ := buildFastLocalPIOPForTest(t, table, index, challenges)
	assertFastLocalPIOPRelationsEqual(t, got, want)
	assertFastLocalPIOPTerminalsEqual(t, got, want)
}

func TestFastLocalPIOPMatchesTwoAndFourRankAdapter(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	for _, world := range []int{2, 4} {
		t.Run(adapterWorldName(world), func(t *testing.T) {
			extracted := extractAllLocalPIOPRanks(t, spr, witness, world)
			challenges, slowRelations := buildAllExtractedLocalRelations(t, extracted)
			for rank := range extracted {
				got, w0, w1 := buildFastLocalPIOPForTest(
					t,
					extracted[rank].table,
					extracted[rank].index,
					challenges,
				)
				for wire := 0; wire < LocalWireCount; wire++ {
					assertFastLocalPIOPPolynomialEqual(t, w0.Wires[wire], slowRelations[rank].Wires[wire], "W0 wire")
					assertFastLocalPIOPPolynomialEqual(t, w1.Tags[wire], slowRelations[rank].Tags[wire], "W1 tag")
					assertFastLocalPIOPPolynomialEqual(t, w1.IdentityTags[wire], slowRelations[rank].IdentityTags[wire], "W1 identity tag")
				}
				assertFastLocalPIOPPolynomialEqual(t, w1.Accumulator, slowRelations[rank].Accumulator, "W1 accumulator")
				assertFastLocalPIOPElementEqual(t, w1.EndpointCorrection, slowRelations[rank].EndpointCorrection, "W1 endpoint")
				assertFastLocalPIOPElementEqual(t, w1.BlockFactor, slowRelations[rank].BlockFactor, "W1 rho")
				assertFastLocalPIOPRelationsEqual(t, got, slowRelations[rank])
				assertFastLocalPIOPTerminalsEqual(t, got, slowRelations[rank])
			}
		})
	}
}

func TestFastLocalPIOPRandomizedExactEquality(t *testing.T) {
	for _, size := range []int{8, 16} {
		for seed := int64(1); seed <= 4; seed++ {
			table, index, challenges := fastLocalPIOPRandomFixture(t, size, seed)
			want, err := BuildLocalPIOPRelation(table, index, challenges)
			if err != nil {
				t.Fatalf("size %d seed %d slow reference: %v", size, seed, err)
			}
			got, _, _ := buildFastLocalPIOPForTest(t, table, index, challenges)
			assertFastLocalPIOPRelationsEqual(t, got, want)
			assertFastLocalPIOPTerminalsEqual(t, got, want)
		}
	}
}

func TestFastLocalPIOPReusesPreprocessedFixedColumnsAcrossWitnesses(t *testing.T) {
	firstTable, secondTable, index, challenges := fastLocalPIOPReusableFixedFixtures(t, 16)
	fixed, err := PreprocessFastLocalPIOPFixed(firstTable, index)
	if err != nil {
		t.Fatalf("preprocess fixed columns: %v", err)
	}
	if fixed.DomainSize() != 16 {
		t.Fatalf("fixed domain size = %d, want 16", fixed.DomainSize())
	}

	firstWant, err := BuildLocalPIOPRelation(firstTable, index, challenges)
	if err != nil {
		t.Fatalf("first slow reference: %v", err)
	}
	firstGot := buildFastLocalPIOPWithFixedForTest(t, firstTable, fixed, challenges)
	assertFastLocalPIOPRelationsEqual(t, firstGot, firstWant)

	// Completed relations own their fixed coefficient slices; mutating one
	// cannot poison the setup object reused for the next witness.
	firstGot.Selectors[LocalSelectorM][0].SetZero()
	firstGot.SigmaX[LocalWireA][0].SetZero()

	secondWant, err := BuildLocalPIOPRelation(secondTable, index, challenges)
	if err != nil {
		t.Fatalf("second slow reference: %v", err)
	}
	secondGot := buildFastLocalPIOPWithFixedForTest(t, secondTable, fixed, challenges)
	assertFastLocalPIOPRelationsEqual(t, secondGot, secondWant)
	if equalLocalPolynomials(firstWant.Wires[LocalWireA], secondGot.Wires[LocalWireA]) {
		t.Fatal("two-witness fixture did not change the online witness polynomial")
	}
}

func TestFastLocalPIOPFiatShamirOrderAndW1LambdaIndependence(t *testing.T) {
	table, index, challenges := localPIOPTestFixture(t)

	var nilState *FastLocalPIOPState
	if _, err := nilState.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); !errors.Is(err, ErrFastLocalPIOPStage) {
		t.Fatalf("nil W1 error = %v, want ErrFastLocalPIOPStage", err)
	}
	if _, err := nilState.BuildQuotient(challenges.Lambda); !errors.Is(err, ErrFastLocalPIOPStage) {
		t.Fatalf("nil W2 error = %v, want ErrFastLocalPIOPStage", err)
	}

	first, err := PrepareFastLocalPIOP(table, index)
	if err != nil {
		t.Fatalf("prepare first state: %v", err)
	}
	if first.Stage() != FastLocalPIOPW0Ready {
		t.Fatalf("stage = %d, want W0 ready", first.Stage())
	}
	if _, err := first.BuildQuotient(challenges.Lambda); !errors.Is(err, ErrFastLocalPIOPStage) {
		t.Fatalf("early W2 error = %v, want ErrFastLocalPIOPStage", err)
	}
	w0, err := first.W0()
	if err != nil {
		t.Fatalf("read W0: %v", err)
	}
	w0.Wires[0][0].SetZero()
	w0Again, err := first.W0()
	if err != nil {
		t.Fatalf("read W0 again: %v", err)
	}
	assertFastLocalPIOPPolynomialEqual(t, w0Again.Wires[0], first.relation.Wires[0], "W0 defensive copy")

	w1First, err := first.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma)
	if err != nil {
		t.Fatalf("build first W1: %v", err)
	}
	if first.Stage() != FastLocalPIOPW1Ready {
		t.Fatalf("stage = %d, want W1 ready", first.Stage())
	}
	if _, err := first.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); !errors.Is(err, ErrFastLocalPIOPStage) {
		t.Fatalf("repeated W1 error = %v, want ErrFastLocalPIOPStage", err)
	}
	firstRelation, err := first.BuildQuotient(challenges.Lambda)
	if err != nil {
		t.Fatalf("build first W2: %v", err)
	}
	if first.Stage() != FastLocalPIOPW2Ready {
		t.Fatalf("stage = %d, want W2 ready", first.Stage())
	}
	if _, err := first.BuildQuotient(challenges.Lambda); !errors.Is(err, ErrFastLocalPIOPStage) {
		t.Fatalf("repeated W2 error = %v, want ErrFastLocalPIOPStage", err)
	}
	assertFastLocalPIOPW1Equal(t, w1First, firstRelation)

	second, err := PrepareFastLocalPIOP(table, index)
	if err != nil {
		t.Fatalf("prepare second state: %v", err)
	}
	w1Second, err := second.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma)
	if err != nil {
		t.Fatalf("build second W1: %v", err)
	}
	otherLambda := fr.NewElement(13)
	secondRelation, err := second.BuildQuotient(otherLambda)
	if err != nil {
		t.Fatalf("build second W2: %v", err)
	}
	assertFastLocalPIOPW1SnapshotsEqual(t, w1First, w1Second)
	assertFastLocalPIOPW1Equal(t, w1Second, secondRelation)
	if fastLocalPIOPQuotientsEqual(firstRelation.Quotient, secondRelation.Quotient) {
		t.Fatal("nontrivial fixture produced identical quotients for distinct lambda values")
	}
}

func TestFastLocalPIOPRejectsInvalidInputs(t *testing.T) {
	t.Run("malformed table", func(t *testing.T) {
		table, index, _ := localPIOPTestFixture(t)
		table.Selectors[LocalSelectorC] = table.Selectors[LocalSelectorC][:3]
		if _, err := PrepareFastLocalPIOP(table, index); !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("error = %v, want ErrInvalidLocalPIOPTable", err)
		}
	})

	t.Run("malformed fixed and online split", func(t *testing.T) {
		table, index, _ := localPIOPTestFixture(t)
		malformedFixed := table
		malformedFixed.SigmaX[LocalWireB] = malformedFixed.SigmaX[LocalWireB][:3]
		if _, err := PreprocessFastLocalPIOPFixed(malformedFixed, index); !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("fixed error = %v, want ErrInvalidLocalPIOPTable", err)
		}
		if _, err := PrepareFastLocalPIOPWithFixed(table, nil); !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("nil fixed error = %v, want ErrInvalidLocalPIOPTable", err)
		}
		fixed, err := PreprocessFastLocalPIOPFixed(table, index)
		if err != nil {
			t.Fatalf("preprocess fixed: %v", err)
		}
		malformedOnline := table
		malformedOnline.Wires[LocalWireC] = malformedOnline.Wires[LocalWireC][:3]
		if _, err := PrepareFastLocalPIOPWithFixed(malformedOnline, fixed); !errors.Is(err, ErrInvalidLocalPIOPTable) {
			t.Fatalf("online error = %v, want ErrInvalidLocalPIOPTable", err)
		}
	})

	t.Run("zero tag denominator fails closed", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		var identityConstant fr.Element
		identityConstant.Mul(&challenges.EtaPart, &index.SlotLabel)
		var identityLinear fr.Element
		identityLinear.Mul(&challenges.EtaX, &index.WireCosets[LocalWireA])
		identityConstant.Add(&identityConstant, &identityLinear).Add(&identityConstant, &challenges.Gamma)
		table.Wires[LocalWireA][0].Neg(&identityConstant)
		state, err := PrepareFastLocalPIOP(table, index)
		if err != nil {
			t.Fatalf("prepare: %v", err)
		}
		if _, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); !errors.Is(err, ErrZeroLocalTagDenominator) {
			t.Fatalf("error = %v, want ErrZeroLocalTagDenominator", err)
		}
		if state.Stage() != FastLocalPIOPFailed {
			t.Fatalf("stage = %d, want failed", state.Stage())
		}
		if _, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, fr.NewElement(99)); !errors.Is(err, ErrFastLocalPIOPStage) {
			t.Fatalf("retry error = %v, want ErrFastLocalPIOPStage", err)
		}
	})

	t.Run("nondivisible quotient fails closed", func(t *testing.T) {
		table, index, challenges := localPIOPTestFixture(t)
		table.PublicInput[0].SetOne()
		state, err := PrepareFastLocalPIOP(table, index)
		if err != nil {
			t.Fatalf("prepare: %v", err)
		}
		if _, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); err != nil {
			t.Fatalf("build W1: %v", err)
		}
		if _, err := state.BuildQuotient(challenges.Lambda); !errors.Is(err, ErrLocalQuotientNotDivisible) {
			t.Fatalf("error = %v, want ErrLocalQuotientNotDivisible", err)
		}
		if state.Stage() != FastLocalPIOPFailed {
			t.Fatalf("stage = %d, want failed", state.Stage())
		}
	})

	t.Run("sparse division rejects malformed and remainder", func(t *testing.T) {
		if _, err := fastDivideByLocalVanishing(make([]fr.Element, 3), 4); !errors.Is(err, ErrLocalQuotientNotDivisible) {
			t.Fatalf("short numerator error = %v", err)
		}
		numerator := make([]fr.Element, 16)
		numerator[0].SetOne()
		if _, err := fastDivideByLocalVanishing(numerator, 4); !errors.Is(err, ErrLocalQuotientNotDivisible) {
			t.Fatalf("remainder error = %v", err)
		}
	})

	t.Run("BN254 extended-domain guard", func(t *testing.T) {
		if err := validateFastLocalPIOPDomainSize(FastLocalPIOPMaxDomainSize); err != nil {
			t.Fatalf("maximum domain rejected: %v", err)
		}
		if err := validateFastLocalPIOPDomainSize(FastLocalPIOPMaxDomainSize + 1); !errors.Is(err, ErrFastLocalPIOPDomainTooLarge) {
			t.Fatalf("guard error = %v, want ErrFastLocalPIOPDomainTooLarge", err)
		}
	})
}

func TestFastLocalPIOPLargeDomainSmoke(t *testing.T) {
	const size = 1 << 13
	table, index, challenges := fastLocalPIOPRandomFixture(t, size, 20260726)
	relation, _, _ := buildFastLocalPIOPForTest(t, table, index, challenges)
	for chunk := range relation.Quotient {
		if len(relation.Quotient[chunk]) != size {
			t.Fatalf("quotient chunk %d width = %d, want %d", chunk, len(relation.Quotient[chunk]), size)
		}
	}
	alpha := fastLocalPIOPOutsidePoint(size)
	residual, err := relation.QuotientResidualAt(alpha)
	if err != nil {
		t.Fatalf("terminal quotient residual: %v", err)
	}
	if !residual.IsZero() {
		t.Fatalf("terminal quotient residual = %s, want zero", residual.String())
	}
}

func TestFastLocalPIOPProductionSourceUsesExtendedFFTPath(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	source, err := os.ReadFile(filepath.Join(filepath.Dir(filename), "fast_piop.go"))
	if err != nil {
		t.Fatalf("read production source: %v", err)
	}
	for _, forbidden := range [][]byte{
		[]byte("multiplyLocal" + "Polynomials"),
		[]byte("deriveLocalPIOP" + "Core"),
		[]byte("VerifyLocalPIOP" + "Relation"),
	} {
		if bytes.Contains(source, forbidden) {
			t.Fatalf("production source contains forbidden quadratic/reference call %q", forbidden)
		}
	}
	for _, required := range [][]byte{
		[]byte("extendedDomain := state.fixed.extendedDomain"),
		[]byte(".FFT("),
		[]byte(".FFTInverse("),
		[]byte("fastDivideByLocalVanishing"),
	} {
		if !bytes.Contains(source, required) {
			t.Fatalf("production source is missing structural fast-path marker %q", required)
		}
	}
	if FastLocalPIOPArithmeticNotice == "" {
		t.Fatal("production arithmetic benchmark boundary notice is empty")
	}
}

// This benchmark covers only the production arithmetic slice. Transcript
// hashing, MPI transport, commitments, and PCS work are intentionally external.
func BenchmarkFastLocalPIOPArithmeticSlice(b *testing.B) {
	const size = 1 << 10
	table, index, challenges := fastLocalPIOPRandomFixture(b, size, 99)
	fixed, err := PreprocessFastLocalPIOPFixed(table, index)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ReportMetric(float64(size), "rows/op")
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		state, err := PrepareFastLocalPIOPWithFixed(table, fixed)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); err != nil {
			b.Fatal(err)
		}
		if _, err := state.BuildQuotient(challenges.Lambda); err != nil {
			b.Fatal(err)
		}
	}
}

func buildFastLocalPIOPWithFixedForTest(
	t testing.TB,
	table LocalPIOPTable,
	fixed *FastLocalPIOPFixed,
	challenges LocalPIOPChallenges,
) *LocalPIOPRelation {
	t.Helper()
	state, err := PrepareFastLocalPIOPWithFixed(table, fixed)
	if err != nil {
		t.Fatalf("prepare fast local PIOP with fixed columns: %v", err)
	}
	if _, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma); err != nil {
		t.Fatalf("build fast W1 with fixed columns: %v", err)
	}
	relation, err := state.BuildQuotient(challenges.Lambda)
	if err != nil {
		t.Fatalf("build fast W2 with fixed columns: %v", err)
	}
	return relation
}

func buildFastLocalPIOPForTest(
	t testing.TB,
	table LocalPIOPTable,
	index LocalPIOPIndex,
	challenges LocalPIOPChallenges,
) (*LocalPIOPRelation, FastLocalPIOPW0, FastLocalPIOPW1) {
	t.Helper()
	state, err := PrepareFastLocalPIOP(table, index)
	if err != nil {
		t.Fatalf("prepare fast local PIOP: %v", err)
	}
	w0, err := state.W0()
	if err != nil {
		t.Fatalf("read fast W0: %v", err)
	}
	w1, err := state.BuildAccumulator(challenges.EtaPart, challenges.EtaX, challenges.Gamma)
	if err != nil {
		t.Fatalf("build fast W1: %v", err)
	}
	relation, err := state.BuildQuotient(challenges.Lambda)
	if err != nil {
		t.Fatalf("build fast W2: %v", err)
	}
	return relation, w0, w1
}

func fastLocalPIOPRandomFixture(t testing.TB, size int, seed int64) (LocalPIOPTable, LocalPIOPIndex, LocalPIOPChallenges) {
	t.Helper()
	domain := fft.NewDomain(uint64(size))
	index := LocalPIOPIndex{
		SlotLabel: fr.NewElement(uint64(seed + 17)),
		WireCosets: [LocalWireCount]fr.Element{
			fr.One(),
			domain.FrMultiplicativeGen,
		},
	}
	index.WireCosets[LocalWireC].Square(&domain.FrMultiplicativeGen)
	challenges := LocalPIOPChallenges{
		EtaPart: fr.NewElement(uint64(seed + 31)),
		EtaX:    fr.NewElement(uint64(seed + 47)),
		Gamma:   fr.NewElement(uint64(seed + 61)),
		Lambda:  fr.NewElement(uint64(seed + 79)),
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

	random := rand.New(rand.NewSource(seed))
	x := fr.One()
	for row := 0; row < size; row++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			table.Wires[wire][row] = fr.NewElement(random.Uint64())
			table.SigmaPart[wire][row] = index.SlotLabel
			table.SigmaX[wire][row].Mul(&index.WireCosets[wire], &x)
		}
		for selector := LocalSelectorM; selector < LocalSelectorC; selector++ {
			table.Selectors[selector][row] = fr.NewElement(random.Uint64())
		}
		table.PublicInput[row] = fr.NewElement(random.Uint64())

		var gate, term fr.Element
		term.Mul(&table.Wires[LocalWireA][row], &table.Wires[LocalWireB][row]).
			Mul(&term, &table.Selectors[LocalSelectorM][row])
		gate.Add(&gate, &term)
		term.Mul(&table.Wires[LocalWireA][row], &table.Selectors[LocalSelectorL][row])
		gate.Add(&gate, &term)
		term.Mul(&table.Wires[LocalWireB][row], &table.Selectors[LocalSelectorR][row])
		gate.Add(&gate, &term)
		term.Mul(&table.Wires[LocalWireC][row], &table.Selectors[LocalSelectorO][row])
		gate.Add(&gate, &term).Add(&gate, &table.PublicInput[row])
		table.Selectors[LocalSelectorC][row].Neg(&gate)
		x.Mul(&x, &domain.Generator)
	}
	return table, index, challenges
}

func fastLocalPIOPReusableFixedFixtures(
	t testing.TB,
	size int,
) (LocalPIOPTable, LocalPIOPTable, LocalPIOPIndex, LocalPIOPChallenges) {
	t.Helper()
	domain := fft.NewDomain(uint64(size))
	index := LocalPIOPIndex{
		SlotLabel: fr.NewElement(23),
		WireCosets: [LocalWireCount]fr.Element{
			fr.One(),
			domain.FrMultiplicativeGen,
		},
	}
	index.WireCosets[LocalWireC].Square(&domain.FrMultiplicativeGen)
	challenges := LocalPIOPChallenges{
		EtaPart: fr.NewElement(29),
		EtaX:    fr.NewElement(31),
		Gamma:   fr.NewElement(37),
		Lambda:  fr.NewElement(41),
	}

	var fixedTable LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		fixedTable.SigmaX[wire] = make([]fr.Element, size)
		fixedTable.SigmaPart[wire] = make([]fr.Element, size)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		fixedTable.Selectors[selector] = make([]fr.Element, size)
	}
	x := fr.One()
	minusOne := fr.One()
	minusOne.Neg(&minusOne)
	for row := 0; row < size; row++ {
		fixedTable.Selectors[LocalSelectorM][row].SetOne()
		fixedTable.Selectors[LocalSelectorO][row] = minusOne
		for wire := 0; wire < LocalWireCount; wire++ {
			fixedTable.SigmaPart[wire][row] = index.SlotLabel
			fixedTable.SigmaX[wire][row].Mul(&index.WireCosets[wire], &x)
		}
		x.Mul(&x, &domain.Generator)
	}

	makeWitness := func(seed int64) LocalPIOPTable {
		table := fixedTable
		for wire := 0; wire < LocalWireCount; wire++ {
			table.Wires[wire] = make([]fr.Element, size)
		}
		table.PublicInput = make([]fr.Element, size)
		random := rand.New(rand.NewSource(seed))
		for row := 0; row < size; row++ {
			table.Wires[LocalWireA][row] = fr.NewElement(random.Uint64())
			table.Wires[LocalWireB][row] = fr.NewElement(random.Uint64())
			table.Wires[LocalWireC][row].Mul(
				&table.Wires[LocalWireA][row],
				&table.Wires[LocalWireB][row],
			)
		}
		return table
	}
	return makeWitness(101), makeWitness(202), index, challenges
}

func fastLocalPIOPOutsidePoint(size int) fr.Element {
	for candidate := uint64(2); ; candidate++ {
		point := fr.NewElement(candidate)
		power := powerLocalField(point, size)
		if !power.Equal(localPIOPFieldPointer(1)) {
			return point
		}
	}
}

func assertFastLocalPIOPRelationsEqual(t testing.TB, got, want *LocalPIOPRelation) {
	t.Helper()
	if got.DomainSize != want.DomainSize {
		t.Fatalf("domain size = %d, want %d", got.DomainSize, want.DomainSize)
	}
	assertFastLocalPIOPElementEqual(t, got.Omega, want.Omega, "omega")
	assertFastLocalPIOPElementEqual(t, got.XStar, want.XStar, "x_star")
	assertFastLocalPIOPElementEqual(t, got.Index.SlotLabel, want.Index.SlotLabel, "slot label")
	for wire := 0; wire < LocalWireCount; wire++ {
		assertFastLocalPIOPElementEqual(t, got.Index.WireCosets[wire], want.Index.WireCosets[wire], "wire coset")
		assertFastLocalPIOPPolynomialEqual(t, got.Wires[wire], want.Wires[wire], "wire")
		assertFastLocalPIOPPolynomialEqual(t, got.SigmaX[wire], want.SigmaX[wire], "sigma X")
		assertFastLocalPIOPPolynomialEqual(t, got.SigmaPart[wire], want.SigmaPart[wire], "sigma part")
		assertFastLocalPIOPPolynomialEqual(t, got.Tags[wire], want.Tags[wire], "tag")
		assertFastLocalPIOPPolynomialEqual(t, got.IdentityTags[wire], want.IdentityTags[wire], "identity tag")
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		assertFastLocalPIOPPolynomialEqual(t, got.Selectors[selector], want.Selectors[selector], "selector")
	}
	assertFastLocalPIOPPolynomialEqual(t, got.PublicInput, want.PublicInput, "public input")
	assertFastLocalPIOPPolynomialEqual(t, got.Accumulator, want.Accumulator, "accumulator")
	for chunk := range got.Quotient {
		assertFastLocalPIOPPolynomialEqual(t, got.Quotient[chunk], want.Quotient[chunk], "quotient")
	}
	assertFastLocalPIOPElementEqual(t, got.Challenges.EtaPart, want.Challenges.EtaPart, "eta_part")
	assertFastLocalPIOPElementEqual(t, got.Challenges.EtaX, want.Challenges.EtaX, "eta_X")
	assertFastLocalPIOPElementEqual(t, got.Challenges.Gamma, want.Challenges.Gamma, "gamma")
	assertFastLocalPIOPElementEqual(t, got.Challenges.Lambda, want.Challenges.Lambda, "lambda")
	assertFastLocalPIOPElementEqual(t, got.EndpointCorrection, want.EndpointCorrection, "endpoint correction")
	assertFastLocalPIOPElementEqual(t, got.BlockFactor, want.BlockFactor, "rho")
	assertFastLocalPIOPElementEqual(t, got.BoundaryNumerator, want.BoundaryNumerator, "boundary numerator")
	assertFastLocalPIOPElementEqual(t, got.BoundaryDenominator, want.BoundaryDenominator, "boundary denominator")
}

func assertFastLocalPIOPTerminalsEqual(t testing.TB, got, want *LocalPIOPRelation) {
	t.Helper()
	alpha := fastLocalPIOPOutsidePoint(got.DomainSize)
	gotTerminals, err := got.TerminalEvaluations(alpha)
	if err != nil {
		t.Fatalf("fast terminal evaluations: %v", err)
	}
	wantTerminals, err := want.TerminalEvaluations(alpha)
	if err != nil {
		t.Fatalf("slow terminal evaluations: %v", err)
	}
	gotFlat := gotTerminals.Flatten()
	wantFlat := wantTerminals.Flatten()
	assertFastLocalPIOPPolynomialEqual(t, gotFlat, wantFlat, "terminal evaluation")
}

func assertFastLocalPIOPW1Equal(t testing.TB, w1 FastLocalPIOPW1, relation *LocalPIOPRelation) {
	t.Helper()
	for wire := 0; wire < LocalWireCount; wire++ {
		assertFastLocalPIOPPolynomialEqual(t, w1.Tags[wire], relation.Tags[wire], "stable W1 tag")
		assertFastLocalPIOPPolynomialEqual(t, w1.IdentityTags[wire], relation.IdentityTags[wire], "stable W1 identity tag")
	}
	assertFastLocalPIOPPolynomialEqual(t, w1.Accumulator, relation.Accumulator, "stable W1 accumulator")
	assertFastLocalPIOPElementEqual(t, w1.EndpointCorrection, relation.EndpointCorrection, "stable W1 endpoint")
	assertFastLocalPIOPElementEqual(t, w1.BlockFactor, relation.BlockFactor, "stable W1 rho")
}

func assertFastLocalPIOPW1SnapshotsEqual(t testing.TB, left, right FastLocalPIOPW1) {
	t.Helper()
	assertFastLocalPIOPElementEqual(t, left.EtaPart, right.EtaPart, "W1 eta_part")
	assertFastLocalPIOPElementEqual(t, left.EtaX, right.EtaX, "W1 eta_X")
	assertFastLocalPIOPElementEqual(t, left.Gamma, right.Gamma, "W1 gamma")
	for wire := 0; wire < LocalWireCount; wire++ {
		assertFastLocalPIOPPolynomialEqual(t, left.Tags[wire], right.Tags[wire], "W1 tag")
		assertFastLocalPIOPPolynomialEqual(t, left.IdentityTags[wire], right.IdentityTags[wire], "W1 identity tag")
	}
	assertFastLocalPIOPPolynomialEqual(t, left.Accumulator, right.Accumulator, "W1 accumulator")
	assertFastLocalPIOPElementEqual(t, left.EndpointCorrection, right.EndpointCorrection, "W1 endpoint")
	assertFastLocalPIOPElementEqual(t, left.BlockFactor, right.BlockFactor, "W1 rho")
	assertFastLocalPIOPElementEqual(t, left.BoundaryNumerator, right.BoundaryNumerator, "W1 numerator")
	assertFastLocalPIOPElementEqual(t, left.BoundaryDenominator, right.BoundaryDenominator, "W1 denominator")
}

func assertFastLocalPIOPPolynomialEqual(t testing.TB, got, want []fr.Element, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	for coefficient := range got {
		if !got[coefficient].Equal(&want[coefficient]) {
			t.Fatalf("%s coefficient %d = %s, want %s", label, coefficient, got[coefficient].String(), want[coefficient].String())
		}
	}
}

func assertFastLocalPIOPElementEqual(t testing.TB, got, want fr.Element, label string) {
	t.Helper()
	if !got.Equal(&want) {
		t.Fatalf("%s = %s, want %s", label, got.String(), want.String())
	}
}

func fastLocalPIOPQuotientsEqual(left, right [3][]fr.Element) bool {
	for chunk := range left {
		if !equalLocalPolynomials(left[chunk], right[chunk]) {
			return false
		}
	}
	return true
}
