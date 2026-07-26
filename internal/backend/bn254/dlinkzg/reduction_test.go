package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type outerPIOPReductionFixture struct {
	records          []OuterPIOPLocalRecord
	terminals        []LocalTerminalEvaluations
	publicInputs     []fr.Element
	reductionContext OuterPIOPReductionContext
	state            *OuterPIOPCoordinatorState
	instance         *OuterPIOPSumCheckInstance
	proof            OuterPIOPSumCheckProof
	context          OuterPIOPEndpointContext
	piAtR            fr.Element
}

func TestOuterPIOPReductionFromSparseR1CSAdapter(t *testing.T) {
	for _, world := range []int{2, 4} {
		t.Run(adapterWorldName(world), func(t *testing.T) {
			fixture := buildOuterPIOPReductionFixture(t, world)
			if fixture.state.Notice() != OuterPIOPReductionNotice || fixture.instance.Notice() != OuterPIOPReductionNotice {
				t.Fatal("coordinator/prover scope notice is not explicit")
			}

			residuals := fixture.state.LocalResiduals()
			if len(residuals) != world {
				t.Fatalf("local residual count = %d, want %d", len(residuals), world)
			}
			for rank := range residuals {
				if !residuals[rank].IsZero() {
					t.Fatalf("rank %d local residual = %s, want zero", rank, residuals[rank].String())
				}
			}

			table, t0, t1 := fixture.state.CoordinatorProductCheckWitness()
			if len(table) != 2*world || len(t0) != world || len(t1) != world {
				t.Fatalf("ProductCheck sizes = (%d,%d,%d), want (%d,%d,%d)", len(table), len(t0), len(t1), 2*world, world, world)
			}
			if err := VerifyProductCheckTable(table); err != nil {
				t.Fatalf("ProductCheck witness: %v", err)
			}

			oracles, composition := fixture.instance.CoordinatorOracleTables()
			if len(oracles) != outerOracleCount || composition == nil {
				t.Fatalf("SumCheck oracle inventory = (%d,%v), want (%d,non-nil)", len(oracles), composition != nil, outerOracleCount)
			}
			for oracle := range oracles {
				if len(oracles[oracle]) != world {
					t.Fatalf("oracle %d length = %d, want %d", oracle, len(oracles[oracle]), world)
				}
			}
			if claim := sumCompositionTable(oracles, composition); !claim.IsZero() {
				t.Fatalf("SumCheck Boolean claim = %s, want zero", claim.String())
			}

			assertOuterPIOPCircuitFolds(t, fixture.records, fixture.context.Point, fixture.proof.FoldedCircuit)
			assertOuterPIOPTreeClaims(t, table, fixture.context.Point, fixture.proof.TreeEvaluations)

			publicInputs := make([]fr.Element, world)
			for rank := range fixture.records {
				publicInputs[rank] = evaluateLocalPolynomial(fixture.records[rank].Relation.PublicInput, fixture.context.Alpha)
			}
			wantPI := foldOuterPIOPTestTable(t, publicInputs, fixture.context.Point)
			assertOuterPIOPReductionField(t, fixture.piAtR, wantPI, "public-input fold")

			gotEndpoint, err := ReconstructOuterPIOPEndpoint(
				fixture.proof.FoldedCircuit,
				fixture.proof.TreeEvaluations,
				fixture.piAtR,
				fixture.context,
			)
			if err != nil {
				t.Fatalf("reconstruct endpoint: %v", err)
			}
			wantEndpoint := independentlyEvaluateOuterPIOPEndpoint(
				t,
				fixture.proof.FoldedCircuit,
				fixture.proof.TreeEvaluations,
				fixture.piAtR,
				fixture.context,
			)
			assertOuterPIOPReductionField(t, gotEndpoint.R0, wantEndpoint.R0, "endpoint R0")
			assertOuterPIOPReductionField(t, gotEndpoint.R1, wantEndpoint.R1, "endpoint R1")
			assertOuterPIOPReductionField(t, gotEndpoint.R2, wantEndpoint.R2, "endpoint R2")
			assertOuterPIOPReductionField(t, gotEndpoint.Equality, wantEndpoint.Equality, "endpoint equality")
			assertOuterPIOPReductionField(t, gotEndpoint.Value, wantEndpoint.Value, "endpoint value")
			if err := VerifyOuterPIOPSumCheck(fixture.proof, fixture.piAtR, fixture.context); err != nil {
				t.Fatalf("verify outer PIOP SumCheck: %v", err)
			}
		})
	}
}

func TestOuterPIOPReductionRejectsTampering(t *testing.T) {
	fixture := buildOuterPIOPReductionFixture(t, 4)
	one := fr.One()

	t.Run("folded circuit terminal", func(t *testing.T) {
		proof := cloneOuterPIOPSumCheckProof(fixture.proof)
		proof.FoldedCircuit.Alpha[AlphaQC].Add(&proof.FoldedCircuit.Alpha[AlphaQC], &one)
		if err := VerifyOuterPIOPSumCheck(proof, fixture.piAtR, fixture.context); !errors.Is(err, ErrOuterPIOPEndpoint) {
			t.Fatalf("error = %v, want ErrOuterPIOPEndpoint", err)
		}
	})

	t.Run("tree evaluation", func(t *testing.T) {
		proof := cloneOuterPIOPSumCheckProof(fixture.proof)
		proof.TreeEvaluations[0].Add(&proof.TreeEvaluations[0], &one)
		if err := VerifyOuterPIOPSumCheck(proof, fixture.piAtR, fixture.context); !errors.Is(err, ErrOuterPIOPEndpoint) {
			t.Fatalf("error = %v, want ErrOuterPIOPEndpoint", err)
		}
	})

	t.Run("ProductCheck root", func(t *testing.T) {
		proof := cloneOuterPIOPSumCheckProof(fixture.proof)
		proof.TreeEvaluations[4].SetUint64(2)
		if err := VerifyOuterPIOPSumCheck(proof, fixture.piAtR, fixture.context); !errors.Is(err, ErrOuterPIOPProductRoot) {
			t.Fatalf("error = %v, want ErrOuterPIOPProductRoot", err)
		}
	})

	t.Run("SumCheck round", func(t *testing.T) {
		proof := cloneOuterPIOPSumCheckProof(fixture.proof)
		proof.Transcript.Rounds[0][0].Add(&proof.Transcript.Rounds[0][0], &one)
		if err := VerifyOuterPIOPSumCheck(proof, fixture.piAtR, fixture.context); !errors.Is(err, ErrOuterPIOPEndpoint) {
			t.Fatalf("error = %v, want ErrOuterPIOPEndpoint", err)
		}
	})

	t.Run("public input fold", func(t *testing.T) {
		badPI := fixture.piAtR
		badPI.Add(&badPI, &one)
		if err := VerifyOuterPIOPSumCheck(fixture.proof, badPI, fixture.context); !errors.Is(err, ErrOuterPIOPEndpoint) {
			t.Fatalf("error = %v, want ErrOuterPIOPEndpoint", err)
		}
	})
}

func TestOuterPIOPReductionRejectsMalformedRecordsAndShapes(t *testing.T) {
	fixture := buildOuterPIOPReductionFixture(t, 4)

	t.Run("non-power-of-two record count", func(t *testing.T) {
		_, err := BuildOuterPIOPCoordinatorStateFromTerminals(
			fixture.terminals[:3], fixture.publicInputs[:3], fixture.reductionContext,
		)
		if !errors.Is(err, ErrInvalidOuterPIOPShape) {
			t.Fatalf("error = %v, want ErrInvalidOuterPIOPShape", err)
		}
	})

	t.Run("W3 local residual", func(t *testing.T) {
		terminals := append([]LocalTerminalEvaluations(nil), fixture.terminals...)
		terminals[0].Alpha[AlphaH].Add(&terminals[0].Alpha[AlphaH], valuePointer(fr.One()))
		_, err := BuildOuterPIOPCoordinatorStateFromTerminals(terminals, fixture.publicInputs, fixture.reductionContext)
		if !errors.Is(err, ErrOuterPIOPLocalResidual) {
			t.Fatalf("error = %v, want ErrOuterPIOPLocalResidual", err)
		}
	})

	t.Run("W3 zero boundary denominator", func(t *testing.T) {
		terminals := append([]LocalTerminalEvaluations(nil), fixture.terminals...)
		terminals[0].XStar[XStarPhiPrimeA].SetZero()
		_, err := BuildOuterPIOPCoordinatorStateFromTerminals(terminals, fixture.publicInputs, fixture.reductionContext)
		if !errors.Is(err, ErrInvalidOuterPIOPRecord) {
			t.Fatalf("error = %v, want ErrInvalidOuterPIOPRecord", err)
		}
	})

	t.Run("statement PI shape", func(t *testing.T) {
		_, err := BuildOuterPIOPCoordinatorStateFromTerminals(
			fixture.terminals, fixture.publicInputs[:3], fixture.reductionContext,
		)
		if !errors.Is(err, ErrInvalidOuterPIOPShape) {
			t.Fatalf("error = %v, want ErrInvalidOuterPIOPShape", err)
		}
	})

	t.Run("statement PI value", func(t *testing.T) {
		publicInputs := append([]fr.Element(nil), fixture.publicInputs...)
		publicInputs[0].Add(&publicInputs[0], valuePointer(fr.One()))
		_, err := BuildOuterPIOPCoordinatorStateFromTerminals(
			fixture.terminals, publicInputs, fixture.reductionContext,
		)
		if !errors.Is(err, ErrOuterPIOPLocalResidual) {
			t.Fatalf("error = %v, want ErrOuterPIOPLocalResidual", err)
		}
	})

	t.Run("terminal does not match relation", func(t *testing.T) {
		records := append([]OuterPIOPLocalRecord(nil), fixture.records...)
		records[0].Terminals.Alpha[AlphaH].Add(&records[0].Terminals.Alpha[AlphaH], valuePointer(fr.One()))
		_, err := BuildOuterPIOPCoordinatorState(records, fixture.context.Alpha)
		if !errors.Is(err, ErrInvalidOuterPIOPRecord) {
			t.Fatalf("error = %v, want ErrInvalidOuterPIOPRecord", err)
		}
	})

	t.Run("wrong SumCheck arity", func(t *testing.T) {
		_, err := fixture.state.BuildSumCheckInstance(fixture.context.Zeta, fixture.context.Theta[:1])
		if !errors.Is(err, ErrOuterPIOPSumCheckInstance) {
			t.Fatalf("error = %v, want ErrOuterPIOPSumCheckInstance", err)
		}
		_, err = fixture.instance.Prove(fixture.context.Point[:1])
		if !errors.Is(err, ErrOuterPIOPSumCheckInstance) {
			t.Fatalf("error = %v, want ErrOuterPIOPSumCheckInstance", err)
		}
	})

	t.Run("invalid endpoint shape", func(t *testing.T) {
		context := fixture.context
		context.M = 3
		_, err := ReconstructOuterPIOPEndpoint(fixture.proof.FoldedCircuit, fixture.proof.TreeEvaluations, fixture.piAtR, context)
		if !errors.Is(err, ErrOuterPIOPEndpoint) {
			t.Fatalf("error = %v, want ErrOuterPIOPEndpoint", err)
		}
	})
}

func TestOuterPIOPReductionRejectsBrokenGlobalProduct(t *testing.T) {
	const world = 4
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	extracted := extractAllLocalPIOPRanks(t, spr, witness, world)
	challenges, relations := buildAllExtractedLocalRelations(t, extracted)
	alpha := fr.NewElement(5)
	records := make([]OuterPIOPLocalRecord, world)
	terminals := make([]LocalTerminalEvaluations, world)
	publicInputs := make([]fr.Element, world)
	for rank := range relations {
		var err error
		records[rank], err = NewOuterPIOPLocalRecord(relations[rank], alpha)
		if err != nil {
			t.Fatalf("record rank %d: %v", rank, err)
		}
		terminals[rank] = records[rank].Terminals
		publicInputs[rank] = evaluateLocalPolynomial(relations[rank].PublicInput, alpha)
	}
	reductionContext := outerPIOPReductionContextFromRelation(relations[0], alpha)

	for delta := uint64(1); delta <= 32; delta++ {
		badTable := cloneOuterPIOPTestTable(extracted[0].table)
		increment := fr.NewElement(delta)
		badTable.SigmaPart[LocalWireA][0].Add(&badTable.SigmaPart[LocalWireA][0], &increment)
		badRelation, err := BuildLocalPIOPRelation(badTable, extracted[0].index, challenges)
		if errors.Is(err, ErrZeroLocalTagDenominator) {
			continue
		}
		if err != nil {
			t.Fatalf("build locally valid tampered relation: %v", err)
		}
		badRecord, err := NewOuterPIOPLocalRecord(badRelation, alpha)
		if err != nil {
			t.Fatalf("build tampered record: %v", err)
		}
		candidate := append([]LocalTerminalEvaluations(nil), terminals...)
		candidatePI := append([]fr.Element(nil), publicInputs...)
		candidate[0] = badRecord.Terminals
		candidatePI[0] = evaluateLocalPolynomial(badRelation.PublicInput, alpha)
		_, err = BuildOuterPIOPCoordinatorStateFromTerminals(candidate, candidatePI, reductionContext)
		if errors.Is(err, ErrOuterPIOPProductRoot) {
			return
		}
		if err != nil {
			t.Fatalf("unexpected error for locally valid product tamper: %v", err)
		}
	}
	t.Fatal("could not construct a locally valid record with a nonunit global product")
}

func buildOuterPIOPReductionFixture(t *testing.T, world int) outerPIOPReductionFixture {
	t.Helper()
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	extracted := extractAllLocalPIOPRanks(t, spr, witness, world)
	_, relations := buildAllExtractedLocalRelations(t, extracted)
	alpha := fr.NewElement(5)
	records := make([]OuterPIOPLocalRecord, world)
	terminals := make([]LocalTerminalEvaluations, world)
	publicInputs := make([]fr.Element, world)
	for rank := range relations {
		var err error
		records[rank], err = NewOuterPIOPLocalRecord(relations[rank], alpha)
		if err != nil {
			t.Fatalf("record rank %d: %v", rank, err)
		}
		terminals[rank] = records[rank].Terminals
		publicInputs[rank] = evaluateLocalPolynomial(relations[rank].PublicInput, alpha)
	}
	reductionContext := outerPIOPReductionContextFromRelation(relations[0], alpha)
	state, err := BuildOuterPIOPCoordinatorStateFromTerminals(terminals, publicInputs, reductionContext)
	if err != nil {
		t.Fatalf("build coordinator state: %v", err)
	}
	wrapperState, err := BuildOuterPIOPCoordinatorState(records, alpha)
	if err != nil {
		t.Fatalf("build full-relation convenience state: %v", err)
	}
	coreProduct, _, _ := state.CoordinatorProductCheckWitness()
	wrapperProduct, _, _ := wrapperState.CoordinatorProductCheckWitness()
	for i := range coreProduct {
		assertOuterPIOPReductionField(t, coreProduct[i], wrapperProduct[i], "terminal/full-relation ProductCheck")
	}
	logM := 0
	for size := world; size > 1; size >>= 1 {
		logM++
	}
	theta := make([]fr.Element, logM)
	point := make([]fr.Element, logM)
	for coordinate := 0; coordinate < logM; coordinate++ {
		theta[coordinate].SetUint64(uint64(11 + coordinate))
		point[coordinate].SetUint64(uint64(17 + coordinate))
	}
	zeta := fr.NewElement(13)
	instance, err := state.BuildSumCheckInstance(zeta, theta)
	if err != nil {
		t.Fatalf("build SumCheck instance: %v", err)
	}
	proof, err := instance.Prove(point)
	if err != nil {
		t.Fatalf("prove outer PIOP: %v", err)
	}
	piAtR, err := state.CoordinatorPublicInputAt(point)
	if err != nil {
		t.Fatalf("fold public input: %v", err)
	}
	return outerPIOPReductionFixture{
		records:          records,
		terminals:        terminals,
		publicInputs:     publicInputs,
		reductionContext: reductionContext,
		state:            state,
		instance:         instance,
		proof:            proof,
		context:          instance.EndpointContext(point),
		piAtR:            piAtR,
	}
}

func outerPIOPReductionContextFromRelation(relation *LocalPIOPRelation, alpha fr.Element) OuterPIOPReductionContext {
	return OuterPIOPReductionContext{
		DomainSize: relation.DomainSize,
		Omega:      relation.Omega,
		XStar:      relation.XStar,
		Alpha:      alpha,
		WireCosets: relation.Index.WireCosets,
		Challenges: relation.Challenges,
	}
}

func assertOuterPIOPCircuitFolds(t *testing.T, records []OuterPIOPLocalRecord, point []fr.Element, got LocalTerminalEvaluations) {
	t.Helper()
	gotFlat := got.Flatten()
	for terminal := 0; terminal < LocalTerminalTotalCount; terminal++ {
		table := make([]fr.Element, len(records))
		for rank := range records {
			table[rank] = records[rank].Terminals.Flatten()[terminal]
		}
		want := foldOuterPIOPTestTable(t, table, point)
		assertOuterPIOPReductionField(t, gotFlat[terminal], want, "folded circuit terminal")
	}
}

func assertOuterPIOPTreeClaims(t *testing.T, table []fr.Element, point []fr.Element, got [terminalTreeClaims]fr.Element) {
	t.Helper()
	m := len(table) / 2
	even := make([]fr.Element, m)
	odd := make([]fr.Element, m)
	for row := 0; row < m; row++ {
		even[row] = table[2*row]
		odd[row] = table[2*row+1]
	}
	want := [terminalTreeClaims]fr.Element{
		foldOuterPIOPTestTable(t, table[:m], point),
		foldOuterPIOPTestTable(t, table[m:], point),
		foldOuterPIOPTestTable(t, even, point),
		foldOuterPIOPTestTable(t, odd, point),
		table[2*m-2],
	}
	for claim := range want {
		assertOuterPIOPReductionField(t, got[claim], want[claim], "tree functional")
	}
}

func independentlyEvaluateOuterPIOPEndpoint(
	t *testing.T,
	folded LocalTerminalEvaluations,
	tree [terminalTreeClaims]fr.Element,
	publicInput fr.Element,
	context OuterPIOPEndpointContext,
) OuterPIOPEndpointEvaluation {
	t.Helper()
	var result OuterPIOPEndpointEvaluation
	var wires [LocalWireCount]fr.Element
	partitionLabel := independentlyEvaluatePartitionLabel(context.Point)
	for wire := 0; wire < LocalWireCount; wire++ {
		wires[wire] = folded.Alpha[int(AlphaPhiPrimeA)+wire]
		var term fr.Element
		term.Mul(&context.Challenges.EtaPart, &partitionLabel)
		wires[wire].Sub(&wires[wire], &term)
		term.Mul(&context.WireCosets[wire], &context.Alpha).Mul(&term, &context.Challenges.EtaX)
		wires[wire].Sub(&wires[wire], &term)
		wires[wire].Sub(&wires[wire], &context.Challenges.Gamma)
	}

	var gate, term fr.Element
	term.Mul(&wires[LocalWireA], &wires[LocalWireB]).Mul(&term, &folded.Alpha[AlphaQM])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireA], &folded.Alpha[AlphaQL])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireB], &folded.Alpha[AlphaQR])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireC], &folded.Alpha[AlphaQO])
	gate.Add(&gate, &term)
	gate.Add(&gate, &folded.Alpha[AlphaQC])
	gate.Add(&gate, &publicInput)

	fAlpha := independentlyMultiplyThree(folded.Alpha[AlphaPhiA], folded.Alpha[AlphaPhiB], folded.Alpha[AlphaPhiC])
	fPrimeAlpha := independentlyMultiplyThree(folded.Alpha[AlphaPhiPrimeA], folded.Alpha[AlphaPhiPrimeB], folded.Alpha[AlphaPhiPrimeC])
	fXStar := independentlyMultiplyThree(folded.XStar[XStarPhiA], folded.XStar[XStarPhiB], folded.XStar[XStarPhiC])
	fPrimeXStar := independentlyMultiplyThree(folded.XStar[XStarPhiPrimeA], folded.XStar[XStarPhiPrimeB], folded.XStar[XStarPhiPrimeC])

	alphaT := fr.One()
	for i := 0; i < context.T; i++ {
		alphaT.Mul(&alphaT, &context.Alpha)
	}
	alphaTMinusOne := alphaT
	one := fr.One()
	alphaTMinusOne.Sub(&alphaTMinusOne, &one)
	tAsField := fr.NewElement(uint64(context.T))
	var denominator, lZero, lStar fr.Element
	denominator.Sub(&context.Alpha, &one).Mul(&denominator, &tAsField)
	lZero.Div(&alphaTMinusOne, &denominator)
	denominator.Sub(&context.Alpha, &context.XStar).Mul(&denominator, &tAsField)
	lStar.Mul(&context.XStar, &alphaTMinusOne).Div(&lStar, &denominator)

	var boundary, transition, endpoint fr.Element
	boundary.Sub(&folded.Alpha[AlphaZ], &one).Mul(&boundary, &lZero)
	transition.Mul(&folded.Alpha[AlphaZ], &fAlpha)
	term.Mul(&folded.OmegaAlpha[0], &fPrimeAlpha)
	transition.Sub(&transition, &term)
	endpoint.Mul(&folded.XStar[XStarZ], &fXStar).Sub(&endpoint, &fPrimeXStar)
	term.Mul(&lStar, &endpoint)
	transition.Sub(&transition, &term)

	result.R0 = gate
	term.Mul(&context.Challenges.Lambda, &boundary)
	result.R0.Add(&result.R0, &term)
	var lambdaSquared fr.Element
	lambdaSquared.Square(&context.Challenges.Lambda)
	term.Mul(&lambdaSquared, &transition)
	result.R0.Add(&result.R0, &term)
	term.Mul(&alphaTMinusOne, &folded.Alpha[AlphaH])
	result.R0.Sub(&result.R0, &term)

	result.R1.Mul(&tree[0], &fPrimeXStar)
	term.Mul(&folded.XStar[XStarZ], &fXStar)
	result.R1.Sub(&result.R1, &term)
	term.Mul(&tree[2], &tree[3])
	result.R2.Sub(&tree[1], &term)

	result.Equality.SetOne()
	for coordinate := range context.Theta {
		var left, right, factor fr.Element
		left.Sub(&one, &context.Theta[coordinate])
		right.Sub(&one, &context.Point[coordinate])
		factor.Mul(&left, &right)
		left.Mul(&context.Theta[coordinate], &context.Point[coordinate])
		factor.Add(&factor, &left)
		result.Equality.Mul(&result.Equality, &factor)
	}
	result.Value = result.R0
	term.Mul(&context.Zeta, &result.R1)
	result.Value.Add(&result.Value, &term)
	var zetaSquared fr.Element
	zetaSquared.Square(&context.Zeta)
	term.Mul(&zetaSquared, &result.R2)
	result.Value.Add(&result.Value, &term)
	result.Value.Mul(&result.Value, &result.Equality)
	return result
}

func independentlyEvaluatePartitionLabel(point []fr.Element) fr.Element {
	var result fr.Element
	weight := fr.One()
	for coordinate := range point {
		var term fr.Element
		term.Mul(&weight, &point[coordinate])
		result.Add(&result, &term)
		weight.Double(&weight)
	}
	return result
}

func independentlyMultiplyThree(a, b, c fr.Element) fr.Element {
	var result fr.Element
	result.Mul(&a, &b).Mul(&result, &c)
	return result
}

func foldOuterPIOPTestTable(t *testing.T, table, point []fr.Element) fr.Element {
	t.Helper()
	working := append([]fr.Element(nil), table...)
	for coordinate := range point {
		if len(working)%2 != 0 {
			t.Fatalf("cannot fold odd table length %d", len(working))
		}
		oneMinus := fr.One()
		oneMinus.Sub(&oneMinus, &point[coordinate])
		next := make([]fr.Element, len(working)/2)
		for row := range next {
			var even, odd fr.Element
			even.Mul(&working[2*row], &oneMinus)
			odd.Mul(&working[2*row+1], &point[coordinate])
			next[row].Add(&even, &odd)
		}
		working = next
	}
	if len(working) != 1 {
		t.Fatalf("fold left %d values", len(working))
	}
	return working[0]
}

func cloneOuterPIOPTestTable(table LocalPIOPTable) LocalPIOPTable {
	var result LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		result.Wires[wire] = cloneElements(table.Wires[wire])
		result.SigmaX[wire] = cloneElements(table.SigmaX[wire])
		result.SigmaPart[wire] = cloneElements(table.SigmaPart[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		result.Selectors[selector] = cloneElements(table.Selectors[selector])
	}
	result.PublicInput = cloneElements(table.PublicInput)
	return result
}

func cloneOuterPIOPSumCheckProof(proof OuterPIOPSumCheckProof) OuterPIOPSumCheckProof {
	result := proof
	result.Transcript.Rounds = make([][]fr.Element, len(proof.Transcript.Rounds))
	for round := range proof.Transcript.Rounds {
		result.Transcript.Rounds[round] = cloneElements(proof.Transcript.Rounds[round])
	}
	return result
}

func assertOuterPIOPReductionField(t *testing.T, got, want fr.Element, label string) {
	t.Helper()
	if !got.Equal(&want) {
		t.Fatalf("%s = %s, want %s", label, got.String(), want.String())
	}
}
