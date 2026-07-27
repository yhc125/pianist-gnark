package dlinkzg

import (
	"errors"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

func TestDLinkZGOpeningHonestRandomizedM2M4(t *testing.T) {
	for _, parties := range []int{2, 4} {
		for seed := int64(1); seed <= 3; seed++ {
			t.Run(fmt.Sprintf("M=%d/seed=%d", parties, seed), func(t *testing.T) {
				fixture := newDLinkZGOpeningTestFixture(t, parties, 8, seed)
				proof, err := ProveDLinkZGOpening(fixture.instance, fixture.input)
				if err != nil {
					t.Fatalf("prove: %v", err)
				}
				if err := VerifyDLinkZGOpening(fixture.instance, proof, fixture.input.VerifierSRS); err != nil {
					t.Fatalf("verify: %v", err)
				}
				assertDLinkZGOpeningMatchesMonolithic(t, fixture, proof)
			})
		}
	}
}

func TestDLinkZGOpeningAllowsZeroSourceLinkPoint(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 101)
	factory := func(context TranscriptContext) (dlinkzgOpeningTranscript, error) {
		if err := validateTranscriptContext(context); err != nil {
			return nil, err
		}
		return &dlinkzgOpeningFixedTranscript{
			xi:    fr.NewElement(7),
			nu:    fr.NewElement(11),
			z:     fr.Element{},
			beta:  fr.NewElement(13),
			kappa: fr.NewElement(17),
			delta: fr.NewElement(19),
		}, nil
	}
	proof, err := proveDLinkZGOpeningWithTranscript(fixture.instance, fixture.input, factory)
	if err != nil {
		t.Fatalf("prove with z_ch=0: %v", err)
	}
	if err := verifyDLinkZGOpeningWithTranscript(fixture.instance, proof, fixture.input.VerifierSRS, factory); err != nil {
		t.Fatalf("verify with z_ch=0: %v", err)
	}
}

func TestDLinkZGOpeningRejectsEveryPhaseTamper(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 211)
	proof, err := ProveDLinkZGOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	one := fr.One()
	point := dlinkzgOpeningTestG1(991)
	tests := []struct {
		name   string
		mutate func(*DLinkZGOpeningProof)
	}{
		{"U0 commitment", func(p *DLinkZGOpeningProof) { p.U0.PartialCommitments[0] = point }},
		{"U1 link value", func(p *DLinkZGOpeningProof) { p.U1.LinkEvaluations[1].Add(&p.U1.LinkEvaluations[1], &one) }},
		{"U1 link commitment", func(p *DLinkZGOpeningProof) { p.U1.LinkCommitment = point }},
		{"U1 Laurent commitment", func(p *DLinkZGOpeningProof) { p.U1.LaurentCommitment = point }},
		{"U2 partial beta", func(p *DLinkZGOpeningProof) { p.U2.PartialAtBeta[2].Add(&p.U2.PartialAtBeta[2], &one) }},
		{"U2 partial beta inverse", func(p *DLinkZGOpeningProof) { p.U2.PartialAtBetaInverse[0].Add(&p.U2.PartialAtBetaInverse[0], &one) }},
		{"U2 batch beta", func(p *DLinkZGOpeningProof) { p.U2.BatchAtBeta[1].Add(&p.U2.BatchAtBeta[1], &one) }},
		{"U2 derived inverse", func(p *DLinkZGOpeningProof) { p.U2.BatchAtBetaInverse[3].Add(&p.U2.BatchAtBetaInverse[3], &one) }},
		{"U3 WN", func(p *DLinkZGOpeningProof) { p.U3.WN = point }},
		{"U3 piZ", func(p *DLinkZGOpeningProof) { p.U3.PiZ = point }},
		{"U3 piY", func(p *DLinkZGOpeningProof) { p.U3.PiY = point }},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			tampered := proof
			testCase.mutate(&tampered)
			if err := VerifyDLinkZGOpening(fixture.instance, tampered, fixture.input.VerifierSRS); err == nil {
				t.Fatal("tampered proof accepted")
			}
		})
	}
}

func TestDLinkZGOpeningRejectsInstanceAndSemanticOrderTamper(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 307)
	proof, err := ProveDLinkZGOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	one := fr.One()
	point := dlinkzgOpeningTestG1(1231)
	tests := []struct {
		name   string
		mutate func(*DLinkZGOpeningInstance)
	}{
		{"source commitment", func(i *DLinkZGOpeningInstance) { i.SourceCommitments[0] = point }},
		{"circuit claim", func(i *DLinkZGOpeningInstance) { i.CircuitClaims[1].Add(&i.CircuitClaims[1], &one) }},
		{"tree commitment", func(i *DLinkZGOpeningInstance) { i.TreeCommitments[0] = point }},
		{"tree claim", func(i *DLinkZGOpeningInstance) { i.TreeClaims[3].Add(&i.TreeClaims[3], &one) }},
		{"semantic query order", func(i *DLinkZGOpeningInstance) {
			i.SemanticQueryPoints[0], i.SemanticQueryPoints[1] = i.SemanticQueryPoints[1], i.SemanticQueryPoints[0]
		}},
		{"source order", func(i *DLinkZGOpeningInstance) {
			i.SourceCommitments[0], i.SourceCommitments[2] = i.SourceCommitments[2], i.SourceCommitments[0]
		}},
		{"shift", func(i *DLinkZGOpeningInstance) { i.Shift.Add(&i.Shift, &one) }},
		{"partition point", func(i *DLinkZGOpeningInstance) { i.PartitionPoint[0].Add(&i.PartitionPoint[0], &one) }},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			tampered := cloneDLinkZGOpeningInstance(fixture.instance)
			testCase.mutate(&tampered)
			if err := VerifyDLinkZGOpening(tampered, proof, fixture.input.VerifierSRS); err == nil {
				t.Fatal("tampered public instance accepted")
			}
		})
	}
}

func TestDLinkZGOpeningDigestRejectsCancellingPairTamper(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 353)
	proof, err := ProveDLinkZGOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	challenges := replayDLinkZGOpeningChallenges(t, fixture.instance, proof)
	adjustment := fr.NewElement(1234567)

	t.Run("two circuit claims cancel under xi", func(t *testing.T) {
		tampered := cloneDLinkZGOpeningInstance(fixture.instance)
		xiInverse := dlinkzgOpeningInverse(challenges.xi)
		var cancelling fr.Element
		cancelling.Mul(&adjustment, &xiInverse)
		tampered.CircuitClaims[0].Add(&tampered.CircuitClaims[0], &adjustment)
		tampered.CircuitClaims[1].Sub(&tampered.CircuitClaims[1], &cancelling)
		before := dlinkzgOpeningLinearTarget(fixture.instance, proof.U1.LinkEvaluations, challenges.xi, challenges.nu)
		after := dlinkzgOpeningLinearTarget(tampered, proof.U1.LinkEvaluations, challenges.xi, challenges.nu)
		if !before.Equal(&after) {
			t.Fatal("test mutation does not cancel under xi")
		}
		assertDLinkZGOpeningDigestRejects(t, tampered, proof, fixture.input.VerifierSRS)
	})

	t.Run("two source commitments cancel under xi", func(t *testing.T) {
		tampered := cloneDLinkZGOpeningInstance(fixture.instance)
		point := dlinkzgOpeningTestG1(1777)
		xiInverse := dlinkzgOpeningInverse(challenges.xi)
		var negativeXiInverse fr.Element
		negativeXiInverse.Neg(&xiInverse)
		tampered.SourceCommitments[0] = dlinkzgOpeningTestAddScaledG1(tampered.SourceCommitments[0], point, fr.One())
		tampered.SourceCommitments[1] = dlinkzgOpeningTestAddScaledG1(tampered.SourceCommitments[1], point, negativeXiInverse)
		before := foldG1(fixture.instance.SourceCommitments[:], challenges.xi)
		after := foldG1(tampered.SourceCommitments[:], challenges.xi)
		if !before.Equal(&after) {
			t.Fatal("test source mutation does not cancel under xi")
		}
		assertDLinkZGOpeningDigestRejects(t, tampered, proof, fixture.input.VerifierSRS)
	})

	t.Run("tree commitments cancel under kappa", func(t *testing.T) {
		tampered := cloneDLinkZGOpeningInstance(fixture.instance)
		point := dlinkzgOpeningTestG1(1879)
		kappaInverse := dlinkzgOpeningInverse(challenges.kappa)
		var negativeKappaInverse fr.Element
		negativeKappaInverse.Neg(&kappaInverse)
		tampered.TreeCommitments[0] = dlinkzgOpeningTestAddScaledG1(tampered.TreeCommitments[0], point, fr.One())
		tampered.TreeCommitments[1] = dlinkzgOpeningTestAddScaledG1(tampered.TreeCommitments[1], point, negativeKappaInverse)
		beforeCommitments := []bn254.G1Affine{proof.U1.LinkCommitment, fixture.instance.TreeCommitments[0], fixture.instance.TreeCommitments[1], proof.U1.LaurentCommitment}
		afterCommitments := []bn254.G1Affine{proof.U1.LinkCommitment, tampered.TreeCommitments[0], tampered.TreeCommitments[1], proof.U1.LaurentCommitment}
		before := foldG1(beforeCommitments, challenges.kappa)
		after := foldG1(afterCommitments, challenges.kappa)
		if !before.Equal(&after) {
			t.Fatal("test tree mutation does not cancel under kappa")
		}
		assertDLinkZGOpeningDigestRejects(t, tampered, proof, fixture.input.VerifierSRS)
	})
}

func TestDLinkZGOpeningContextBinderPreservesOuterPrefix(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 2, 8, 367)
	base := fixture.instance.TranscriptContext
	base.PublicStatementDigest = NewTranscriptDigest([]byte("unbound immediate statement"))
	bound := BindDLinkZGOpeningTranscriptContext(fixture.instance, base)
	if bound.PublicStatementDigest != DLinkZGOpeningInstanceDigest(fixture.instance) {
		t.Fatal("binder did not install canonical opening-instance digest")
	}
	if bound.PriorTranscriptDigest != base.PriorTranscriptDigest {
		t.Fatal("binder changed the outer prior transcript digest")
	}
	if bound.SRSDigest != base.SRSDigest || bound.IndexDigest != base.IndexDigest || bound.PartyManifestDigest != base.PartyManifestDigest || bound.SessionNonce != base.SessionNonce {
		t.Fatal("binder changed an unrelated transcript context field")
	}
}

func TestDLinkZGOpeningRejectsRankAndWitnessOrder(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 401)
	rankSwapped := fixture.input
	rankSwapped.PartySRS = append([]*cryptodlinkzg.PartyRowSRS(nil), fixture.input.PartySRS...)
	rankSwapped.PartySRS[0], rankSwapped.PartySRS[1] = rankSwapped.PartySRS[1], rankSwapped.PartySRS[0]
	if _, err := ProveDLinkZGOpening(fixture.instance, rankSwapped); !errors.Is(err, ErrInvalidDLinkZGOpeningInput) {
		t.Fatalf("rank-swapped SRS: got %v, want ErrInvalidDLinkZGOpeningInput", err)
	}

	witnessSwapped := cloneDLinkZGOpeningProverInput(fixture.input)
	for claim := range witnessSwapped.Sources {
		witnessSwapped.Sources[claim][0], witnessSwapped.Sources[claim][1] = witnessSwapped.Sources[claim][1], witnessSwapped.Sources[claim][0]
	}
	if _, err := ProveDLinkZGOpening(fixture.instance, witnessSwapped); err == nil {
		t.Fatal("manifest-swapped source rows produced an accepting proof")
	}
}

func TestDLinkZGOpeningSemanticRowsEqualShiftedNativeRows(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 409)
	for rank := range fixture.input.PartySRS {
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			semantic, err := fixture.input.PartySRS[rank].CommitSemantic(fixture.input.Sources[claim][rank])
			if err != nil {
				t.Fatal(err)
			}
			shifted := cryptodlinkzg.FastTaylorShift(fixture.input.Sources[claim][rank], fixture.instance.Shift)
			native, err := fixture.input.PartySRS[rank].CommitRow(shifted)
			if err != nil {
				t.Fatal(err)
			}
			if !semantic.Equal(&native) {
				t.Fatalf("semantic/native commitment mismatch at rank %d claim %d", rank, claim)
			}
		}
	}
}

func TestDLinkZGOpeningDirectVerifierWeightsMatchDenseReference(t *testing.T) {
	random := rand.New(rand.NewSource(487))
	for coordinates := 1; coordinates <= 4; coordinates++ {
		point := make([]fr.Element, coordinates)
		for i := range point {
			point[i] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
		}
		for sample := 0; sample < 8; sample++ {
			evaluationPoint := fr.NewElement(uint64(1 + random.Int63n(1<<30)))
			want := cryptodlinkzg.Eval(cryptodlinkzg.EqualityWeights(point), evaluationPoint)
			got := dlinkzgOpeningPsiEval(point, evaluationPoint)
			if !got.Equal(&want) {
				t.Fatalf("psi product evaluation mismatch for %d coordinates", coordinates)
			}
		}
	}

	for _, parties := range []int{2, 4} {
		partitionPoint := make([]fr.Element, bits.Len(uint(parties))-1)
		for i := range partitionPoint {
			partitionPoint[i] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
		}
		points := productCheckFunctionalPoints(partitionPoint)
		xi := fr.NewElement(uint64(1 + random.Int63n(1<<30)))
		evaluationPoint := fr.NewElement(uint64(1 + random.Int63n(1<<30)))
		denseA, denseB := treeWeightPolynomials(points, xi, 8)
		wantA := cryptodlinkzg.Eval(denseA, evaluationPoint)
		wantB := cryptodlinkzg.Eval(denseB, evaluationPoint)
		gotA, gotB := dlinkzgOpeningTreeWeightEvals(points, xi, evaluationPoint)
		if !gotA.Equal(&wantA) || !gotB.Equal(&wantB) {
			t.Fatalf("direct tree weights differ for M=%d", parties)
		}
	}
}

func TestDLinkZGOpeningRejectsMalformedShapes(t *testing.T) {
	fixture := newDLinkZGOpeningTestFixture(t, 4, 8, 503)

	malformedInstance := cloneDLinkZGOpeningInstance(fixture.instance)
	malformedInstance.LocalDomainSize = 6
	if _, err := ProveDLinkZGOpening(malformedInstance, fixture.input); !errors.Is(err, ErrInvalidDLinkZGOpeningInstance) {
		t.Fatalf("non-power-of-two T: got %v", err)
	}
	malformedInstance.LocalDomainSize = FastLocalPIOPMaxDomainSize << 1
	if _, err := ProveDLinkZGOpening(malformedInstance, fixture.input); !errors.Is(err, ErrInvalidDLinkZGOpeningInstance) {
		t.Fatalf("unsupported T: got %v", err)
	}

	malformedInput := cloneDLinkZGOpeningProverInput(fixture.input)
	malformedInput.T0 = malformedInput.T0[:3]
	if _, err := ProveDLinkZGOpening(fixture.instance, malformedInput); !errors.Is(err, ErrInvalidDLinkZGOpeningInput) {
		t.Fatalf("short t0: got %v", err)
	}

	malformedInput = cloneDLinkZGOpeningProverInput(fixture.input)
	malformedInput.Sources[2][1] = make([]fr.Element, 9)
	if _, err := ProveDLinkZGOpening(fixture.instance, malformedInput); !errors.Is(err, ErrInvalidDLinkZGOpeningInput) {
		t.Fatalf("wide row: got %v", err)
	}

	nonCanonical := cloneDLinkZGOpeningInstance(fixture.instance)
	nonCanonical.CircuitClaims[0] = fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	if err := validateTranscriptField(&nonCanonical.CircuitClaims[0]); err == nil {
		t.Fatal("non-canonical field fixture unexpectedly canonical")
	}
	if _, err := ProveDLinkZGOpening(nonCanonical, fixture.input); !errors.Is(err, ErrInvalidDLinkZGOpeningInstance) {
		t.Fatalf("non-canonical public field: got %v", err)
	}

	if err := VerifyDLinkZGOpening(fixture.instance, DLinkZGOpeningProof{}, fixture.input.VerifierSRS); err == nil {
		t.Fatal("zero proof accepted")
	}
	if err := VerifyDLinkZGOpening(fixture.instance, DLinkZGOpeningProof{}, nil); !errors.Is(err, ErrInvalidDLinkZGOpeningInstance) {
		t.Fatalf("nil verifier SRS: got %v", err)
	}
}

func TestDLinkZGOpeningProofHasNoPartySizedData(t *testing.T) {
	typ := reflect.TypeOf(DLinkZGOpeningProof{})
	if path, found := dlinkzgOpeningDynamicFieldPath(typ, "DLinkZGOpeningProof"); found {
		t.Fatalf("public opening proof contains dynamically sized field %s", path)
	}
	if typ.NumField() != 4 || typ.Field(0).Name != "U0" || typ.Field(1).Name != "U1" || typ.Field(2).Name != "U2" || typ.Field(3).Name != "U3" {
		t.Fatalf("proof fields are not exactly ordered U0,U1,U2,U3: %v", typ)
	}
}

type dlinkzgOpeningTestFixture struct {
	instance   DLinkZGOpeningInstance
	input      DLinkZGOpeningProverInput
	monolithic *cryptodlinkzg.MonomialSRS
}

func newDLinkZGOpeningTestFixture(t testing.TB, parties, degreeBound int, seed int64) dlinkzgOpeningTestFixture {
	t.Helper()
	random := rand.New(rand.NewSource(seed))
	tauY := fr.NewElement(uint64(1009 + seed))
	tauZ := fr.NewElement(uint64(2003 + seed))
	sigma := fr.NewElement(uint64(3011 + seed))

	partySRS := make([]*cryptodlinkzg.PartyRowSRS, parties)
	for rank := range partySRS {
		row, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(parties, degreeBound, rank, tauY, tauZ, sigma)
		if err != nil {
			t.Fatalf("party SRS: %v", err)
		}
		partySRS[rank] = row
	}
	coordinatorSRS, err := cryptodlinkzg.NewDeterministicCoordinatorSRS(parties, tauY, tauZ)
	if err != nil {
		t.Fatalf("coordinator SRS: %v", err)
	}
	verifierSRS := cryptodlinkzg.NewDeterministicVerifierSRS(tauY, tauZ)
	monolithic, err := cryptodlinkzg.NewMonomialSRS(parties, degreeBound, tauY, tauZ)
	if err != nil {
		t.Fatalf("monolithic reference SRS: %v", err)
	}

	var sources [dlinkzgOpeningCircuitClaims][][]fr.Element
	for claim := range sources {
		sources[claim] = make([][]fr.Element, parties)
		for rank := range sources[claim] {
			sources[claim][rank] = make([]fr.Element, degreeBound)
			for coefficient := range sources[claim][rank] {
				sources[claim][rank][coefficient] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
			}
		}
	}
	t0 := make([]fr.Element, parties)
	t1 := make([]fr.Element, parties)
	for rank := 0; rank < parties; rank++ {
		t0[rank] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
		t1[rank] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
	}

	partitionPoint := make([]fr.Element, bits.Len(uint(parties))-1)
	for i := range partitionPoint {
		partitionPoint[i] = fr.NewElement(uint64(2 + random.Int63n(1<<20)))
	}
	weights := cryptodlinkzg.EqualityWeights(partitionPoint)

	semanticQueries := [dlinkzgOpeningCircuitClaims]fr.Element{}
	semanticQueries[0].Add(&sigma, valuePointer(fr.NewElement(2)))
	semanticQueries[1].Mul(&semanticQueries[0], valuePointer(fr.NewElement(7)))
	semanticQueries[2].Add(&sigma, valuePointer(fr.NewElement(5)))
	shifted := semanticQueries
	for i := range shifted {
		shifted[i].Sub(&shifted[i], &sigma)
	}
	if _, _, err := translatedQueries(shifted, bits.Len(uint(degreeBound))-1); err != nil {
		t.Fatalf("test semantic query falls in B_T: %v", err)
	}

	instance := DLinkZGOpeningInstance{
		LocalDomainSize:     degreeBound,
		SemanticQueryPoints: semanticQueries,
		Shift:               sigma,
		PartitionPoint:      append([]fr.Element(nil), partitionPoint...),
		TranscriptContext:   dlinkzgOpeningTestContext(seed),
	}
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		var commitment bn254.G1Jac
		for rank := 0; rank < parties; rank++ {
			rowCommitment, commitErr := partySRS[rank].CommitSemantic(sources[claim][rank])
			if commitErr != nil {
				t.Fatalf("source commitment: %v", commitErr)
			}
			dlinkzgOpeningAddG1(&commitment, rowCommitment)
			value := cryptodlinkzg.Eval(sources[claim][rank], semanticQueries[claim])
			var weighted fr.Element
			weighted.Mul(&weights[rank], &value)
			instance.CircuitClaims[claim].Add(&instance.CircuitClaims[claim], &weighted)
		}
		instance.SourceCommitments[claim].FromJacobian(&commitment)
	}
	instance.TreeCommitments[0], err = coordinatorSRS.CommitZ(t0)
	if err != nil {
		t.Fatal(err)
	}
	instance.TreeCommitments[1], err = coordinatorSRS.CommitZ(t1)
	if err != nil {
		t.Fatal(err)
	}
	treePoints := productCheckFunctionalPoints(partitionPoint)
	for claim := 0; claim < dlinkzgOpeningTreeClaims; claim++ {
		instance.TreeClaims[claim], err = evaluateTreeFunctional(t0, t1, treePoints[claim])
		if err != nil {
			t.Fatalf("tree claim: %v", err)
		}
	}
	instance.TranscriptContext = BindDLinkZGOpeningTranscriptContext(instance, instance.TranscriptContext)

	return dlinkzgOpeningTestFixture{
		instance: instance,
		input: DLinkZGOpeningProverInput{
			Sources:        sources,
			T0:             append([]fr.Element(nil), t0...),
			T1:             append([]fr.Element(nil), t1...),
			PartySRS:       partySRS,
			CoordinatorSRS: coordinatorSRS,
			VerifierSRS:    verifierSRS,
		},
		monolithic: monolithic,
	}
}

func assertDLinkZGOpeningMatchesMonolithic(t testing.TB, fixture dlinkzgOpeningTestFixture, proof DLinkZGOpeningProof) {
	t.Helper()
	challenges := replayDLinkZGOpeningChallenges(t, fixture.instance, proof)
	prepared, err := prepareDLinkZGOpeningInstance(fixture.instance)
	if err != nil {
		t.Fatal(err)
	}

	translatedRectangle := make([][]fr.Element, prepared.m)
	for rank := 0; rank < prepared.m; rank++ {
		translatedRectangle[rank] = make([]fr.Element, prepared.t)
		xiPower := fr.One()
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			shifted := cryptodlinkzg.FastTaylorShift(fixture.input.Sources[claim][rank], fixture.instance.Shift)
			dlinkzgOpeningAddScaled(translatedRectangle[rank], shifted, xiPower)
			xiPower.Mul(&xiPower, &challenges.xi)
		}
	}
	sourceCommitment, err := cryptodlinkzg.CommitRect(translatedRectangle, fixture.monolithic)
	if err != nil {
		t.Fatal(err)
	}
	wantSourceCommitment := foldG1(fixture.instance.SourceCommitments[:], challenges.xi)
	if !sourceCommitment.Equal(&wantSourceCommitment) {
		t.Fatal("split semantic source commitment differs from native translated monolithic commitment")
	}
	sourceProof, err := cryptodlinkzg.OpenSourceLink(translatedRectangle, challenges.beta, challenges.z, fixture.monolithic)
	if err != nil {
		t.Fatal(err)
	}
	if !sourceProof.PiZ.Equal(&proof.U3.PiZ) || !sourceProof.PiY.Equal(&proof.U3.PiY) || !sourceProof.ClaimedValue.Equal(&proof.U2.BatchAtBeta[0]) {
		t.Fatal("split source-link quotient differs from canonical monolithic reference")
	}

	betaInverse := dlinkzgOpeningInverse(challenges.beta)
	gPoints := []fr.Element{challenges.z, challenges.beta, betaInverse}
	gInterpolants := make([][]fr.Element, dlinkzgOpeningCircuitClaims)
	for claim := range gInterpolants {
		gInterpolants[claim], err = cryptodlinkzg.Interpolate(gPoints, []fr.Element{
			proof.U1.LinkEvaluations[claim],
			proof.U2.PartialAtBeta[claim],
			proof.U2.PartialAtBetaInverse[claim],
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	numeratorG, err := cryptodlinkzg.FoldSameSetCommitments(proof.U0.PartialCommitments[:], gInterpolants, challenges.kappa, fixture.monolithic)
	if err != nil {
		t.Fatal(err)
	}
	lPoints := []fr.Element{challenges.beta, betaInverse}
	lInterpolants := make([][]fr.Element, 4)
	for polynomial := range lInterpolants {
		lInterpolants[polynomial], err = cryptodlinkzg.Interpolate(lPoints, []fr.Element{
			proof.U2.BatchAtBeta[polynomial],
			proof.U2.BatchAtBetaInverse[polynomial],
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	lCommitments := []bn254.G1Affine{
		proof.U1.LinkCommitment,
		fixture.instance.TreeCommitments[0],
		fixture.instance.TreeCommitments[1],
		proof.U1.LaurentCommitment,
	}
	numeratorL, err := cryptodlinkzg.FoldSameSetCommitments(lCommitments, lInterpolants, challenges.kappa, fixture.monolithic)
	if err != nil {
		t.Fatal(err)
	}
	innerScale := fr.One()
	for i := 0; i < dlinkzgOpeningCircuitClaims; i++ {
		innerScale.Mul(&innerScale, &challenges.kappa)
	}
	statement := cryptodlinkzg.DeltaBatchStatement{
		SourceCommitment: sourceCommitment,
		SourceValue:      proof.U2.BatchAtBeta[0],
		Beta:             challenges.beta,
		ZChallenge:       challenges.z,
		OuterNumerator:   numeratorG,
		InnerNumerator:   numeratorL,
		OuterVanishing:   cryptodlinkzg.VanishingPolynomial(gPoints),
		InnerVanishing:   cryptodlinkzg.VanishingPolynomial(lPoints),
		InnerScale:       innerScale,
	}
	batchProof := cryptodlinkzg.DeltaBatchProof{PiZ: proof.U3.PiZ, PiY: proof.U3.PiY, WN: proof.U3.WN}
	if err := cryptodlinkzg.VerifyDeltaBatch(statement, batchProof, challenges.delta, fixture.monolithic); err != nil {
		t.Fatalf("monolithic final verifier rejected split proof: %v", err)
	}
}

type dlinkzgOpeningChallengeValues struct {
	xi, nu, z, beta, kappa, delta fr.Element
}

func replayDLinkZGOpeningChallenges(t testing.TB, instance DLinkZGOpeningInstance, proof DLinkZGOpeningProof) dlinkzgOpeningChallengeValues {
	t.Helper()
	transcript, err := NewOpeningTranscript(instance.TranscriptContext)
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU0(proof.U0); err != nil {
		t.Fatal(err)
	}
	xi, err := transcript.DeriveXi()
	if err != nil {
		t.Fatal(err)
	}
	nu, err := transcript.DeriveNu()
	if err != nil {
		t.Fatal(err)
	}
	z, err := transcript.DeriveZChallenge()
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU1(proof.U1); err != nil {
		t.Fatal(err)
	}
	beta, err := transcript.DeriveBeta()
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU2(proof.U2); err != nil {
		t.Fatal(err)
	}
	kappa, err := transcript.DeriveKappa()
	if err != nil {
		t.Fatal(err)
	}
	if err := transcript.AppendU3(proof.U3); err != nil {
		t.Fatal(err)
	}
	delta, err := transcript.DeriveDelta()
	if err != nil {
		t.Fatal(err)
	}
	return dlinkzgOpeningChallengeValues{xi: xi.Value, nu: nu.Value, z: z.Value, beta: beta.Value, kappa: kappa.Value, delta: delta.Value}
}

func dlinkzgOpeningTestContext(seed int64) TranscriptContext {
	nonce := [32]byte{}
	nonce[0] = byte(seed + 1)
	nonce[31] = byte(seed + 17)
	return TranscriptContext{
		ProtocolVersion:       "dlinkzg-opening-compiler-test/v1",
		PublicStatementDigest: NewTranscriptDigest([]byte(fmt.Sprintf("statement/%d", seed))),
		SRSDigest:             NewTranscriptDigest([]byte(fmt.Sprintf("srs/%d", seed))),
		IndexDigest:           NewTranscriptDigest([]byte(fmt.Sprintf("index/%d", seed))),
		PartyManifestDigest:   NewTranscriptDigest([]byte(fmt.Sprintf("manifest/%d", seed))),
		SessionNonce:          nonce,
		PriorTranscriptDigest: NewTranscriptDigest([]byte(fmt.Sprintf("prior/%d", seed))),
	}
}

func cloneDLinkZGOpeningInstance(instance DLinkZGOpeningInstance) DLinkZGOpeningInstance {
	result := instance
	result.PartitionPoint = append([]fr.Element(nil), instance.PartitionPoint...)
	return result
}

func cloneDLinkZGOpeningProverInput(input DLinkZGOpeningProverInput) DLinkZGOpeningProverInput {
	result := input
	result.T0 = append([]fr.Element(nil), input.T0...)
	result.T1 = append([]fr.Element(nil), input.T1...)
	result.PartySRS = append([]*cryptodlinkzg.PartyRowSRS(nil), input.PartySRS...)
	for claim := range input.Sources {
		result.Sources[claim] = make([][]fr.Element, len(input.Sources[claim]))
		for rank := range input.Sources[claim] {
			result.Sources[claim][rank] = append([]fr.Element(nil), input.Sources[claim][rank]...)
		}
	}
	return result
}

func assertDLinkZGOpeningDigestRejects(t testing.TB, tampered DLinkZGOpeningInstance, proof DLinkZGOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS) {
	t.Helper()
	if tampered.TranscriptContext.PublicStatementDigest == DLinkZGOpeningInstanceDigest(tampered) {
		t.Fatal("tampered instance unexpectedly retained its canonical digest")
	}
	if err := VerifyDLinkZGOpening(tampered, proof, verifierSRS); !errors.Is(err, ErrInvalidDLinkZGOpeningInstance) {
		t.Fatalf("cancelling tamper: got %v, want ErrInvalidDLinkZGOpeningInstance", err)
	}
}

func dlinkzgOpeningTestAddScaledG1(base, adjustment bn254.G1Affine, scale fr.Element) bn254.G1Affine {
	scaled := scaleG1Prototype(adjustment, scale)
	var baseJacobian, scaledJacobian bn254.G1Jac
	baseJacobian.FromAffine(&base)
	scaledJacobian.FromAffine(&scaled)
	baseJacobian.AddAssign(&scaledJacobian)
	var result bn254.G1Affine
	result.FromJacobian(&baseJacobian)
	return result
}

func dlinkzgOpeningTestG1(scalar int64) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var result bn254.G1Affine
	result.ScalarMultiplication(&generator, big.NewInt(scalar))
	return result
}

func dlinkzgOpeningDynamicFieldPath(typ reflect.Type, path string) (string, bool) {
	switch typ.Kind() {
	case reflect.Slice, reflect.Map, reflect.String, reflect.Interface, reflect.Pointer:
		return path, true
	case reflect.Array:
		return dlinkzgOpeningDynamicFieldPath(typ.Elem(), path+"[]")
	case reflect.Struct:
		for field := 0; field < typ.NumField(); field++ {
			fieldPath := path + "." + typ.Field(field).Name
			if dynamicPath, found := dlinkzgOpeningDynamicFieldPath(typ.Field(field).Type, fieldPath); found {
				return dynamicPath, true
			}
		}
	}
	return "", false
}

type dlinkzgOpeningFixedTranscript struct {
	step                          int
	xi, nu, z, beta, kappa, delta fr.Element
}

func (t *dlinkzgOpeningFixedTranscript) advance(want int) error {
	if t.step != want {
		return ErrTranscriptOrder
	}
	t.step++
	return nil
}

func (t *dlinkzgOpeningFixedTranscript) AppendU0(U0Message) error { return t.advance(0) }
func (t *dlinkzgOpeningFixedTranscript) DeriveXi() (ChallengeOut, error) {
	if err := t.advance(1); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeXi, Value: t.xi}, nil
}
func (t *dlinkzgOpeningFixedTranscript) DeriveNu() (ChallengeOut, error) {
	if err := t.advance(2); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeNu, Value: t.nu}, nil
}
func (t *dlinkzgOpeningFixedTranscript) DeriveZChallenge() (ChallengeOut, error) {
	if err := t.advance(3); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeZ, Value: t.z}, nil
}
func (t *dlinkzgOpeningFixedTranscript) AppendU1(U1Message) error { return t.advance(4) }
func (t *dlinkzgOpeningFixedTranscript) DeriveBeta() (ChallengeOut, error) {
	if err := t.advance(5); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeBeta, Value: t.beta}, nil
}
func (t *dlinkzgOpeningFixedTranscript) AppendU2(U2Message) error { return t.advance(6) }
func (t *dlinkzgOpeningFixedTranscript) DeriveKappa() (ChallengeOut, error) {
	if err := t.advance(7); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeKappa, Value: t.kappa}, nil
}
func (t *dlinkzgOpeningFixedTranscript) AppendU3(U3Message) error { return t.advance(8) }
func (t *dlinkzgOpeningFixedTranscript) DeriveDelta() (ChallengeOut, error) {
	if err := t.advance(9); err != nil {
		return ChallengeOut{}, err
	}
	return ChallengeOut{ID: ChallengeDelta, Value: t.delta}, nil
}
