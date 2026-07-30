package dlinkzg

import (
	"math/bits"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

func TestHybridOpeningHonestRandomized(t *testing.T) {
	for iteration := 0; iteration < 32; iteration++ {
		parties := 2 << uint(iteration%3)
		degreeBound := 16
		fixture := newHybridOpeningTestFixture(t, parties, degreeBound, int64(5000+iteration))
		proof, err := ProveHybridOpening(fixture.instance, fixture.input)
		if err != nil {
			t.Fatalf("iteration %d prove: %v", iteration, err)
		}
		if err := VerifyHybridOpening(fixture.instance, proof, fixture.input.VerifierSRS); err != nil {
			t.Fatalf("iteration %d verify: %v", iteration, err)
		}
	}
}

func TestHybridOpeningRejectsTampering(t *testing.T) {
	fixture := newHybridOpeningTestFixture(t, 4, 16, 7001)
	proof, err := ProveHybridOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	proof.U2.CircuitCrossValues[0][0].Add(&proof.U2.CircuitCrossValues[0][0], valuePointer(fr.One()))
	if err := VerifyHybridOpening(fixture.instance, proof, fixture.input.VerifierSRS); err == nil {
		t.Fatal("tampered hybrid cross-value accepted")
	}

	proof, err = ProveHybridOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	proof.U2.LaurentAtBeta[3].Add(&proof.U2.LaurentAtBeta[3], valuePointer(fr.One()))
	if err := VerifyHybridOpening(fixture.instance, proof, fixture.input.VerifierSRS); err == nil {
		t.Fatal("tampered hybrid Laurent value accepted")
	}
}

func TestCompiledMPIHybridLaurentRecurrenceMatchesPolynomial(t *testing.T) {
	random := rand.New(rand.NewSource(8103))
	for _, size := range []int{2, 4, 8, 16} {
		for iteration := 0; iteration < 32; iteration++ {
			vectors := [6][]fr.Element{}
			for vector := range vectors {
				vectors[vector] = make([]fr.Element, size)
				for index := range vectors[vector] {
					vectors[vector][index] = fr.NewElement(uint64(1 + random.Int63n(1<<30)))
				}
			}
			point := fr.NewElement(uint64(2 + random.Int63n(1<<30)))
			polynomial := cryptodlinkzg.BuildLocalLaurent(cryptodlinkzg.LocalLaurentInput{
				HXi: vectors[0], PsiR: vectors[3], Nu: fr.One(),
				T0: vectors[1], T1: vectors[2], AXi: vectors[4], BXi: vectors[5],
			})
			want := cryptodlinkzg.Eval(polynomial, point)
			got := compiledMPIHybridLaurentValue(
				vectors[0], vectors[1], vectors[2], vectors[3], vectors[4], vectors[5], point,
			)
			if !got.Equal(&want) {
				t.Fatalf("M=%d iteration=%d recurrence mismatch", size, iteration)
			}
		}
	}
}

func TestHybridOpeningTranscriptBindsInstance(t *testing.T) {
	fixture := newHybridOpeningTestFixture(t, 4, 16, 8001)
	proof, err := ProveHybridOpening(fixture.instance, fixture.input)
	if err != nil {
		t.Fatal(err)
	}
	mutated := fixture.instance
	mutated.CircuitClaims[0].Add(&mutated.CircuitClaims[0], valuePointer(fr.One()))
	if err := VerifyHybridOpening(mutated, proof, fixture.input.VerifierSRS); err == nil {
		t.Fatal("hybrid proof accepted for a mutated instance")
	}
}

type hybridOpeningTestFixture struct {
	instance DLinkZGOpeningInstance
	input    DLinkZGOpeningProverInput
}

func newHybridOpeningTestFixture(t testing.TB, parties, degreeBound int, seed int64) hybridOpeningTestFixture {
	t.Helper()
	random := rand.New(rand.NewSource(seed))
	tauY := fr.NewElement(uint64(4001 + seed))
	tauZ := fr.NewElement(uint64(5003 + seed))
	sigma := fr.NewElement(uint64(6007 + seed))
	partySRS := make([]*cryptodlinkzg.PartyRowSRS, parties)
	for rank := range partySRS {
		row, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(parties, degreeBound, rank, tauY, tauZ, sigma)
		if err != nil {
			t.Fatal(err)
		}
		partySRS[rank] = row
	}
	coordinatorSRS, err := cryptodlinkzg.NewDeterministicCoordinatorSRSWithShift(parties, tauY, tauZ, sigma)
	if err != nil {
		t.Fatal(err)
	}
	verifierSRS := cryptodlinkzg.NewDeterministicVerifierSRSWithShift(tauY, tauZ, sigma)

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
		if partitionPoint[i].Equal(valuePointer(fr.One())) {
			partitionPoint[i].SetUint64(2)
		}
	}
	weights := cryptodlinkzg.EqualityWeights(partitionPoint)
	semantic := [3]fr.Element{fr.NewElement(2), fr.NewElement(7), fr.NewElement(11)}
	instance := DLinkZGOpeningInstance{
		LocalDomainSize:     degreeBound,
		SemanticQueryPoints: semantic,
		Shift:               sigma,
		PartitionPoint:      append([]fr.Element(nil), partitionPoint...),
		TranscriptContext:   dlinkzgOpeningTestContext(seed),
	}
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		var commitment bn254.G1Jac
		for rank := 0; rank < parties; rank++ {
			rowCommitment, commitErr := partySRS[rank].CommitSemantic(sources[claim][rank])
			if commitErr != nil {
				t.Fatal(commitErr)
			}
			dlinkzgOpeningAddG1(&commitment, rowCommitment)
			value := cryptodlinkzg.Eval(sources[claim][rank], semantic[claim])
			var weighted fr.Element
			weighted.Mul(&weights[rank], &value)
			instance.CircuitClaims[claim].Add(&instance.CircuitClaims[claim], &weighted)
		}
		instance.SourceCommitments[claim].FromJacobian(&commitment)
	}
	instance.TreeCommitments[0], err = coordinatorSRS.CommitU(t0)
	if err != nil {
		t.Fatal(err)
	}
	instance.TreeCommitments[1], err = coordinatorSRS.CommitU(t1)
	if err != nil {
		t.Fatal(err)
	}
	treePoints := productCheckFunctionalPoints(partitionPoint)
	for claim := 0; claim < dlinkzgOpeningTreeClaims; claim++ {
		instance.TreeClaims[claim], err = evaluateTreeFunctional(t0, t1, treePoints[claim])
		if err != nil {
			t.Fatal(err)
		}
	}
	instance.TranscriptContext = BindHybridOpeningTranscriptContext(instance, instance.TranscriptContext)
	return hybridOpeningTestFixture{
		instance: instance,
		input: DLinkZGOpeningProverInput{
			Sources:        sources,
			T0:             t0,
			T1:             t1,
			PartySRS:       partySRS,
			CoordinatorSRS: coordinatorSRS,
			VerifierSRS:    verifierSRS,
		},
	}
}
