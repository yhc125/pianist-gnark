package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
)

func TestTerminalSourceCommitmentsMatchDirectAdapterFastPIOP(t *testing.T) {
	const rounds = 16
	spr, witness := compileLocalPIOPAdapterCircuit(t, rounds, 2, fr.Element{})
	for _, parties := range []int{2, 4} {
		t.Run(adapterWorldName(parties), func(t *testing.T) {
			extracted := extractAllLocalPIOPRanks(t, spr, witness, parties)
			piopChallenges, _ := buildAllExtractedLocalRelations(t, extracted)
			domainSize := len(extracted[0].table.Wires[0])
			alpha := sourceCommitmentAlphaOutsideDomain(t, domainSize)
			challenges := LocalTerminalSourceCommitmentChallenges{
				EtaPart: piopChallenges.EtaPart,
				EtaX:    piopChallenges.EtaX,
				Gamma:   piopChallenges.Gamma,
				Alpha:   alpha,
				Mu: [LocalCompressedSourceCount]fr.Element{
					fr.NewElement(29), fr.NewElement(31), fr.NewElement(37),
				},
			}
			tauY := fr.NewElement(41)
			tauZ := fr.NewElement(43)
			sigma := fr.NewElement(47)

			slots := make([]LocalPIOPFixedCommitmentSlot, parties)
			public := AggregateLocalPIOPPublicCommitments{
				Parties:    parties,
				DomainSize: domainSize,
			}
			var semanticDirect, nativeShiftedDirect [LocalCompressedSourceCount]bn254.G1Affine
			for rank := 0; rank < parties; rank++ {
				preprocessing, err := PreprocessLocalPIOP(spr, rank, parties)
				if err != nil {
					t.Fatalf("preprocess rank %d: %v", rank, err)
				}
				rowSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(
					parties, domainSize, rank, tauY, tauZ, sigma,
				)
				if err != nil {
					t.Fatalf("row SRS rank %d: %v", rank, err)
				}
				slots[rank], err = SetupLocalPIOPFixedCommitmentSlot(preprocessing, rowSRS)
				if err != nil {
					t.Fatalf("fixed commitments rank %d: %v", rank, err)
				}

				relation, w0, w1 := buildFastLocalPIOPForTest(
					t, extracted[rank].table, extracted[rank].index, piopChallenges,
				)
				for wire := 0; wire < LocalWireCount; wire++ {
					commitment, err := rowSRS.CommitSemantic(w0.Wires[wire])
					if err != nil {
						t.Fatalf("W0 rank %d wire %d: %v", rank, wire, err)
					}
					sourceCommitmentAdd(&public.W0[wire], &commitment)
				}
				commitment, err := rowSRS.CommitSemantic(w1.Accumulator)
				if err != nil {
					t.Fatalf("W1 rank %d: %v", rank, err)
				}
				sourceCommitmentAdd(&public.W1, &commitment)
				for chunk := range relation.Quotient {
					commitment, err = rowSRS.CommitSemantic(relation.Quotient[chunk])
					if err != nil {
						t.Fatalf("W2 rank %d chunk %d: %v", rank, chunk, err)
					}
					sourceCommitmentAdd(&public.W2[chunk], &commitment)
				}

				compression, err := CompressLocalTerminalSources(relation, alpha, challenges.Mu)
				if err != nil {
					t.Fatalf("compress rank %d terminal sources: %v", rank, err)
				}
				for group := 0; group < LocalCompressedSourceCount; group++ {
					semantic, err := rowSRS.CommitSemantic(compression.Polynomials[group])
					if err != nil {
						t.Fatalf("semantic direct rank %d group %d: %v", rank, group, err)
					}
					translated := cryptodlinkzg.FastTaylorShift(compression.Polynomials[group], sigma)
					native, err := rowSRS.CommitRow(translated)
					if err != nil {
						t.Fatalf("native shifted direct rank %d group %d: %v", rank, group, err)
					}
					if !semantic.Equal(&native) {
						t.Fatalf("rank %d group %d semantic/native shifted commitments differ", rank, group)
					}
					sourceCommitmentAdd(&semanticDirect[group], &semantic)
					sourceCommitmentAdd(&nativeShiftedDirect[group], &native)
				}
			}

			fixed, err := AggregateLocalPIOPFixedCommitmentSlots(slots)
			if err != nil {
				t.Fatalf("aggregate fixed commitments: %v", err)
			}
			derived, err := DeriveLocalTerminalSourceCommitments(
				AggregateLocalPIOPCommitmentView{Fixed: fixed, Public: public},
				domainSize,
				extracted[0].index.WireCosets,
				challenges,
			)
			if err != nil {
				t.Fatalf("derive terminal source commitments: %v", err)
			}
			for group := 0; group < LocalCompressedSourceCount; group++ {
				if !derived.Commitments[group].Equal(&semanticDirect[group]) {
					t.Fatalf("group %d derived commitment differs from direct semantic aggregate", group)
				}
				if !derived.Commitments[group].Equal(&nativeShiftedDirect[group]) {
					t.Fatalf("group %d derived commitment differs from direct translated native aggregate", group)
				}
			}
		})
	}
}

func TestFixedCommitmentSetupIsWitnessIndependent(t *testing.T) {
	const rounds = 16
	spr, firstWitness := compileLocalPIOPAdapterCircuit(t, rounds, 2, fr.Element{})
	secondX := fr.NewElement(3)
	secondY := secondX
	for round := 0; round < rounds; round++ {
		secondY.Mul(&secondY, &secondX)
	}
	secondWitness := localPIOPConcreteWitness(t, &localPIOPAdapterCircuit{
		X: secondX, Y: secondY, rounds: rounds,
	})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	firstSolution, err := spr.Solve(firstWitness, proverConfig)
	if err != nil {
		t.Fatalf("solve first witness: %v", err)
	}
	secondSolution, err := spr.Solve(secondWitness, proverConfig)
	if err != nil {
		t.Fatalf("solve second witness: %v", err)
	}

	preprocessing, err := PreprocessLocalPIOP(spr, 0, 2)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	rowSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(
		2, preprocessing.LocalRows(), 0,
		fr.NewElement(53), fr.NewElement(59), fr.NewElement(61),
	)
	if err != nil {
		t.Fatalf("row SRS: %v", err)
	}
	before, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, rowSRS)
	if err != nil {
		t.Fatalf("fixed commitments before witnesses: %v", err)
	}
	firstTable, _, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, firstSolution)
	if err != nil {
		t.Fatalf("first online table: %v", err)
	}
	secondTable, _, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, secondSolution)
	if err != nil {
		t.Fatalf("second online table: %v", err)
	}
	if equalLocalPolynomials(firstTable.Wires[LocalWireA], secondTable.Wires[LocalWireA]) {
		t.Fatal("distinct witnesses produced the same local A wire")
	}
	after, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, rowSRS)
	if err != nil {
		t.Fatalf("fixed commitments after witnesses: %v", err)
	}
	assertSourceCommitmentSlotsEqual(t, before, after)
	for wire := 0; wire < LocalWireCount; wire++ {
		if !equalLocalPolynomials(firstTable.SigmaX[wire], secondTable.SigmaX[wire]) ||
			!equalLocalPolynomials(firstTable.SigmaPart[wire], secondTable.SigmaPart[wire]) {
			t.Fatalf("witness changed fixed wiring column %d", wire)
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if !equalLocalPolynomials(firstTable.Selectors[selector], secondTable.Selectors[selector]) {
			t.Fatalf("witness changed selector %d", selector)
		}
	}
}

func TestFixedCommitmentSetupAndAggregationRejectMalformedInputs(t *testing.T) {
	if LocalPIOPFixedCommitmentsPerSlot != 13 {
		t.Fatalf("fixed commitments per slot = %d, want 13", LocalPIOPFixedCommitmentsPerSlot)
	}
	spr, _ := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
	preprocessing, err := PreprocessLocalPIOP(spr, 0, 2)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	tauY := fr.NewElement(67)
	tauZ := fr.NewElement(71)
	sigma := fr.NewElement(73)

	if _, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, nil); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("nil SRS error = %v", err)
	}
	wrongRank, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(2, preprocessing.LocalRows(), 1, tauY, tauZ, sigma)
	if err != nil {
		t.Fatalf("wrong-rank SRS fixture: %v", err)
	}
	if _, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, wrongRank); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("wrong-rank SRS error = %v", err)
	}
	shortSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(2, 4, 0, tauY, tauZ, sigma)
	if err != nil {
		t.Fatalf("short SRS fixture: %v", err)
	}
	if _, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, shortSRS); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("short SRS error = %v", err)
	}
	malformedSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(2, preprocessing.LocalRows(), 0, tauY, tauZ, sigma)
	if err != nil {
		t.Fatalf("malformed SRS fixture: %v", err)
	}
	malformedSRS.G1SemanticRow = malformedSRS.G1SemanticRow[:len(malformedSRS.G1SemanticRow)-1]
	if _, err := SetupLocalPIOPFixedCommitmentSlot(preprocessing, malformedSRS); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("malformed SRS error = %v", err)
	}

	slots := sourceCommitmentFixedSlotsForTest(t, spr, 2, tauY, tauZ, sigma)
	if _, err := AggregateLocalPIOPFixedCommitmentSlots(slots[:1]); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("short slot vector error = %v", err)
	}
	reordered := append([]LocalPIOPFixedCommitmentSlot(nil), slots...)
	reordered[0], reordered[1] = reordered[1], reordered[0]
	if _, err := AggregateLocalPIOPFixedCommitmentSlots(reordered); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("reordered slot vector error = %v", err)
	}
	badLabel := append([]LocalPIOPFixedCommitmentSlot(nil), slots...)
	badLabel[1].SlotLabel = fr.NewElement(99)
	if _, err := AggregateLocalPIOPFixedCommitmentSlots(badLabel); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("tampered slot label error = %v", err)
	}
	badPoint := append([]LocalPIOPFixedCommitmentSlot(nil), slots...)
	badPoint[0].Selectors[0].X.SetUint64(1)
	badPoint[0].Selectors[0].Y.SetUint64(1)
	if _, err := AggregateLocalPIOPFixedCommitmentSlots(badPoint); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("malformed fixed point error = %v", err)
	}
	badProtocolShape := append([]LocalPIOPFixedCommitmentSlot(nil), slots...)
	for rank := range badProtocolShape {
		badProtocolShape[rank].DomainSize = 2
	}
	if _, err := AggregateLocalPIOPFixedCommitmentSlots(badProtocolShape); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("T<4 protocol shape error = %v", err)
	}
	if _, err := sourceCommitmentLinearCombination(make([]bn254.G1Affine, 1), nil); !errors.Is(err, ErrInvalidLocalPIOPPublicCommitments) {
		t.Fatalf("linear-combination shape error = %v", err)
	}
}

func TestTerminalSourceCommitmentDerivationRejectsAndDetectsTampering(t *testing.T) {
	view, wireCosets, challenges := sourceCommitmentViewForTest(t, 2)
	derived, err := DeriveLocalTerminalSourceCommitments(view, view.Fixed.DomainSize, wireCosets, challenges)
	if err != nil {
		t.Fatalf("derive baseline: %v", err)
	}

	tampered := view
	increment := testG1(101)
	sourceCommitmentAdd(&tampered.Public.W0[LocalWireA], &increment)
	tamperedDerived, err := DeriveLocalTerminalSourceCommitments(tampered, tampered.Fixed.DomainSize, wireCosets, challenges)
	if err != nil {
		t.Fatalf("derive valid-point tamper: %v", err)
	}
	if tamperedDerived.Commitments[LocalTerminalAtAlpha].Equal(&derived.Commitments[LocalTerminalAtAlpha]) ||
		tamperedDerived.Commitments[LocalTerminalAtXStar].Equal(&derived.Commitments[LocalTerminalAtXStar]) {
		t.Fatal("W0 tamper did not change both source groups containing phi_a")
	}

	metadataMismatch := view
	metadataMismatch.Public.Parties++
	if _, err := DeriveLocalTerminalSourceCommitments(metadataMismatch, view.Fixed.DomainSize, wireCosets, challenges); !errors.Is(err, ErrInvalidLocalPIOPPublicCommitments) {
		t.Fatalf("metadata mismatch error = %v", err)
	}
	malformedPublic := view
	malformedPublic.Public.W2[0].X.SetUint64(1)
	malformedPublic.Public.W2[0].Y.SetUint64(1)
	if _, err := DeriveLocalTerminalSourceCommitments(malformedPublic, view.Fixed.DomainSize, wireCosets, challenges); !errors.Is(err, ErrInvalidLocalPIOPPublicCommitments) {
		t.Fatalf("malformed W2 error = %v", err)
	}
	malformedFixed := view
	malformedFixed.Fixed.Public.One.X.SetUint64(1)
	malformedFixed.Fixed.Public.One.Y.SetUint64(1)
	if _, err := DeriveLocalTerminalSourceCommitments(malformedFixed, view.Fixed.DomainSize, wireCosets, challenges); !errors.Is(err, ErrInvalidLocalPIOPFixedCommitments) {
		t.Fatalf("malformed fixed aggregate error = %v", err)
	}
	badCosets := wireCosets
	badCosets[1] = fr.Element{}
	if _, err := DeriveLocalTerminalSourceCommitments(view, view.Fixed.DomainSize, badCosets, challenges); !errors.Is(err, ErrInvalidLocalPIOPPublicCommitments) {
		t.Fatalf("bad wire cosets error = %v", err)
	}
	badAlpha := challenges
	badAlpha.Alpha.SetOne()
	if _, err := DeriveLocalTerminalSourceCommitments(view, view.Fixed.DomainSize, wireCosets, badAlpha); !errors.Is(err, ErrInvalidLocalTerminalPoint) {
		t.Fatalf("in-domain alpha error = %v", err)
	}
	if _, err := DeriveLocalTerminalSourceCommitments(view, view.Fixed.DomainSize/2, wireCosets, challenges); !errors.Is(err, ErrInvalidLocalPIOPPublicCommitments) {
		t.Fatalf("wrong domain size error = %v", err)
	}
}

func sourceCommitmentViewForTest(
	t *testing.T,
	parties int,
) (AggregateLocalPIOPCommitmentView, [LocalWireCount]fr.Element, LocalTerminalSourceCommitmentChallenges) {
	t.Helper()
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	extracted := extractAllLocalPIOPRanks(t, spr, witness, parties)
	piopChallenges, _ := buildAllExtractedLocalRelations(t, extracted)
	domainSize := len(extracted[0].table.Wires[0])
	tauY := fr.NewElement(79)
	tauZ := fr.NewElement(83)
	sigma := fr.NewElement(89)
	slots := sourceCommitmentFixedSlotsForTest(t, spr, parties, tauY, tauZ, sigma)
	fixed, err := AggregateLocalPIOPFixedCommitmentSlots(slots)
	if err != nil {
		t.Fatalf("aggregate fixed fixture: %v", err)
	}
	public := AggregateLocalPIOPPublicCommitments{Parties: parties, DomainSize: domainSize}
	for rank := 0; rank < parties; rank++ {
		rowSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(parties, domainSize, rank, tauY, tauZ, sigma)
		if err != nil {
			t.Fatalf("row SRS rank %d: %v", rank, err)
		}
		relation, w0, w1 := buildFastLocalPIOPForTest(t, extracted[rank].table, extracted[rank].index, piopChallenges)
		for wire := 0; wire < LocalWireCount; wire++ {
			commitment, err := rowSRS.CommitSemantic(w0.Wires[wire])
			if err != nil {
				t.Fatalf("fixture W0 rank %d wire %d: %v", rank, wire, err)
			}
			sourceCommitmentAdd(&public.W0[wire], &commitment)
		}
		commitment, err := rowSRS.CommitSemantic(w1.Accumulator)
		if err != nil {
			t.Fatalf("fixture W1 rank %d: %v", rank, err)
		}
		sourceCommitmentAdd(&public.W1, &commitment)
		for chunk := range relation.Quotient {
			commitment, err = rowSRS.CommitSemantic(relation.Quotient[chunk])
			if err != nil {
				t.Fatalf("fixture W2 rank %d chunk %d: %v", rank, chunk, err)
			}
			sourceCommitmentAdd(&public.W2[chunk], &commitment)
		}
	}
	challenges := LocalTerminalSourceCommitmentChallenges{
		EtaPart: piopChallenges.EtaPart,
		EtaX:    piopChallenges.EtaX,
		Gamma:   piopChallenges.Gamma,
		Alpha:   sourceCommitmentAlphaOutsideDomain(t, domainSize),
		Mu: [LocalCompressedSourceCount]fr.Element{
			fr.NewElement(97), fr.NewElement(101), fr.NewElement(103),
		},
	}
	return AggregateLocalPIOPCommitmentView{Fixed: fixed, Public: public}, extracted[0].index.WireCosets, challenges
}

func sourceCommitmentFixedSlotsForTest(
	t *testing.T,
	spr *cs.SparseR1CS,
	parties int,
	tauY, tauZ, sigma fr.Element,
) []LocalPIOPFixedCommitmentSlot {
	t.Helper()
	result := make([]LocalPIOPFixedCommitmentSlot, parties)
	for rank := 0; rank < parties; rank++ {
		preprocessing, err := PreprocessLocalPIOP(spr, rank, parties)
		if err != nil {
			t.Fatalf("preprocess fixed rank %d: %v", rank, err)
		}
		rowSRS, err := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(
			parties, preprocessing.LocalRows(), rank, tauY, tauZ, sigma,
		)
		if err != nil {
			t.Fatalf("fixed row SRS rank %d: %v", rank, err)
		}
		result[rank], err = SetupLocalPIOPFixedCommitmentSlot(preprocessing, rowSRS)
		if err != nil {
			t.Fatalf("fixed slot rank %d: %v", rank, err)
		}
	}
	return result
}

func sourceCommitmentAlphaOutsideDomain(t *testing.T, domainSize int) fr.Element {
	t.Helper()
	for candidate := uint64(11); candidate < 128; candidate++ {
		alpha := fr.NewElement(candidate)
		if validateAlphaOutsideLocalDomain(alpha, domainSize) == nil {
			return alpha
		}
	}
	t.Fatal("could not find terminal alpha outside local domain")
	return fr.Element{}
}

func assertSourceCommitmentSlotsEqual(t *testing.T, got, want LocalPIOPFixedCommitmentSlot) {
	t.Helper()
	if got.Rank != want.Rank || got.Parties != want.Parties || got.DomainSize != want.DomainSize || !got.SlotLabel.Equal(&want.SlotLabel) {
		t.Fatal("fixed slot metadata differs")
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if !got.Selectors[selector].Equal(&want.Selectors[selector]) {
			t.Fatalf("fixed selector commitment %d differs", selector)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !got.SigmaX[wire].Equal(&want.SigmaX[wire]) || !got.SigmaPart[wire].Equal(&want.SigmaPart[wire]) {
			t.Fatalf("fixed wiring commitments %d differ", wire)
		}
	}
	if !got.One.Equal(&want.One) || !got.X.Equal(&want.X) {
		t.Fatal("fixed public-family basis commitments differ")
	}
}
