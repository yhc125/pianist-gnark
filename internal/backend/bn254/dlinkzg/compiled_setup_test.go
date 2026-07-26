package dlinkzg

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend"
)

func TestDeterministicCompiledSetupRealSparseR1CSRoles(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}

	for _, partitions := range []int{2, 4} {
		t.Run(adapterWorldName(partitions), func(t *testing.T) {
			setup, err := NewDeterministicCompiledSetup(
				spr, partitions, fr.NewElement(101), fr.NewElement(103),
			)
			if err != nil {
				t.Fatalf("new setup: %v", err)
			}
			if err := setup.Validate(); err != nil {
				t.Fatalf("validate setup: %v", err)
			}
			if err := setup.ValidateOnlineRoles(); err != nil {
				t.Fatalf("validate online roles: %v", err)
			}
			if DeterministicCompiledSetupNotice == "" || !strings.Contains(DeterministicCompiledSetupNotice, "benchmark/test only") {
				t.Fatalf("missing benchmark/test-only notice: %q", DeterministicCompiledSetupNotice)
			}
			if setup.Metadata.PartitionCount != partitions || setup.Metadata.LocalDomainSize < 4 ||
				setup.Metadata.LocalDomainSize > FastLocalPIOPMaxDomainSize {
				t.Fatalf("metadata shape = M=%d,T=%d", setup.Metadata.PartitionCount, setup.Metadata.LocalDomainSize)
			}
			if setup.Metadata.CircuitDigest != TranscriptDigest(sparseR1CSAdapterDigest(spr)) {
				t.Fatal("metadata does not bind the SparseR1CS digest")
			}
			if err := VerifyIndexShift(
				setup.Metadata.IndexPrefixDigest,
				uint64(setup.Metadata.LocalDomainSize),
				setup.Metadata.LocalDomainGenerator,
				setup.Metadata.IndexShift.Counter,
				setup.Metadata.IndexShift.Sigma,
			); err != nil {
				t.Fatalf("verify setup shift: %v", err)
			}

			if len(setup.Parties) != partitions || setup.Coordinator.SRS.Parties != partitions {
				t.Fatalf("role count mismatch")
			}
			if len(setup.Coordinator.SRS.G1Y) != partitions ||
				len(setup.Coordinator.SRS.G1ZShared) != partitions {
				t.Fatal("coordinator does not have exactly its O(M) role view")
			}
			if len(setup.Verifier.SRS.G1ZVerifier) != 3 || len(setup.Verifier.SRS.G2Y) != 2 ||
				len(setup.Verifier.SRS.G2Z) != 4 {
				t.Fatal("verifier does not have the constant role view")
			}

			for rank := range setup.Parties {
				party := &setup.Parties[rank]
				if party.Rank != rank || party.RowSRS.Rank != rank || party.RowSRS.Parties != partitions {
					t.Fatalf("rank %d role metadata mismatch", rank)
				}
				if party.ConstraintSystem != spr {
					t.Fatalf("rank %d missing trapdoor-free online constraint system", rank)
				}
				if len(party.RowSRS.G1SemanticRow) != setup.Metadata.LocalDomainSize ||
					len(party.RowSRS.G1Row) != setup.Metadata.LocalDomainSize ||
					len(party.RowSRS.G1ZShared) != setup.Metadata.LocalDomainSize {
					t.Fatalf("rank %d does not have exactly three O(T) rows", rank)
				}
				table, index, err := BuildLocalPIOPTableFromSolution(
					party.Preprocessing, party.ConstraintSystem, solution,
				)
				if err != nil {
					t.Fatalf("rank %d online table: %v", rank, err)
				}
				wantSlotLabel := fr.NewElement(uint64(rank))
				if !index.SlotLabel.Equal(&wantSlotLabel) {
					t.Fatalf("rank %d online table returned wrong slot label", rank)
				}
				state, err := PrepareFastLocalPIOPWithFixed(table, party.FastFixed)
				if err != nil {
					t.Fatalf("rank %d prepare fast PIOP: %v", rank, err)
				}
				if state.Stage() != FastLocalPIOPW0Ready {
					t.Fatalf("rank %d fast PIOP stage = %d", rank, state.Stage())
				}
			}

			outerContext := compiledSetupOuterContextForTest(setup)
			if err := setup.ValidateOuterContext(outerContext); err != nil {
				t.Fatalf("validate outer context: %v", err)
			}
			openingContext := compiledSetupOpeningContextForTest(setup)
			if err := setup.ValidateOpeningContext(openingContext); err != nil {
				t.Fatalf("validate opening context: %v", err)
			}
		})
	}
}

func TestDeterministicCompiledSetupIsWitnessIndependentAndRetainsNoTrapdoorScalar(t *testing.T) {
	const rounds = 16
	spr, firstWitness := compileLocalPIOPAdapterCircuit(t, rounds, 2, fr.Element{})
	tauY := fr.NewElement(107)
	tauZ := fr.NewElement(109)
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

	for _, partitions := range []int{2, 4} {
		t.Run(adapterWorldName(partitions), func(t *testing.T) {
			setup, err := NewDeterministicCompiledSetup(spr, partitions, tauY, tauZ)
			if err != nil {
				t.Fatalf("new setup: %v", err)
			}
			metadataBefore := setup.Metadata
			fixedBefore := make([]LocalPIOPFixedCommitmentSlot, partitions)
			for rank := range setup.Parties {
				fixedBefore[rank] = setup.Parties[rank].FixedCommitments
			}

			witnessChanged := false
			for rank := range setup.Parties {
				party := &setup.Parties[rank]
				firstTable, _, err := BuildLocalPIOPTableFromSolution(
					party.Preprocessing, party.ConstraintSystem, firstSolution,
				)
				if err != nil {
					t.Fatalf("rank %d first table: %v", rank, err)
				}
				secondTable, _, err := BuildLocalPIOPTableFromSolution(
					party.Preprocessing, party.ConstraintSystem, secondSolution,
				)
				if err != nil {
					t.Fatalf("rank %d second table: %v", rank, err)
				}
				for wire := 0; wire < LocalWireCount; wire++ {
					if !compiledElementsEqual(firstTable.Wires[wire], secondTable.Wires[wire]) {
						witnessChanged = true
					}
				}
				if _, err := PrepareFastLocalPIOPWithFixed(firstTable, party.FastFixed); err != nil {
					t.Fatalf("rank %d first fast path: %v", rank, err)
				}
				if _, err := PrepareFastLocalPIOPWithFixed(secondTable, party.FastFixed); err != nil {
					t.Fatalf("rank %d second fast path: %v", rank, err)
				}
			}
			if !witnessChanged {
				t.Fatal("distinct witnesses did not change any online wire table")
			}
			if setup.Metadata != metadataBefore {
				t.Fatal("online witnesses changed setup metadata")
			}
			for rank := range setup.Parties {
				if !compiledFixedSlotEqual(fixedBefore[rank], setup.Parties[rank].FixedCommitments) {
					t.Fatalf("online witnesses changed rank %d fixed commitments", rank)
				}
			}
			if err := setup.Validate(); err != nil {
				t.Fatalf("validate after two witnesses: %v", err)
			}

			secondSetup, err := NewDeterministicCompiledSetup(spr, partitions, tauY, tauZ)
			if err != nil {
				t.Fatalf("repeat deterministic setup: %v", err)
			}
			if secondSetup.Metadata != setup.Metadata {
				t.Fatal("deterministic setup metadata changed across identical factories")
			}
		})
	}
	compiledAssertNoTrapdoorScalarField(t, reflect.TypeOf(CompiledSetup{}), map[reflect.Type]bool{})
}

func TestDeterministicCompiledSetupRejectsMalformedShapesAndRoles(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	tauY := fr.NewElement(113)
	tauZ := fr.NewElement(127)

	tests := []struct {
		name string
		run  func() error
	}{
		{"nil circuit", func() error {
			_, err := NewDeterministicCompiledSetup(nil, 2, tauY, tauZ)
			return err
		}},
		{"non-power-of-two M", func() error {
			_, err := NewDeterministicCompiledSetup(spr, 3, tauY, tauZ)
			return err
		}},
		{"zero tauY", func() error {
			_, err := NewDeterministicCompiledSetup(spr, 2, fr.Element{}, tauZ)
			return err
		}},
		{"zero tauZ", func() error {
			_, err := NewDeterministicCompiledSetup(spr, 2, tauY, fr.Element{})
			return err
		}},
		{"M exceeds T", func() error {
			_, err := NewDeterministicCompiledSetup(spr, 32, tauY, tauZ)
			return err
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.run(); !errors.Is(err, ErrInvalidCompiledSetup) {
				t.Fatalf("error = %v, want ErrInvalidCompiledSetup", err)
			}
		})
	}

	malformed := []struct {
		name   string
		mutate func(*CompiledSetup)
	}{
		{"reordered manifest", func(setup *CompiledSetup) {
			setup.Parties[0], setup.Parties[1] = setup.Parties[1], setup.Parties[0]
		}},
		{"wrong party SRS rank", func(setup *CompiledSetup) {
			setup.Parties[0].RowSRS.Rank = 1
		}},
		{"nil fast fixed", func(setup *CompiledSetup) {
			setup.Parties[0].FastFixed = nil
		}},
		{"mutated fast fixed", func(setup *CompiledSetup) {
			setup.Parties[0].FastFixed.selectors[0][0].Add(
				&setup.Parties[0].FastFixed.selectors[0][0], &tauY,
			)
		}},
		{"wrong constraint system", func(setup *CompiledSetup) {
			other, _ := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
			setup.Parties[0].ConstraintSystem = other
		}},
		{"wrong fixed slot", func(setup *CompiledSetup) {
			setup.Parties[0].FixedCommitments.Rank = 1
		}},
		{"wrong coordinator role", func(setup *CompiledSetup) {
			setup.Coordinator.SRS.Parties++
		}},
		{"wrong verifier role", func(setup *CompiledSetup) {
			setup.Verifier.SRS.G2Z = setup.Verifier.SRS.G2Z[:3]
		}},
		{"wrong placement role", func(setup *CompiledSetup) {
			setup.Verifier.PublicInputPlacement = nil
		}},
		{"tampered shift", func(setup *CompiledSetup) {
			setup.Metadata.IndexShift.Sigma.Add(&setup.Metadata.IndexShift.Sigma, &tauY)
		}},
		{"wrong role setup digest", func(setup *CompiledSetup) {
			setup.Parties[1].Metadata.SetupDigest[0] ^= 0xff
		}},
	}
	for _, test := range malformed {
		t.Run(test.name, func(t *testing.T) {
			setup, err := NewDeterministicCompiledSetup(spr, 2, tauY, tauZ)
			if err != nil {
				t.Fatalf("new setup: %v", err)
			}
			test.mutate(setup)
			if err := setup.Validate(); err == nil {
				t.Fatal("malformed role was accepted")
			}
		})
	}
}

func TestCompiledSetupContextBindingRejectsDifferentAuthenticatedMetadata(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	setup, err := NewDeterministicCompiledSetup(spr, 4, fr.NewElement(131), fr.NewElement(137))
	if err != nil {
		t.Fatalf("new setup: %v", err)
	}

	outer := compiledSetupOuterContextForTest(setup)
	tamperedOuter := outer
	tamperedOuter.SRSDigest[0] ^= 0xff
	if err := setup.ValidateOuterContext(tamperedOuter); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("tampered outer SRS digest error = %v", err)
	}
	tamperedOuter = outer
	tamperedOuter.IndexDigest[0] ^= 0xff
	if err := setup.ValidateOuterContext(tamperedOuter); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("tampered outer index digest error = %v", err)
	}
	tamperedOuter = outer
	tamperedOuter.PartyManifestDigest[0] ^= 0xff
	if err := setup.ValidateOuterContext(tamperedOuter); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("tampered outer manifest error = %v", err)
	}

	opening := compiledSetupOpeningContextForTest(setup)
	opening.SRSDigest[0] ^= 0xff
	if err := setup.ValidateOpeningContext(opening); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("tampered opening SRS digest error = %v", err)
	}
}

func TestCompiledSetupPointerMutationIsIndependentOrRejected(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	setup, err := NewDeterministicCompiledSetup(spr, 2, fr.NewElement(139), fr.NewElement(149))
	if err != nil {
		t.Fatalf("new setup: %v", err)
	}

	partyOnePoint := setup.Parties[1].RowSRS.G1SemanticRow[0]
	setup.Parties[0].RowSRS.G1SemanticRow[0] = setup.Parties[0].RowSRS.G1SemanticRow[1]
	if !setup.Parties[1].RowSRS.G1SemanticRow[0].Equal(&partyOnePoint) {
		t.Fatal("party SRS row slices alias across ranks")
	}
	if err := setup.Validate(); err == nil {
		t.Fatal("offline validation accepted a mutated party SRS row")
	}

	setup, err = NewDeterministicCompiledSetup(spr, 2, fr.NewElement(139), fr.NewElement(149))
	if err != nil {
		t.Fatalf("new setup for aggregate mutation: %v", err)
	}
	verifierSelector := setup.Verifier.FixedCommitments.Selectors[0]
	setup.Coordinator.FixedCommitments.Selectors[0] = setup.Coordinator.FixedCommitments.Selectors[1]
	if !setup.Verifier.FixedCommitments.Selectors[0].Equal(&verifierSelector) {
		t.Fatal("coordinator and verifier fixed-commitment values alias")
	}
	if err := setup.ValidateOnlineRoles(); err == nil {
		t.Fatal("online role validation accepted inconsistent aggregate fixed metadata")
	}

	setup, err = NewDeterministicCompiledSetup(spr, 2, fr.NewElement(139), fr.NewElement(149))
	if err != nil {
		t.Fatalf("new setup for circuit mutation: %v", err)
	}
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	spr.Constraints[0].K++
	if err := setup.Validate(); err == nil {
		t.Fatal("offline validation accepted caller mutation of the retained SparseR1CS")
	}
	if _, _, err := BuildLocalPIOPTableFromSolution(
		setup.Parties[0].Preprocessing,
		setup.Parties[0].ConstraintSystem,
		solution,
	); !errors.Is(err, ErrLocalPIOPPreprocessingMismatch) {
		t.Fatalf("mutated circuit online build error = %v", err)
	}
}

func TestCompiledSetupOnlineRejectsCrossSetupRolesAndPlacement(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	otherSPR, _ := compileLocalPIOPAdapterCircuit(t, 15, 2, fr.Element{})
	tauY := fr.NewElement(151)
	tauZ := fr.NewElement(157)

	setup, err := NewDeterministicCompiledSetup(spr, 2, tauY, tauZ)
	if err != nil {
		t.Fatalf("new setup: %v", err)
	}
	other, err := NewDeterministicCompiledSetup(otherSPR, 2, tauY, tauZ)
	if err != nil {
		t.Fatalf("new same-shape setup: %v", err)
	}
	if setup.Metadata.LocalDomainSize != other.Metadata.LocalDomainSize ||
		setup.Metadata.PartitionCount != other.Metadata.PartitionCount {
		t.Fatalf(
			"cross-setup fixture is not same-shaped: first M=%d,T=%d second M=%d,T=%d",
			setup.Metadata.PartitionCount, setup.Metadata.LocalDomainSize,
			other.Metadata.PartitionCount, other.Metadata.LocalDomainSize,
		)
	}
	for rank := range setup.Parties {
		setup.Parties[rank] = other.Parties[rank]
		// Model a transported role whose outer tag was relabeled without
		// changing its circuit-specific preprocessing.
		setup.Parties[rank].Metadata = setup.Metadata
	}
	if err := setup.ValidateOnlineRoles(); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("cross-setup party substitution error = %v", err)
	}

	setup, err = NewDeterministicCompiledSetup(spr, 2, tauY, tauZ)
	if err != nil {
		t.Fatalf("new setup for placement replacement: %v", err)
	}
	wrongPlacement := *setup.Coordinator.PublicInputPlacement
	wrongPlacement.publicVariables++
	setup.Coordinator.PublicInputPlacement = &wrongPlacement
	setup.Verifier.PublicInputPlacement = &wrongPlacement
	if err := setup.ValidateOnlineRoles(); !errors.Is(err, ErrCompiledSetupMetadataMismatch) {
		t.Fatalf("cross-placement substitution error = %v", err)
	}

	setup, err = NewDeterministicCompiledSetup(spr, 2, tauY, tauZ)
	if err != nil {
		t.Fatalf("new setup for slot-label mutation: %v", err)
	}
	setup.Parties[1].Preprocessing.index.SlotLabel.SetUint64(99)
	if err := setup.ValidateOnlineRoles(); err == nil {
		t.Fatalf("preprocessing slot-label substitution error = %v", err)
	}
}

func TestCompiledSetupOfflineRejectsSameShapeInnerRoleSwaps(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	otherSPR, _ := compileLocalPIOPAdapterCircuit(t, 15, 2, fr.Element{})
	tauY := fr.NewElement(163)
	tauZ := fr.NewElement(167)
	otherTauY := fr.NewElement(173)
	otherTauZ := fr.NewElement(179)

	otherCircuit, err := NewDeterministicCompiledSetup(otherSPR, 2, tauY, tauZ)
	if err != nil {
		t.Fatalf("other-circuit setup: %v", err)
	}
	otherSRS, err := NewDeterministicCompiledSetup(spr, 2, otherTauY, otherTauZ)
	if err != nil {
		t.Fatalf("other-SRS setup: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*CompiledSetup)
	}{
		{"fast fixed", func(setup *CompiledSetup) {
			setup.Parties[0].FastFixed = otherCircuit.Parties[0].FastFixed
		}},
		{"party SRS", func(setup *CompiledSetup) {
			setup.Parties[0].RowSRS = otherSRS.Parties[0].RowSRS
		}},
		{"coordinator SRS", func(setup *CompiledSetup) {
			setup.Coordinator.SRS = otherSRS.Coordinator.SRS
		}},
		{"verifier SRS", func(setup *CompiledSetup) {
			setup.Verifier.SRS = otherSRS.Verifier.SRS
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			setup, err := NewDeterministicCompiledSetup(spr, 2, tauY, tauZ)
			if err != nil {
				t.Fatalf("new setup: %v", err)
			}
			test.mutate(setup)
			if err := setup.Validate(); err == nil {
				t.Fatal("offline validation accepted a same-shape inner-role swap")
			}
		})
	}
}

func compiledSetupOuterContextForTest(setup *CompiledSetup) OuterTranscriptContext {
	metadata := setup.Metadata
	return OuterTranscriptContext{
		ProtocolVersion:       "dlinkzg-compiled-setup-test/v1",
		PublicStatementDigest: NewTranscriptDigest([]byte("compiled setup statement")),
		SRSDigest:             metadata.SRSDigest,
		IndexDigest:           metadata.IndexDigest,
		IndexPrefixDigest:     metadata.IndexPrefixDigest,
		ShiftCounter:          metadata.IndexShift.Counter,
		Shift:                 metadata.IndexShift.Sigma,
		PartyManifestDigest:   metadata.PartyManifestDigest,
		SessionNonce:          [32]byte(NewTranscriptDigest([]byte("compiled setup session"))),
		PartitionCount:        uint64(metadata.PartitionCount),
		LocalDomainSize:       uint64(metadata.LocalDomainSize),
		LocalDomainGenerator:  metadata.LocalDomainGenerator,
	}
}

func compiledSetupOpeningContextForTest(setup *CompiledSetup) TranscriptContext {
	metadata := setup.Metadata
	return TranscriptContext{
		ProtocolVersion:       "dlinkzg-compiled-setup-test/v1",
		PublicStatementDigest: NewTranscriptDigest([]byte("compiled setup statement")),
		SRSDigest:             metadata.SRSDigest,
		IndexDigest:           metadata.IndexDigest,
		PartyManifestDigest:   metadata.PartyManifestDigest,
		SessionNonce:          [32]byte(NewTranscriptDigest([]byte("compiled setup session"))),
		PriorTranscriptDigest: NewTranscriptDigest([]byte("compiled setup prior transcript")),
	}
}

func compiledFixedSlotEqual(left, right LocalPIOPFixedCommitmentSlot) bool {
	if left.Rank != right.Rank || left.Parties != right.Parties ||
		left.DomainSize != right.DomainSize || !left.SlotLabel.Equal(&right.SlotLabel) {
		return false
	}
	for selector := range left.Selectors {
		if !left.Selectors[selector].Equal(&right.Selectors[selector]) {
			return false
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !left.SigmaX[wire].Equal(&right.SigmaX[wire]) ||
			!left.SigmaPart[wire].Equal(&right.SigmaPart[wire]) {
			return false
		}
	}
	return left.One.Equal(&right.One) && left.X.Equal(&right.X)
}

func compiledAssertNoTrapdoorScalarField(t *testing.T, typ reflect.Type, seen map[reflect.Type]bool) {
	t.Helper()
	for typ.Kind() == reflect.Pointer || typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array {
		typ = typ.Elem()
	}
	if seen[typ] || typ.Kind() != reflect.Struct {
		return
	}
	seen[typ] = true
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		lower := strings.ToLower(field.Name)
		if strings.Contains(lower, "trapdoor") || strings.Contains(lower, "tauy") || strings.Contains(lower, "tauz") {
			t.Fatalf("returned setup type retains a trapdoor-looking field %s.%s", typ, field.Name)
		}
		fieldType := field.Type
		for fieldType.Kind() == reflect.Pointer || fieldType.Kind() == reflect.Slice || fieldType.Kind() == reflect.Array {
			fieldType = fieldType.Elem()
		}
		// ConstraintSystem is a witness-independent gnark input, and elliptic
		// curve points necessarily encode trapdoor-derived public SRS values.
		// Only recurse through the bundle's own role types.
		if fieldType.PkgPath() == reflect.TypeOf(CompiledSetup{}).PkgPath() {
			compiledAssertNoTrapdoorScalarField(t, fieldType, seen)
		}
	}
}
