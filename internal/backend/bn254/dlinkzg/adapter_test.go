package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
	bn254witness "github.com/consensys/gnark/internal/backend/bn254/witness"
)

type localPIOPAdapterCircuit struct {
	X      frontend.Variable
	Y      frontend.Variable `gnark:",public"`
	rounds int
}

func (c *localPIOPAdapterCircuit) Define(api frontend.API) error {
	value := c.X
	for round := 0; round < c.rounds; round++ {
		value = api.Mul(value, c.X)
	}
	api.AssertIsEqual(value, c.Y)
	return nil
}

type localPIOPManyPublicCircuit struct {
	X      frontend.Variable
	Public [4]frontend.Variable `gnark:",public"`
}

func (c *localPIOPManyPublicCircuit) Define(api frontend.API) error {
	for i := range c.Public {
		api.AssertIsEqual(c.X, c.Public[i])
	}
	return nil
}

type extractedLocalPIOPRank struct {
	table LocalPIOPTable
	index LocalPIOPIndex
}

func TestSparseR1CSAdapterTwoAndFourRanks(t *testing.T) {
	const rounds = 16
	spr, witness := compileLocalPIOPAdapterCircuit(t, rounds, 2, fr.Element{})

	for _, world := range []int{2, 4} {
		t.Run(adapterWorldName(world), func(t *testing.T) {
			extracted := extractAllLocalPIOPRanks(t, spr, witness, world)
			localRows := len(extracted[0].table.Wires[LocalWireA])
			if localRows < 2 || !isPowerOfTwo(localRows) {
				t.Fatalf("local rows = %d, want a power of two at least two", localRows)
			}

			// Partition labels are the paper's canonical rank embeddings;
			// only the local-row coordinate uses an FFT root and wire cosets.
			localDomain := fft.NewDomain(uint64(localRows))
			for rank := range extracted {
				wantSlot := fr.NewElement(uint64(rank))
				assertLocalFieldEqual(t, extracted[rank].index.SlotLabel, wantSlot, "partition slot label")
				assertLocalFieldEqual(t, extracted[rank].index.WireCosets[LocalWireA], fr.One(), "wire-A coset")
				assertLocalFieldEqual(t, extracted[rank].index.WireCosets[LocalWireB], localDomain.FrMultiplicativeGen, "wire-B coset")
				var cosetSquared fr.Element
				cosetSquared.Square(&localDomain.FrMultiplicativeGen)
				assertLocalFieldEqual(t, extracted[rank].index.WireCosets[LocalWireC], cosetSquared, "wire-C coset")
			}

			assertLocalPIOPAdapterRows(t, spr, witness, extracted)
			crossPartitionEdges := assertLocalPIOPAdapterPermutation(t, spr, extracted, world)
			if crossPartitionEdges == 0 {
				t.Fatal("test circuit did not create a cross-partition copy edge")
			}

			challenges, relations := buildAllExtractedLocalRelations(t, extracted)
			var product fr.Element
			product.SetOne()
			for rank, relation := range relations {
				if err := VerifyLocalPIOPRelation(relation); err != nil {
					t.Fatalf("rank %d local relation: %v", rank, err)
				}
				product.Mul(&product, &relation.BlockFactor)
			}
			if !product.IsOne() {
				t.Fatalf("global product of rho values = %s, want one", product.String())
			}
			assertLocalPIOPAdapterTagMultisets(t, extracted, localDomain.Generator, challenges)
		})
	}
}

func TestSparseR1CSAdapterFromSolutionMatchesSolveWrapper(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}

	for _, world := range []int{2, 4} {
		for rank := 0; rank < world; rank++ {
			viaWrapper, wrapperIndex, err := ExtractLocalPIOPTable(
				spr,
				witness,
				LocalPIOPAdapterConfig{Rank: rank, World: world, ProverConfig: proverConfig},
			)
			if err != nil {
				t.Fatalf("wrapper rank %d/%d: %v", rank, world, err)
			}
			viaSolution, solutionIndex, err := ExtractLocalPIOPTableFromSolution(spr, solution, rank, world)
			if err != nil {
				t.Fatalf("solution rank %d/%d: %v", rank, world, err)
			}
			assertLocalPIOPAdapterTableEqual(t, viaSolution, viaWrapper)
			assertLocalFieldEqual(t, solutionIndex.SlotLabel, wrapperIndex.SlotLabel, "solution slot label")
			for wire := 0; wire < LocalWireCount; wire++ {
				assertLocalFieldEqual(t, solutionIndex.WireCosets[wire], wrapperIndex.WireCosets[wire], "solution wire coset")
			}
		}
	}
}

func TestSparseR1CSAdapterFromSolutionRejectsMalformedInputs(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}
	longSolution := append(append([]fr.Element(nil), solution...), fr.One())

	tests := []struct {
		name      string
		system    *cs.SparseR1CS
		solution  []fr.Element
		rank      int
		world     int
		wantError error
	}{
		{
			name:      "nil system",
			solution:  solution,
			world:     2,
			wantError: ErrMalformedSparseR1CS,
		},
		{
			name:      "short solution",
			system:    spr,
			solution:  solution[:len(solution)-1],
			world:     2,
			wantError: ErrMalformedSparseR1CS,
		},
		{
			name:      "long solution",
			system:    spr,
			solution:  longSolution,
			world:     2,
			wantError: ErrMalformedSparseR1CS,
		},
		{
			name:      "world is not power of two",
			system:    spr,
			solution:  solution,
			world:     3,
			wantError: ErrInvalidLocalPIOPAdapterConfig,
		},
		{
			name:      "rank outside world",
			system:    spr,
			solution:  solution,
			rank:      2,
			world:     2,
			wantError: ErrInvalidLocalPIOPAdapterConfig,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := ExtractLocalPIOPTableFromSolution(test.system, test.solution, test.rank, test.world)
			if !errors.Is(err, test.wantError) {
				t.Fatalf("error = %v, want %v", err, test.wantError)
			}
		})
	}

	t.Run("malformed constraint system", func(t *testing.T) {
		malformed := *spr
		malformed.Coefficients = nil
		_, _, err := ExtractLocalPIOPTableFromSolution(&malformed, solution, 0, 2)
		if !errors.Is(err, ErrMalformedSparseR1CS) {
			t.Fatalf("error = %v, want ErrMalformedSparseR1CS", err)
		}
	})
}

func TestSparseR1CSAdapterPreprocessingIsolationAndBinding(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}
	preprocessing, err := PreprocessLocalPIOP(spr, 0, 2)
	if err != nil {
		t.Fatalf("preprocess without a solution: %v", err)
	}
	if preprocessing.Rank() != 0 || preprocessing.World() != 2 ||
		preprocessing.LocalRows() < 2 || preprocessing.TotalRows() != 2*preprocessing.LocalRows() ||
		preprocessing.LogicalRows() != spr.NbPublicVariables+len(spr.Constraints) {
		t.Fatalf(
			"preprocessing layout = rank %d world %d logical %d local %d total %d",
			preprocessing.Rank(), preprocessing.World(), preprocessing.LogicalRows(),
			preprocessing.LocalRows(), preprocessing.TotalRows(),
		)
	}
	wantIndex := preprocessing.Index()
	assertLocalFieldEqual(t, wantIndex.SlotLabel, fr.NewElement(0), "preprocessed rank label")

	fixedSnapshot := preprocessing.FixedTable()
	if fixedSnapshot.Wires[0] != nil || fixedSnapshot.PublicInput != nil {
		t.Fatal("witness-dependent columns appear in fixed preprocessing")
	}
	returnedFixed := preprocessing.FixedTable()
	returnedFixed.Selectors[LocalSelectorL][0].Add(&returnedFixed.Selectors[LocalSelectorL][0], valuePointer(fr.One()))
	returnedFixed.SigmaX[LocalWireA][0].Add(&returnedFixed.SigmaX[LocalWireA][0], valuePointer(fr.One()))
	assertLocalPIOPAdapterTableEqual(t, preprocessing.FixedTable(), fixedSnapshot)

	first, firstIndex, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, solution)
	if err != nil {
		t.Fatalf("first online build: %v", err)
	}
	first.Selectors[LocalSelectorL][0].Add(&first.Selectors[LocalSelectorL][0], valuePointer(fr.One()))
	first.SigmaPart[LocalWireA][0].Add(&first.SigmaPart[LocalWireA][0], valuePointer(fr.One()))
	second, secondIndex, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, solution)
	if err != nil {
		t.Fatalf("second online build: %v", err)
	}
	want, _, err := ExtractLocalPIOPTableFromSolution(spr, solution, 0, 2)
	if err != nil {
		t.Fatalf("fresh extraction: %v", err)
	}
	assertLocalPIOPAdapterTableEqual(t, second, want)
	assertLocalFieldEqual(t, firstIndex.SlotLabel, secondIndex.SlotLabel, "reused preprocessing index")

	originalCoefficient := spr.Coefficients[0]
	spr.Coefficients[0].Add(&spr.Coefficients[0], valuePointer(fr.One()))
	if _, _, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, solution); !errors.Is(err, ErrLocalPIOPPreprocessingMismatch) {
		t.Fatalf("mutated-system error = %v, want ErrLocalPIOPPreprocessingMismatch", err)
	}
	assertLocalPIOPAdapterTableEqual(t, preprocessing.FixedTable(), fixedSnapshot)
	spr.Coefficients[0] = originalCoefficient

	otherSystem, otherWitness := compileLocalPIOPAdapterCircuit(t, 12, 2, fr.Element{})
	otherSolution, err := otherSystem.Solve(otherWitness, proverConfig)
	if err != nil {
		t.Fatalf("solve other fixture: %v", err)
	}
	if _, _, err := BuildLocalPIOPTableFromSolution(preprocessing, otherSystem, otherSolution); !errors.Is(err, ErrLocalPIOPPreprocessingMismatch) {
		t.Fatalf("mismatched-system error = %v, want ErrLocalPIOPPreprocessingMismatch", err)
	}
	if _, _, err := BuildLocalPIOPTableFromSolution(nil, spr, solution); !errors.Is(err, ErrLocalPIOPPreprocessingMismatch) {
		t.Fatalf("nil-preprocessing error = %v, want ErrLocalPIOPPreprocessingMismatch", err)
	}
	if _, err := PreprocessLocalPIOP(spr, 2, 2); !errors.Is(err, ErrInvalidLocalPIOPAdapterConfig) {
		t.Fatalf("mismatched-rank error = %v, want ErrInvalidLocalPIOPAdapterConfig", err)
	}
}

func TestSparseR1CSAdapterBatchPreprocessingMatchesIndividual(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}

	for _, world := range []int{2, 4} {
		t.Run(adapterWorldName(world), func(t *testing.T) {
			batch, err := PreprocessAllLocalPIOP(spr, world)
			if err != nil {
				t.Fatalf("batch preprocess: %v", err)
			}
			if len(batch) != world {
				t.Fatalf("batch length = %d, want %d", len(batch), world)
			}
			for rank := 0; rank < world; rank++ {
				individual, err := PreprocessLocalPIOP(spr, rank, world)
				if err != nil {
					t.Fatalf("individual rank %d: %v", rank, err)
				}
				assertLocalPIOPAdapterPreprocessingEqual(t, batch[rank], individual)

				batchTable, batchIndex, err := BuildLocalPIOPTableFromSolution(batch[rank], spr, solution)
				if err != nil {
					t.Fatalf("batch online table rank %d: %v", rank, err)
				}
				individualTable, individualIndex, err := BuildLocalPIOPTableFromSolution(individual, spr, solution)
				if err != nil {
					t.Fatalf("individual online table rank %d: %v", rank, err)
				}
				assertLocalPIOPAdapterTableEqual(t, batchTable, individualTable)
				assertLocalFieldEqual(t, batchIndex.SlotLabel, individualIndex.SlotLabel, "batch slot label")
				for wire := 0; wire < LocalWireCount; wire++ {
					assertLocalFieldEqual(
						t, batchIndex.WireCosets[wire], individualIndex.WireCosets[wire], "batch wire coset",
					)
				}
			}
		})
	}
}

func TestSparseR1CSAdapterBatchPreprocessingDoesNotAliasRanks(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 16, 2, fr.Element{})
	batch, err := PreprocessAllLocalPIOP(spr, 4)
	if err != nil {
		t.Fatalf("batch preprocess: %v", err)
	}
	if batch[0] == batch[1] {
		t.Fatal("rank preprocessings alias")
	}
	rankOneSnapshot := batch[1].FixedTable()
	batch[0].fixed.Selectors[LocalSelectorL][0].Add(
		&batch[0].fixed.Selectors[LocalSelectorL][0], valuePointer(fr.One()),
	)
	batch[0].fixed.SigmaX[LocalWireA][0].Add(
		&batch[0].fixed.SigmaX[LocalWireA][0], valuePointer(fr.One()),
	)
	batch[0].fixed.SigmaPart[LocalWireA][0].Add(
		&batch[0].fixed.SigmaPart[LocalWireA][0], valuePointer(fr.One()),
	)
	assertLocalPIOPAdapterFixedColumnsEqual(t, batch[1].FixedTable(), rankOneSnapshot)
}

func TestSparseR1CSAdapterBatchPreprocessingRejectsMalformedInputs(t *testing.T) {
	spr, _ := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
	tests := []struct {
		name      string
		system    *cs.SparseR1CS
		world     int
		wantError error
	}{
		{name: "nil system", system: nil, world: 2, wantError: ErrMalformedSparseR1CS},
		{name: "zero world", system: spr, world: 0, wantError: ErrInvalidLocalPIOPAdapterConfig},
		{name: "non-power-of-two world", system: spr, world: 3, wantError: ErrInvalidLocalPIOPAdapterConfig},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := PreprocessAllLocalPIOP(test.system, test.world)
			if !errors.Is(err, test.wantError) {
				t.Fatalf("error = %v, want %v", err, test.wantError)
			}
			if result != nil {
				t.Fatalf("malformed batch returned %d roles", len(result))
			}
		})
	}

	malformed := *spr
	malformed.Coefficients = nil
	if result, err := PreprocessAllLocalPIOP(&malformed, 2); !errors.Is(err, ErrMalformedSparseR1CS) || result != nil {
		t.Fatalf("malformed-system result length/error = %d/%v", len(result), err)
	}
}

func TestSparseR1CSAdapterFixedPreprocessingIsWitnessIndependent(t *testing.T) {
	const rounds = 8
	system, err := frontend.Compile(ecc.BN254, scs.NewBuilder, &localPIOPAdapterCircuit{rounds: rounds})
	if err != nil {
		t.Fatalf("compile fixture: %v", err)
	}
	spr := system.(*cs.SparseR1CS)
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}

	solutions := make([][]fr.Element, 2)
	for fixture, xValue := range []uint64{2, 3} {
		x := fr.NewElement(xValue)
		y := x
		for round := 0; round < rounds; round++ {
			y.Mul(&y, &x)
		}
		witness := localPIOPConcreteWitness(t, &localPIOPAdapterCircuit{X: x, Y: y, rounds: rounds})
		solutions[fixture], err = spr.Solve(witness, proverConfig)
		if err != nil {
			t.Fatalf("solve fixture %d: %v", fixture, err)
		}
	}

	preprocessing, err := PreprocessLocalPIOP(spr, 0, 2)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	first, _, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, solutions[0])
	if err != nil {
		t.Fatalf("build first witness: %v", err)
	}
	second, _, err := BuildLocalPIOPTableFromSolution(preprocessing, spr, solutions[1])
	if err != nil {
		t.Fatalf("build second witness: %v", err)
	}
	assertLocalPIOPAdapterFixedColumnsEqual(t, first, second)
	if first.PublicInput[0].Equal(&second.PublicInput[0]) {
		t.Fatal("two distinct satisfying witnesses produced the same public-input column")
	}
}

func TestSparseR1CSAdapterPublicPlaceholderSemantics(t *testing.T) {
	const rounds = 8
	spr, witness := compileLocalPIOPAdapterCircuit(t, rounds, 3, fr.Element{})
	extracted := extractAllLocalPIOPRanks(t, spr, witness, 2)
	rankZero := extracted[0].table

	if spr.NbPublicVariables != 1 {
		t.Fatalf("fixture public variables = %d, want one", spr.NbPublicVariables)
	}
	assertLocalFieldEqual(t, rankZero.Wires[LocalWireA][0], witness[0], "placeholder A")
	assertLocalFieldEqual(t, rankZero.Wires[LocalWireB][0], witness[0], "placeholder B")
	assertLocalFieldEqual(t, rankZero.Wires[LocalWireC][0], witness[0], "placeholder C")
	minusOne := fr.One()
	minusOne.Neg(&minusOne)
	assertLocalFieldEqual(t, rankZero.Selectors[LocalSelectorL][0], minusOne, "placeholder q_L")
	assertLocalFieldEqual(t, rankZero.PublicInput[0], witness[0], "placeholder PI")
	if !rankZero.Selectors[LocalSelectorM][0].IsZero() ||
		!rankZero.Selectors[LocalSelectorR][0].IsZero() ||
		!rankZero.Selectors[LocalSelectorO][0].IsZero() ||
		!rankZero.Selectors[LocalSelectorC][0].IsZero() {
		t.Fatal("placeholder has a nonzero selector other than q_L")
	}

	for rank := range extracted {
		for row := range extracted[rank].table.PublicInput {
			if rank == 0 && row < spr.NbPublicVariables {
				continue
			}
			if !extracted[rank].table.PublicInput[row].IsZero() {
				t.Fatalf("rank %d row %d unexpectedly carries public input", rank, row)
			}
		}
	}
}

func TestSparseR1CSAdapterRejectsMalformedInputs(t *testing.T) {
	spr, witness := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.Element{})
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}

	tests := []struct {
		name      string
		system    *cs.SparseR1CS
		witness   bn254witness.Witness
		config    LocalPIOPAdapterConfig
		wantError error
	}{
		{
			name:      "nil system",
			system:    nil,
			witness:   witness,
			config:    LocalPIOPAdapterConfig{Rank: 0, World: 2, ProverConfig: proverConfig},
			wantError: ErrMalformedSparseR1CS,
		},
		{
			name:      "world is not power of two",
			system:    spr,
			witness:   witness,
			config:    LocalPIOPAdapterConfig{Rank: 0, World: 3, ProverConfig: proverConfig},
			wantError: ErrInvalidLocalPIOPAdapterConfig,
		},
		{
			name:      "rank outside world",
			system:    spr,
			witness:   witness,
			config:    LocalPIOPAdapterConfig{Rank: 2, World: 2, ProverConfig: proverConfig},
			wantError: ErrInvalidLocalPIOPAdapterConfig,
		},
		{
			name:      "wrong witness length",
			system:    spr,
			witness:   witness[:len(witness)-1],
			config:    LocalPIOPAdapterConfig{Rank: 0, World: 2, ProverConfig: proverConfig},
			wantError: ErrMalformedSparseR1CS,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := ExtractLocalPIOPTable(test.system, test.witness, test.config)
			if !errors.Is(err, test.wantError) {
				t.Fatalf("error = %v, want %v", err, test.wantError)
			}
		})
	}

	t.Run("empty coefficient table", func(t *testing.T) {
		malformed := *spr
		malformed.Coefficients = nil
		_, _, err := ExtractLocalPIOPTable(
			&malformed,
			witness,
			LocalPIOPAdapterConfig{Rank: 0, World: 2, ProverConfig: proverConfig},
		)
		if !errors.Is(err, ErrMalformedSparseR1CS) {
			t.Fatalf("error = %v, want ErrMalformedSparseR1CS", err)
		}
	})

	t.Run("unsatisfied witness", func(t *testing.T) {
		_, badWitness := compileLocalPIOPAdapterCircuit(t, 8, 2, fr.NewElement(99))
		_, _, err := ExtractLocalPIOPTable(
			spr,
			badWitness,
			LocalPIOPAdapterConfig{Rank: 0, World: 2, ProverConfig: proverConfig},
		)
		if !errors.Is(err, ErrLocalPIOPAdapterSolve) {
			t.Fatalf("error = %v, want ErrLocalPIOPAdapterSolve", err)
		}
	})
}

func TestSparseR1CSAdapterRejectsPublicInputsThatDoNotFit(t *testing.T) {
	circuit := &localPIOPManyPublicCircuit{}
	system, err := frontend.Compile(ecc.BN254, scs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("compile many-public circuit: %v", err)
	}
	spr := system.(*cs.SparseR1CS)
	assignment := &localPIOPManyPublicCircuit{X: 7}
	for i := range assignment.Public {
		assignment.Public[i] = 7
	}
	witness := localPIOPConcreteWitness(t, assignment)
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	_, _, err = ExtractLocalPIOPTable(
		spr,
		witness,
		LocalPIOPAdapterConfig{Rank: 0, World: 4, ProverConfig: proverConfig},
	)
	if !errors.Is(err, ErrLocalPublicInputsDoNotFit) {
		t.Fatalf("error = %v, want ErrLocalPublicInputsDoNotFit", err)
	}
}

func compileLocalPIOPAdapterCircuit(t *testing.T, rounds int, x uint64, forcedY fr.Element) (*cs.SparseR1CS, bn254witness.Witness) {
	t.Helper()
	circuit := &localPIOPAdapterCircuit{rounds: rounds}
	system, err := frontend.Compile(ecc.BN254, scs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("compile adapter circuit: %v", err)
	}

	xElement := fr.NewElement(x)
	yElement := xElement
	for round := 0; round < rounds; round++ {
		yElement.Mul(&yElement, &xElement)
	}
	if !forcedY.IsZero() {
		yElement = forcedY
	}
	assignment := &localPIOPAdapterCircuit{X: xElement, Y: yElement, rounds: rounds}
	return system.(*cs.SparseR1CS), localPIOPConcreteWitness(t, assignment)
}

func localPIOPConcreteWitness(t *testing.T, assignment frontend.Circuit) bn254witness.Witness {
	t.Helper()
	witness, err := frontend.NewWitness(assignment, ecc.BN254)
	if err != nil {
		t.Fatalf("build adapter witness: %v", err)
	}
	concrete, ok := witness.Vector.(*bn254witness.Witness)
	if !ok {
		t.Fatalf("unexpected witness type %T", witness.Vector)
	}
	return append(bn254witness.Witness(nil), (*concrete)...)
}

func extractAllLocalPIOPRanks(t *testing.T, spr *cs.SparseR1CS, witness bn254witness.Witness, world int) []extractedLocalPIOPRank {
	t.Helper()
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	result := make([]extractedLocalPIOPRank, world)
	for rank := 0; rank < world; rank++ {
		result[rank].table, result[rank].index, err = ExtractLocalPIOPTable(
			spr,
			witness,
			LocalPIOPAdapterConfig{Rank: rank, World: world, ProverConfig: proverConfig},
		)
		if err != nil {
			t.Fatalf("extract rank %d/%d: %v", rank, world, err)
		}
	}
	return result
}

func assertLocalPIOPAdapterRows(t *testing.T, spr *cs.SparseR1CS, witness bn254witness.Witness, extracted []extractedLocalPIOPRank) {
	t.Helper()
	localRows := len(extracted[0].table.Wires[0])
	logicalRows := spr.NbPublicVariables + len(spr.Constraints)
	for rank := range extracted {
		table := &extracted[rank].table
		for row := 0; row < localRows; row++ {
			var gate, term fr.Element
			term.Mul(&table.Wires[LocalWireA][row], &table.Wires[LocalWireB][row]).
				Mul(&term, &table.Selectors[LocalSelectorM][row])
			gate.Add(&gate, &term)
			term.Mul(&table.Wires[LocalWireA][row], &table.Selectors[LocalSelectorL][row])
			gate.Add(&gate, &term)
			term.Mul(&table.Wires[LocalWireB][row], &table.Selectors[LocalSelectorR][row])
			gate.Add(&gate, &term)
			term.Mul(&table.Wires[LocalWireC][row], &table.Selectors[LocalSelectorO][row])
			gate.Add(&gate, &term)
			gate.Add(&gate, &table.Selectors[LocalSelectorC][row])
			gate.Add(&gate, &table.PublicInput[row])
			if !gate.IsZero() {
				t.Fatalf("rank %d row %d gate value = %s", rank, row, gate.String())
			}

			globalRow := rank*localRows + row
			if globalRow >= logicalRows {
				for wire := 0; wire < LocalWireCount; wire++ {
					assertLocalFieldEqual(t, table.Wires[wire][row], witness[0], "padding wire")
				}
				for selector := 0; selector < LocalSelectorCount; selector++ {
					if !table.Selectors[selector][row].IsZero() {
						t.Fatalf("rank %d padding row %d selector %d is nonzero", rank, row, selector)
					}
				}
				if !table.PublicInput[row].IsZero() {
					t.Fatalf("rank %d padding row %d PI is nonzero", rank, row)
				}
			}
		}
	}
}

func assertLocalPIOPAdapterPermutation(t *testing.T, spr *cs.SparseR1CS, extracted []extractedLocalPIOPRank, world int) int {
	t.Helper()
	localRows := len(extracted[0].table.Wires[0])
	totalRows := world * localRows
	lro := make([]int, 3*totalRows)
	for public := 0; public < spr.NbPublicVariables; public++ {
		lro[public] = public
	}
	for constraintIndex := range spr.Constraints {
		row := spr.NbPublicVariables + constraintIndex
		lro[row] = spr.Constraints[constraintIndex].L.WireID()
		lro[totalRows+row] = spr.Constraints[constraintIndex].R.WireID()
		lro[2*totalRows+row] = spr.Constraints[constraintIndex].O.WireID()
	}
	last := make([]int, spr.NbPublicVariables+spr.NbSecretVariables+spr.NbInternalVariables)
	for variable := range last {
		last[variable] = -1
	}
	permutation := make([]int, len(lro))
	for position := range permutation {
		permutation[position] = -1
		variable := lro[position]
		if last[variable] != -1 {
			permutation[position] = last[variable]
		}
		last[variable] = position
	}
	for position := range permutation {
		if permutation[position] == -1 {
			permutation[position] = last[lro[position]]
		}
	}

	localDomain := fft.NewDomain(uint64(localRows))
	localPoints := make([]fr.Element, localRows)
	localPoints[0].SetOne()
	for row := 1; row < localRows; row++ {
		localPoints[row].Mul(&localPoints[row-1], &localDomain.Generator)
	}
	crossPartitionEdges := 0
	for rank := range extracted {
		for wire := 0; wire < LocalWireCount; wire++ {
			for row := 0; row < localRows; row++ {
				globalRow := rank*localRows + row
				position := wire*totalRows + globalRow
				destination := permutation[position]
				if lro[position] != lro[destination] {
					t.Fatalf("reference permutation changes variable at position %d", position)
				}
				destinationWire := destination / totalRows
				withinWire := destination % totalRows
				destinationRank := withinWire / localRows
				destinationRow := withinWire % localRows
				if destinationRank != rank {
					crossPartitionEdges++
				}
				wantPart := fr.NewElement(uint64(destinationRank))
				var wantX fr.Element
				wantX.Mul(&extracted[rank].index.WireCosets[destinationWire], &localPoints[destinationRow])
				assertLocalFieldEqual(t, extracted[rank].table.SigmaPart[wire][row], wantPart, "destination partition coordinate")
				assertLocalFieldEqual(t, extracted[rank].table.SigmaX[wire][row], wantX, "destination row coordinate")
			}
		}
	}
	return crossPartitionEdges
}

func buildAllExtractedLocalRelations(t *testing.T, extracted []extractedLocalPIOPRank) (LocalPIOPChallenges, []*LocalPIOPRelation) {
	t.Helper()
	challenges := LocalPIOPChallenges{
		EtaPart: fr.NewElement(2),
		EtaX:    fr.NewElement(3),
		Lambda:  fr.NewElement(7),
	}
	for gamma := uint64(5); gamma < 128; gamma++ {
		challenges.Gamma.SetUint64(gamma)
		relations := make([]*LocalPIOPRelation, len(extracted))
		retry := false
		for rank := range extracted {
			var err error
			relations[rank], err = BuildLocalPIOPRelation(extracted[rank].table, extracted[rank].index, challenges)
			if errors.Is(err, ErrZeroLocalTagDenominator) {
				retry = true
				break
			}
			if err != nil {
				t.Fatalf("build rank %d local relation: %v", rank, err)
			}
		}
		if !retry {
			return challenges, relations
		}
	}
	t.Fatal("could not find a nonzero identity-tag challenge")
	return LocalPIOPChallenges{}, nil
}

func assertLocalPIOPAdapterTagMultisets(
	t *testing.T,
	extracted []extractedLocalPIOPRank,
	localOmega fr.Element,
	challenges LocalPIOPChallenges,
) {
	t.Helper()
	identityCoordinates := make(map[string]int)
	destinationCoordinates := make(map[string]int)
	identityTags := make(map[string]int)
	destinationTags := make(map[string]int)
	for rank := range extracted {
		x := fr.One()
		for row := range extracted[rank].table.Wires[0] {
			for wire := 0; wire < LocalWireCount; wire++ {
				var identityX fr.Element
				identityX.Mul(&extracted[rank].index.WireCosets[wire], &x)
				identityCoordinates[localPIOPCoordinateKey(extracted[rank].index.SlotLabel, identityX)]++
				destinationCoordinates[localPIOPCoordinateKey(
					extracted[rank].table.SigmaPart[wire][row],
					extracted[rank].table.SigmaX[wire][row],
				)]++

				wireValue := extracted[rank].table.Wires[wire][row]
				identityTag := localPIOPAdapterTagValue(
					wireValue,
					extracted[rank].index.SlotLabel,
					identityX,
					challenges,
				)
				destinationTag := localPIOPAdapterTagValue(
					wireValue,
					extracted[rank].table.SigmaPart[wire][row],
					extracted[rank].table.SigmaX[wire][row],
					challenges,
				)
				identityTags[string(identityTag.Marshal())]++
				destinationTags[string(destinationTag.Marshal())]++
			}
			x.Mul(&x, &localOmega)
		}
	}
	assertLocalPIOPStringMultisetEqual(t, destinationCoordinates, identityCoordinates, "coordinate")
	assertLocalPIOPStringMultisetEqual(t, destinationTags, identityTags, "tag-value")
}

func assertLocalPIOPAdapterTableEqual(t *testing.T, got, want LocalPIOPTable) {
	t.Helper()
	for wire := 0; wire < LocalWireCount; wire++ {
		for row := range want.Wires[wire] {
			assertLocalFieldEqual(t, got.Wires[wire][row], want.Wires[wire][row], "solution wire")
			assertLocalFieldEqual(t, got.SigmaX[wire][row], want.SigmaX[wire][row], "solution sigma-X")
			assertLocalFieldEqual(t, got.SigmaPart[wire][row], want.SigmaPart[wire][row], "solution sigma-part")
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		for row := range want.Selectors[selector] {
			assertLocalFieldEqual(t, got.Selectors[selector][row], want.Selectors[selector][row], "solution selector")
		}
	}
	for row := range want.PublicInput {
		assertLocalFieldEqual(t, got.PublicInput[row], want.PublicInput[row], "solution public input")
	}
}

func assertLocalPIOPAdapterFixedColumnsEqual(t *testing.T, got, want LocalPIOPTable) {
	t.Helper()
	for wire := 0; wire < LocalWireCount; wire++ {
		for row := range want.SigmaX[wire] {
			assertLocalFieldEqual(t, got.SigmaX[wire][row], want.SigmaX[wire][row], "fixed sigma-X")
			assertLocalFieldEqual(t, got.SigmaPart[wire][row], want.SigmaPart[wire][row], "fixed sigma-part")
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		for row := range want.Selectors[selector] {
			assertLocalFieldEqual(t, got.Selectors[selector][row], want.Selectors[selector][row], "fixed selector")
		}
	}
}

func assertLocalPIOPAdapterPreprocessingEqual(
	t *testing.T,
	got, want *LocalPIOPPreprocessing,
) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("nil preprocessing: got=%v want=%v", got == nil, want == nil)
	}
	if got.rank != want.rank || got.world != want.world || got.layout != want.layout ||
		got.systemDigest != want.systemDigest {
		t.Fatalf(
			"preprocessing metadata differs: got rank/world %d/%d, want %d/%d",
			got.rank, got.world, want.rank, want.world,
		)
	}
	assertLocalFieldEqual(t, got.index.SlotLabel, want.index.SlotLabel, "preprocessing slot label")
	for wire := 0; wire < LocalWireCount; wire++ {
		assertLocalFieldEqual(
			t, got.index.WireCosets[wire], want.index.WireCosets[wire], "preprocessing wire coset",
		)
	}
	assertLocalPIOPAdapterFixedColumnsEqual(t, got.fixed, want.fixed)
}

func localPIOPAdapterTagValue(wire, partition, row fr.Element, challenges LocalPIOPChallenges) fr.Element {
	result := wire
	var term fr.Element
	term.Mul(&challenges.EtaPart, &partition)
	result.Add(&result, &term)
	term.Mul(&challenges.EtaX, &row)
	result.Add(&result, &term)
	result.Add(&result, &challenges.Gamma)
	return result
}

func localPIOPCoordinateKey(partition, row fr.Element) string {
	return string(partition.Marshal()) + string(row.Marshal())
}

func assertLocalPIOPStringMultisetEqual(t *testing.T, got, want map[string]int, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s multiset has %d keys, want %d", label, len(got), len(want))
	}
	for key, wantCount := range want {
		if got[key] != wantCount {
			t.Fatalf("%s multiset count = %d, want %d", label, got[key], wantCount)
		}
	}
}

func adapterWorldName(world int) string {
	if world == 2 {
		return "world-2"
	}
	return "world-4"
}
