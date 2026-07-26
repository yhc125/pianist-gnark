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
)

type statementPublicInputCircuit struct {
	X      frontend.Variable
	Public [3]frontend.Variable `gnark:",public"`
	rounds int
}

func (c *statementPublicInputCircuit) Define(api frontend.API) error {
	value := c.X
	for round := 0; round < c.rounds; round++ {
		value = api.Mul(value, c.X)
	}
	api.AssertIsEqual(value, c.Public[0])
	api.AssertIsEqual(api.Add(value, 7), c.Public[1])
	api.AssertIsEqual(api.Mul(value, c.Public[1]), c.Public[2])
	return nil
}

func TestStatementPublicInputMatchesAdapterAndMultilinearFold(t *testing.T) {
	const rounds = 16
	system, err := frontend.Compile(
		ecc.BN254,
		scs.NewBuilder,
		&statementPublicInputCircuit{rounds: rounds},
	)
	if err != nil {
		t.Fatalf("compile fixture: %v", err)
	}
	spr := system.(*cs.SparseR1CS)
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}

	alpha := fr.NewElement(123456789)
	for _, world := range []int{2, 4} {
		t.Run(adapterWorldName(world), func(t *testing.T) {
			preprocessings := preprocessStatementPublicInputRanks(t, spr, world)
			placement, err := NewStatementPublicInputPlacement(preprocessings)
			if err != nil {
				t.Fatalf("new placement: %v", err)
			}
			if placement.Partitions() != world || placement.PublicVariables() != spr.NbPublicVariables {
				t.Fatalf(
					"placement shape = M=%d P=%d, want M=%d P=%d",
					placement.Partitions(), placement.PublicVariables(), world, spr.NbPublicVariables,
				)
			}

			point := make([]fr.Element, 0, 2)
			point = append(point, fr.NewElement(11))
			if world == 4 {
				point = append(point, fr.NewElement(13))
			}

			var previousPublic []fr.Element
			for fixture, x := range []uint64{2, 3} {
				assignment := statementPublicInputAssignment(x, rounds)
				witness := localPIOPConcreteWitness(t, assignment)
				solution, err := spr.Solve(witness, proverConfig)
				if err != nil {
					t.Fatalf("solve fixture %d: %v", fixture, err)
				}
				publicPrefix := cloneElements(solution[:spr.NbPublicVariables])
				publicSnapshot := cloneElements(publicPrefix)
				pointSnapshot := cloneElements(point)

				got, err := placement.EvaluationsAtAlpha(publicPrefix, alpha)
				if err != nil {
					t.Fatalf("evaluate fixture %d: %v", fixture, err)
				}
				if len(got) != world {
					t.Fatalf("evaluation count = %d, want %d", len(got), world)
				}

				want := make([]fr.Element, world)
				for rank := 0; rank < world; rank++ {
					table, _, err := BuildLocalPIOPTableFromSolution(preprocessings[rank], spr, solution)
					if err != nil {
						t.Fatalf("build rank %d fixture %d: %v", rank, fixture, err)
					}
					domain := fft.NewDomain(uint64(preprocessings[rank].LocalRows()))
					coefficients := interpolateLocalTable(table.PublicInput, domain)
					want[rank] = evaluateLocalPolynomial(coefficients, alpha)
					assertLocalFieldEqual(t, got[rank], want[rank], "statement PI(alpha)")
				}

				gotFold, err := placement.FoldAtPartitionPoint(publicPrefix, alpha, point)
				if err != nil {
					t.Fatalf("fold fixture %d: %v", fixture, err)
				}
				wantFold, err := EvaluateMultilinear(want, point)
				if err != nil {
					t.Fatalf("reference multilinear fold: %v", err)
				}
				assertLocalFieldEqual(t, gotFold, wantFold, "folded statement PI")
				assertStatementPublicInputSliceEqual(t, publicPrefix, publicSnapshot, "public prefix mutated")
				assertStatementPublicInputSliceEqual(t, point, pointSnapshot, "partition point mutated")

				// Returned storage is independent, and the placement retained no
				// alias to the preprocessing objects used at construction.
				got[0].Add(&got[0], valuePointer(fr.One()))
				preprocessings[0].layout.publicVariables = 0
				again, err := placement.EvaluationsAtAlpha(publicPrefix, alpha)
				if err != nil {
					t.Fatalf("reevaluate after caller mutation: %v", err)
				}
				assertLocalFieldEqual(t, again[0], want[0], "placement or result alias")
				preprocessings[0].layout.publicVariables = spr.NbPublicVariables

				if fixture > 0 && statementPublicInputSlicesEqual(previousPublic, publicPrefix) {
					t.Fatal("distinct witnesses produced the same public prefix")
				}
				previousPublic = cloneElements(publicPrefix)
			}
		})
	}
}

func TestStatementPublicInputRejectsMalformedPlacement(t *testing.T) {
	system, err := frontend.Compile(
		ecc.BN254,
		scs.NewBuilder,
		&statementPublicInputCircuit{rounds: 16},
	)
	if err != nil {
		t.Fatalf("compile fixture: %v", err)
	}
	spr := system.(*cs.SparseR1CS)
	preprocessings := preprocessStatementPublicInputRanks(t, spr, 4)

	tests := []struct {
		name      string
		candidate func() []*LocalPIOPPreprocessing
		wantErr   error
	}{
		{
			name: "too few ranks",
			candidate: func() []*LocalPIOPPreprocessing {
				return preprocessings[:1]
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "non-power-of-two rank count",
			candidate: func() []*LocalPIOPPreprocessing {
				return preprocessings[:3]
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "nil rank",
			candidate: func() []*LocalPIOPPreprocessing {
				result := append([]*LocalPIOPPreprocessing(nil), preprocessings...)
				result[2] = nil
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "rank order",
			candidate: func() []*LocalPIOPPreprocessing {
				result := append([]*LocalPIOPPreprocessing(nil), preprocessings...)
				result[1], result[2] = result[2], result[1]
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "rank binding",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[2].rank = 1
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "world binding",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[1].world = 2
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "system digest",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[3].systemDigest[0] ^= 0xff
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "layout mismatch",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[1].layout.constraints++
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "noncanonical domain",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				for rank := range result {
					result[rank].layout.localRows = 3
					result[rank].layout.totalRows = 12
				}
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "slot label",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[2].index.SlotLabel.SetUint64(3)
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
		{
			name: "wire coset",
			candidate: func() []*LocalPIOPPreprocessing {
				result := cloneStatementPublicInputPreprocessings(preprocessings)
				result[0].index.WireCosets[LocalWireB].SetUint64(6)
				return result
			},
			wantErr: ErrInvalidStatementPublicInput,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewStatementPublicInputPlacement(test.candidate())
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestStatementPublicInputRejectsMalformedEvaluation(t *testing.T) {
	const rounds = 16
	system, err := frontend.Compile(
		ecc.BN254,
		scs.NewBuilder,
		&statementPublicInputCircuit{rounds: rounds},
	)
	if err != nil {
		t.Fatalf("compile fixture: %v", err)
	}
	spr := system.(*cs.SparseR1CS)
	placement, err := NewStatementPublicInputPlacement(preprocessStatementPublicInputRanks(t, spr, 4))
	if err != nil {
		t.Fatalf("new placement: %v", err)
	}

	assignment := statementPublicInputAssignment(2, rounds)
	witness := localPIOPConcreteWitness(t, assignment)
	proverConfig, err := backend.NewProverConfig()
	if err != nil {
		t.Fatalf("prover config: %v", err)
	}
	solution, err := spr.Solve(witness, proverConfig)
	if err != nil {
		t.Fatalf("solve fixture: %v", err)
	}
	publicPrefix := cloneElements(solution[:spr.NbPublicVariables])
	alpha := fr.NewElement(123456789)
	point := []fr.Element{fr.NewElement(11), fr.NewElement(13)}

	shortPrefix := publicPrefix[:len(publicPrefix)-1]
	longPrefix := append(cloneElements(publicPrefix), fr.One())
	for name, candidate := range map[string][]fr.Element{
		"short prefix": shortPrefix,
		"long prefix":  longPrefix,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := placement.EvaluationsAtAlpha(candidate, alpha); !errors.Is(err, ErrInvalidStatementPublicInput) {
				t.Fatalf("evaluation error = %v, want ErrInvalidStatementPublicInput", err)
			}
			if _, err := placement.FoldAtPartitionPoint(candidate, alpha, point); !errors.Is(err, ErrInvalidStatementPublicInput) {
				t.Fatalf("fold error = %v, want ErrInvalidStatementPublicInput", err)
			}
		})
	}

	for name, badAlpha := range map[string]fr.Element{
		"zero":        fr.Element{},
		"domain one":  fr.One(),
		"domain root": placement.omega,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := placement.EvaluationsAtAlpha(publicPrefix, badAlpha); !errors.Is(err, ErrInvalidStatementPublicInput) {
				t.Fatalf("evaluation error = %v, want ErrInvalidStatementPublicInput", err)
			}
			if _, err := placement.FoldAtPartitionPoint(publicPrefix, badAlpha, point); !errors.Is(err, ErrInvalidStatementPublicInput) {
				t.Fatalf("fold error = %v, want ErrInvalidStatementPublicInput", err)
			}
		})
	}

	if _, err := placement.FoldAtPartitionPoint(publicPrefix, alpha, point[:1]); !errors.Is(err, ErrInvalidStatementPublicInput) {
		t.Fatalf("short-point error = %v, want ErrInvalidStatementPublicInput", err)
	}
	if _, err := placement.FoldAtPartitionPoint(publicPrefix, alpha, append(point, fr.One())); !errors.Is(err, ErrInvalidStatementPublicInput) {
		t.Fatalf("long-point error = %v, want ErrInvalidStatementPublicInput", err)
	}
	var nilPlacement *StatementPublicInputPlacement
	if _, err := nilPlacement.EvaluationsAtAlpha(publicPrefix, alpha); !errors.Is(err, ErrInvalidStatementPublicInput) {
		t.Fatalf("nil evaluation error = %v, want ErrInvalidStatementPublicInput", err)
	}
	if _, err := nilPlacement.FoldAtPartitionPoint(publicPrefix, alpha, point); !errors.Is(err, ErrInvalidStatementPublicInput) {
		t.Fatalf("nil fold error = %v, want ErrInvalidStatementPublicInput", err)
	}
}

func preprocessStatementPublicInputRanks(
	t *testing.T,
	spr *cs.SparseR1CS,
	world int,
) []*LocalPIOPPreprocessing {
	t.Helper()
	result := make([]*LocalPIOPPreprocessing, world)
	for rank := range result {
		var err error
		result[rank], err = PreprocessLocalPIOP(spr, rank, world)
		if err != nil {
			t.Fatalf("preprocess rank %d/%d: %v", rank, world, err)
		}
	}
	return result
}

func statementPublicInputAssignment(xValue uint64, rounds int) *statementPublicInputCircuit {
	x := fr.NewElement(xValue)
	value := x
	for round := 0; round < rounds; round++ {
		value.Mul(&value, &x)
	}
	assignment := &statementPublicInputCircuit{X: x, rounds: rounds}
	assignment.Public[0] = value
	publicOne := value
	publicOne.Add(&publicOne, valuePointer(fr.NewElement(7)))
	publicTwo := value
	publicTwo.Mul(&publicTwo, &publicOne)
	assignment.Public[1] = publicOne
	assignment.Public[2] = publicTwo
	return assignment
}

func cloneStatementPublicInputPreprocessings(
	source []*LocalPIOPPreprocessing,
) []*LocalPIOPPreprocessing {
	result := make([]*LocalPIOPPreprocessing, len(source))
	for rank := range source {
		if source[rank] != nil {
			copyOfRank := *source[rank]
			result[rank] = &copyOfRank
		}
	}
	return result
}

func assertStatementPublicInputSliceEqual(
	t *testing.T,
	got []fr.Element,
	want []fr.Element,
	label string,
) {
	t.Helper()
	if !statementPublicInputSlicesEqual(got, want) {
		t.Fatalf("%s", label)
	}
}

func statementPublicInputSlicesEqual(left, right []fr.Element) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if !left[index].Equal(&right[index]) {
			return false
		}
	}
	return true
}
