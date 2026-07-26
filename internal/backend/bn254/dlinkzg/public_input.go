package dlinkzg

// This file is the verifier-side public-input placement map. It derives the
// PI evaluations used by the outer reduction from the public statement and
// fixed preprocessing alone; neither a full SparseR1CS solution nor a prover
// message is accepted at this boundary.

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// ErrInvalidStatementPublicInput reports a malformed fixed placement, public
// prefix, alpha, or partition-fold point.
var ErrInvalidStatementPublicInput = errors.New("dlinkzg: invalid statement public input")

// StatementPublicInputPlacement is the validated placement of a SparseR1CS
// public-solution prefix into rank-local PI columns. The adapter puts every
// public placeholder in the first rows of rank zero, and every other PI entry
// is zero. Construction performs the O(M) cross-rank validation once; the
// evaluation methods do not retain or mutate caller-owned field elements.
type StatementPublicInputPlacement struct {
	partitions      int
	publicVariables int
	localRows       int
	omega           fr.Element
}

// NewStatementPublicInputPlacement validates a complete, rank-ordered set of
// local preprocessings. All entries must describe the same constraint system
// and canonical adapter layout, with entry i bound to rank i of M.
func NewStatementPublicInputPlacement(
	preprocessings []*LocalPIOPPreprocessing,
) (*StatementPublicInputPlacement, error) {
	m := len(preprocessings)
	if m < 2 || !isPowerOfTwo(m) {
		return nil, fmt.Errorf(
			"%w: preprocessing count M=%d is not a power of two at least two",
			ErrInvalidStatementPublicInput, m,
		)
	}
	if preprocessings[0] == nil {
		return nil, fmt.Errorf("%w: nil preprocessing at rank 0", ErrInvalidStatementPublicInput)
	}

	first := preprocessings[0]
	if err := validateStatementPublicInputLayout(first.layout, m); err != nil {
		return nil, err
	}
	for rank, preprocessing := range preprocessings {
		if preprocessing == nil {
			return nil, fmt.Errorf(
				"%w: nil preprocessing at rank %d",
				ErrInvalidStatementPublicInput, rank,
			)
		}
		if preprocessing.rank != rank || preprocessing.world != m {
			return nil, fmt.Errorf(
				"%w: entry %d is bound to rank/world %d/%d, want %d/%d",
				ErrInvalidStatementPublicInput, rank,
				preprocessing.rank, preprocessing.world, rank, m,
			)
		}
		if preprocessing.layout != first.layout || preprocessing.systemDigest != first.systemDigest {
			return nil, fmt.Errorf(
				"%w: preprocessing at rank %d has a different system or layout",
				ErrInvalidStatementPublicInput, rank,
			)
		}
		if err := validateStatementPublicInputIndex(preprocessing, rank); err != nil {
			return nil, err
		}
	}

	omega, err := sparseR1CSAdapterRoot(first.layout.localRows)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidStatementPublicInput, err)
	}
	return &StatementPublicInputPlacement{
		partitions:      m,
		publicVariables: first.layout.publicVariables,
		localRows:       first.layout.localRows,
		omega:           omega,
	}, nil
}

// Partitions returns the number of PI evaluations produced by this placement.
func (placement *StatementPublicInputPlacement) Partitions() int {
	if placement == nil {
		return 0
	}
	return placement.partitions
}

// PublicVariables returns the exact required public-solution prefix length.
func (placement *StatementPublicInputPlacement) PublicVariables() int {
	if placement == nil {
		return 0
	}
	return placement.publicVariables
}

// EvaluationsAtAlpha returns the rank-ordered vector
// (PI_0(alpha),...,PI_{M-1}(alpha)). Since public placeholders reside only on
// rank zero, entries 1 through M-1 are canonically zero. Alpha must be outside
// the canonical local domain H_X (and nonzero, as required by the local PIOP).
func (placement *StatementPublicInputPlacement) EvaluationsAtAlpha(
	publicSolutionPrefix []fr.Element,
	alpha fr.Element,
) ([]fr.Element, error) {
	value, err := placement.rankZeroEvaluationAtAlpha(publicSolutionPrefix, alpha)
	if err != nil {
		return nil, err
	}
	result := make([]fr.Element, placement.partitions)
	result[0] = value
	return result, nil
}

// FoldAtPartitionPoint evaluates the multilinear extension of the
// rank-ordered PI(alpha) vector at point. Rank zero has the all-zero Boolean
// index, so the result is PI_0(alpha) * product_k (1-point_k). After the
// placement has been validated, this costs O(P+log M) field operations and
// does not allocate an M-entry table or run an FFT.
func (placement *StatementPublicInputPlacement) FoldAtPartitionPoint(
	publicSolutionPrefix []fr.Element,
	alpha fr.Element,
	point []fr.Element,
) (fr.Element, error) {
	if placement == nil {
		return fr.Element{}, fmt.Errorf("%w: nil placement", ErrInvalidStatementPublicInput)
	}
	wantArity := bits.TrailingZeros(uint(placement.partitions))
	if len(point) != wantArity {
		return fr.Element{}, fmt.Errorf(
			"%w: partition point has arity %d, want log2(M)=%d",
			ErrInvalidStatementPublicInput, len(point), wantArity,
		)
	}

	result, err := placement.rankZeroEvaluationAtAlpha(publicSolutionPrefix, alpha)
	if err != nil {
		return fr.Element{}, err
	}
	for coordinate := range point {
		var oneMinusCoordinate fr.Element
		oneMinusCoordinate.SetOne().Sub(&oneMinusCoordinate, &point[coordinate])
		result.Mul(&result, &oneMinusCoordinate)
	}
	return result, nil
}

func (placement *StatementPublicInputPlacement) rankZeroEvaluationAtAlpha(
	publicSolutionPrefix []fr.Element,
	alpha fr.Element,
) (fr.Element, error) {
	if placement == nil {
		return fr.Element{}, fmt.Errorf("%w: nil placement", ErrInvalidStatementPublicInput)
	}
	if len(publicSolutionPrefix) != placement.publicVariables {
		return fr.Element{}, fmt.Errorf(
			"%w: public-solution prefix has length %d, want %d",
			ErrInvalidStatementPublicInput, len(publicSolutionPrefix), placement.publicVariables,
		)
	}
	if err := validateAlphaOutsideLocalDomain(alpha, placement.localRows); err != nil {
		return fr.Element{}, fmt.Errorf("%w: %v", ErrInvalidStatementPublicInput, err)
	}
	if placement.publicVariables == 0 {
		return fr.Element{}, nil
	}

	// For H_X={omega^j}_{j=0}^{T-1}, the j-th Lagrange basis value is
	//   L_j(alpha) = (alpha^T-1) omega^j / (T (alpha-omega^j)).
	// Only the first P basis values are needed. Batch inversion keeps this
	// linear in P and leaves the caller-owned public prefix untouched.
	denominators := make([]fr.Element, placement.publicVariables)
	powers := make([]fr.Element, placement.publicVariables)
	powers[0].SetOne()
	for row := 1; row < placement.publicVariables; row++ {
		powers[row].Mul(&powers[row-1], &placement.omega)
	}
	for row := range denominators {
		denominators[row].Sub(&alpha, &powers[row])
	}
	inverseDenominators := fr.BatchInvert(denominators)

	alphaToT := powerLocalField(alpha, placement.localRows)
	one := fr.One()
	var scale, inverseT fr.Element
	scale.Sub(&alphaToT, &one)
	inverseT.SetUint64(uint64(placement.localRows)).Inverse(&inverseT)
	scale.Mul(&scale, &inverseT)

	var result fr.Element
	for row := range publicSolutionPrefix {
		var term fr.Element
		term.Mul(&publicSolutionPrefix[row], &powers[row])
		term.Mul(&term, &inverseDenominators[row])
		result.Add(&result, &term)
	}
	result.Mul(&result, &scale)
	return result, nil
}

func validateStatementPublicInputLayout(layout localPIOPAdapterLayout, world int) error {
	if layout.publicVariables < 0 || layout.constraints < 0 || layout.variables <= 0 ||
		layout.logicalRows <= 0 || layout.localRows < 2 || layout.totalRows <= 0 {
		return fmt.Errorf("%w: malformed preprocessing layout", ErrInvalidStatementPublicInput)
	}
	if layout.publicVariables > layout.variables || layout.publicVariables > layout.localRows {
		return fmt.Errorf("%w: public placeholders do not fit the canonical layout", ErrInvalidStatementPublicInput)
	}
	if layout.publicVariables > int(^uint(0)>>1)-layout.constraints ||
		layout.logicalRows != layout.publicVariables+layout.constraints {
		return fmt.Errorf("%w: inconsistent logical-row count", ErrInvalidStatementPublicInput)
	}
	requestedLocalRows := layout.logicalRows / world
	if layout.logicalRows%world != 0 {
		requestedLocalRows++
	}
	if requestedLocalRows < 2 || requestedLocalRows < layout.publicVariables {
		return fmt.Errorf("%w: invalid unpadded local-row count %d", ErrInvalidStatementPublicInput, requestedLocalRows)
	}
	if requestedLocalRows > 1<<bn254AdapterMaxTwoAdicity {
		return fmt.Errorf("%w: local-row request exceeds BN254 two-adicity", ErrInvalidStatementPublicInput)
	}
	canonicalLocalRows := 1 << bits.Len(uint(requestedLocalRows-1))
	if layout.localRows != canonicalLocalRows {
		return fmt.Errorf(
			"%w: local domain size %d is not canonical for %d requested rows",
			ErrInvalidStatementPublicInput, layout.localRows, requestedLocalRows,
		)
	}
	if !isPowerOfTwo(layout.localRows) || layout.localRows > 1<<bn254AdapterMaxTwoAdicity {
		return fmt.Errorf("%w: noncanonical local domain size %d", ErrInvalidStatementPublicInput, layout.localRows)
	}
	if layout.localRows > int(^uint(0)>>1)/world || layout.totalRows != layout.localRows*world {
		return fmt.Errorf("%w: inconsistent total-row count", ErrInvalidStatementPublicInput)
	}
	if layout.logicalRows > layout.totalRows {
		return fmt.Errorf("%w: logical rows exceed padded rows", ErrInvalidStatementPublicInput)
	}
	return nil
}

func validateStatementPublicInputIndex(preprocessing *LocalPIOPPreprocessing, rank int) error {
	wantSlot := fr.NewElement(uint64(rank))
	if !preprocessing.index.SlotLabel.Equal(&wantSlot) {
		return fmt.Errorf(
			"%w: preprocessing at rank %d has a noncanonical slot label",
			ErrInvalidStatementPublicInput, rank,
		)
	}
	wantCosets := [LocalWireCount]fr.Element{
		fr.One(),
		fr.NewElement(5),
		fr.NewElement(25),
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !preprocessing.index.WireCosets[wire].Equal(&wantCosets[wire]) {
			return fmt.Errorf(
				"%w: preprocessing at rank %d has a noncanonical wire coset %d",
				ErrInvalidStatementPublicInput, rank, wire,
			)
		}
	}
	return nil
}
