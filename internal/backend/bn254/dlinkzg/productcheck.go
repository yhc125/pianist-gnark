package dlinkzg

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

var (
	// ErrEmptyBoundaryFactors is returned when no partition boundary factors
	// are supplied.
	ErrEmptyBoundaryFactors = errors.New("dlinkzg: empty boundary-factor vector")
	// ErrInvalidProductCheckSize is returned when a ProductCheck input does
	// not have the power-of-two shape fixed by M=2^m in the protocol.
	ErrInvalidProductCheckSize = errors.New("dlinkzg: invalid ProductCheck size")
	// ErrInvalidProductCheckTable is returned when a table does not satisfy
	// the internal-node equations or the root claim.
	ErrInvalidProductCheckTable = errors.New("dlinkzg: invalid ProductCheck table")
)

// PadBoundaryFactors pads a nonempty vector of logical boundary factors to
// the next power of two with multiplicative identities. The protocol assumes
// M=2^m and M>=2, so a singleton vector is padded to length two.
//
// Padding is explicit: BuildProductCheckTable accepts only the already-padded
// public relation, and therefore cannot silently change M.
func PadBoundaryFactors(factors []fr.Element) ([]fr.Element, error) {
	if len(factors) == 0 {
		return nil, ErrEmptyBoundaryFactors
	}

	paddedLen := nextPowerOfTwoAtLeastTwo(len(factors))
	padded := make([]fr.Element, paddedLen)
	copy(padded, factors)
	for i := len(factors); i < paddedLen; i++ {
		padded[i].SetOne()
	}
	return padded, nil
}

// BuildProductCheckTable constructs the length-2M table from Equation
// (product-tree):
//
//	t[i] = rho_i,
//	t[M+x] = t[2x] * t[2x+1] for 0 <= x < M-1,
//	t[2M-1] = 0.
//
// The returned table is deterministic. In particular, its final unused entry
// is the honest canonical zero. The function does not require the root to be
// one; VerifyProductCheckTable enforces that public claim.
func BuildProductCheckTable(factors []fr.Element) ([]fr.Element, error) {
	m := len(factors)
	if m < 2 || !isPowerOfTwo(m) {
		return nil, fmt.Errorf("%w: boundary-factor length %d is not M=2^m with M>=2", ErrInvalidProductCheckSize, m)
	}

	table := make([]fr.Element, 2*m)
	copy(table[:m], factors)
	for x := 0; x < m-1; x++ {
		table[m+x].Mul(&table[2*x], &table[2*x+1])
	}
	// table[2*m-1] is already the canonical zero.
	return table, nil
}

// VerifyProductCheckTable checks all internal-node equations and the root
// claim t[2M-2]=1. It deliberately ignores t[2M-1]: the paper designates it
// as an unused entry that the honest prover sets to zero, not as an accepted-
// transcript constraint.
func VerifyProductCheckTable(table []fr.Element) error {
	if len(table) < 4 || len(table)%2 != 0 {
		return fmt.Errorf("%w: table length %d is not 2M with M>=2", ErrInvalidProductCheckSize, len(table))
	}
	m := len(table) / 2
	if !isPowerOfTwo(m) {
		return fmt.Errorf("%w: inferred M=%d is not a power of two", ErrInvalidProductCheckSize, m)
	}

	for x := 0; x < m-1; x++ {
		var want fr.Element
		want.Mul(&table[2*x], &table[2*x+1])
		if !want.Equal(&table[m+x]) {
			return fmt.Errorf("%w: node %d does not equal children %d and %d", ErrInvalidProductCheckTable, m+x, 2*x, 2*x+1)
		}
	}

	var one fr.Element
	one.SetOne()
	if !table[2*m-2].Equal(&one) {
		return fmt.Errorf("%w: root at index %d is not one", ErrInvalidProductCheckTable, 2*m-2)
	}
	return nil
}

// SplitProductCheckTable returns the coefficient vectors of the two auxiliary
// polynomials t_0 and t_1. The returned slices do not alias table.
func SplitProductCheckTable(table []fr.Element) (t0, t1 []fr.Element, err error) {
	if len(table) < 4 || len(table)%2 != 0 {
		return nil, nil, fmt.Errorf("%w: table length %d is not 2M with M>=2", ErrInvalidProductCheckSize, len(table))
	}
	m := len(table) / 2
	if !isPowerOfTwo(m) {
		return nil, nil, fmt.Errorf("%w: inferred M=%d is not a power of two", ErrInvalidProductCheckSize, m)
	}

	t0 = append([]fr.Element(nil), table[:m]...)
	t1 = append([]fr.Element(nil), table[m:]...)
	return t0, t1, nil
}

func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

func nextPowerOfTwoAtLeastTwo(n int) int {
	if n <= 2 {
		return 2
	}
	return 1 << bits.Len(uint(n-1))
}
