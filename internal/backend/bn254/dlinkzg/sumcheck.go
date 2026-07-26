package dlinkzg

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// SumCheckRoundDegree is the declared individual degree of every round
	// polynomial in Protocol 1.
	SumCheckRoundDegree = 5
	// SumCheckRoundCoefficients is the canonical encoded length. Lower-degree
	// rounds retain leading zero coefficients up to this length.
	SumCheckRoundCoefficients = SumCheckRoundDegree + 1
)

var (
	ErrInvalidSumCheckInput      = errors.New("dlinkzg: invalid SumCheck input")
	ErrInvalidSumCheckClaim      = errors.New("dlinkzg: invalid SumCheck claim")
	ErrInvalidSumCheckTranscript = errors.New("dlinkzg: invalid SumCheck transcript")
	ErrSumCheckDegree            = errors.New("dlinkzg: SumCheck composition exceeds declared degree five")
)

// SumCheckComposition evaluates the arithmetic composition of the supplied
// multilinear-oracle values. Protocol 1 uses it for
// eq(theta,z)*(R_0(z)+zeta*R_1(z)+zeta^2*R_2(z)). The caller must provide a
// composition whose individual degree is at most five.
type SumCheckComposition func(multilinearValues []fr.Element) fr.Element

// SumCheckTranscript contains coefficient-form round polynomials in ascending
// monomial order. Every round must contain exactly six coefficients.
type SumCheckTranscript struct {
	Rounds [][]fr.Element
}

// ProveDegreeFiveSumCheck runs the prover side of the partition-index
// SumCheck. oracleTables contains Boolean-hypercube evaluation tables of the
// multilinear inputs to composition, all in little-endian index order. The
// challenge vector is supplied by the caller; this primitive performs no
// Fiat--Shamir or network I/O.
//
// The implementation folds the multilinear tables after every challenge, so
// its table work is linear in the initial table size. It returns the final
// folded oracle evaluations needed to construct the authenticated endpoint.
func ProveDegreeFiveSumCheck(
	claim fr.Element,
	oracleTables [][]fr.Element,
	composition SumCheckComposition,
	challenges []fr.Element,
) (SumCheckTranscript, []fr.Element, error) {
	if composition == nil {
		return SumCheckTranscript{}, nil, fmt.Errorf("%w: nil composition", ErrInvalidSumCheckInput)
	}
	if len(oracleTables) == 0 {
		return SumCheckTranscript{}, nil, fmt.Errorf("%w: no multilinear oracle tables", ErrInvalidSumCheckInput)
	}
	if len(challenges) == 0 || len(challenges) >= bits.UintSize {
		return SumCheckTranscript{}, nil, fmt.Errorf("%w: challenge arity %d is outside the supported range", ErrInvalidSumCheckInput, len(challenges))
	}

	expectedLen := 1 << uint(len(challenges))
	working := make([][]fr.Element, len(oracleTables))
	for i := range oracleTables {
		if len(oracleTables[i]) != expectedLen {
			return SumCheckTranscript{}, nil, fmt.Errorf(
				"%w: oracle table %d has length %d, want %d",
				ErrInvalidSumCheckInput, i, len(oracleTables[i]), expectedLen,
			)
		}
		working[i] = append([]fr.Element(nil), oracleTables[i]...)
	}

	transcript := SumCheckTranscript{Rounds: make([][]fr.Element, 0, len(challenges))}
	runningClaim := claim
	for round, challenge := range challenges {
		pairCount := len(working[0]) / 2
		valuesAtPoints := make([]fr.Element, SumCheckRoundCoefficients)
		oracleValues := make([]fr.Element, len(working))

		for pointIndex := 0; pointIndex < SumCheckRoundCoefficients; pointIndex++ {
			point := fr.NewElement(uint64(pointIndex))
			var oneMinusPoint fr.Element
			oneMinusPoint.SetOne().Sub(&oneMinusPoint, &point)

			for suffix := 0; suffix < pairCount; suffix++ {
				for oracle := range working {
					var evenPart, oddPart fr.Element
					evenPart.Mul(&working[oracle][2*suffix], &oneMinusPoint)
					oddPart.Mul(&working[oracle][2*suffix+1], &point)
					oracleValues[oracle].Add(&evenPart, &oddPart)
				}
				term := composition(oracleValues)
				valuesAtPoints[pointIndex].Add(&valuesAtPoints[pointIndex], &term)
			}
		}

		coefficients := interpolateDegreeFive(valuesAtPoints)
		atZero := coefficients[0]
		atOne := evaluateCoefficientPolynomial(coefficients, fr.NewElement(1))
		var roundSum fr.Element
		roundSum.Add(&atZero, &atOne)
		if !roundSum.Equal(&runningClaim) {
			return SumCheckTranscript{}, nil, fmt.Errorf(
				"%w at round %d: polynomial endpoints do not match the running claim",
				ErrInvalidSumCheckClaim, round+1,
			)
		}

		transcript.Rounds = append(transcript.Rounds, coefficients)
		runningClaim = evaluateCoefficientPolynomial(coefficients, challenge)

		var oneMinusChallenge fr.Element
		oneMinusChallenge.SetOne().Sub(&oneMinusChallenge, &challenge)
		for oracle := range working {
			next := make([]fr.Element, pairCount)
			for suffix := 0; suffix < pairCount; suffix++ {
				var evenPart, oddPart fr.Element
				evenPart.Mul(&working[oracle][2*suffix], &oneMinusChallenge)
				oddPart.Mul(&working[oracle][2*suffix+1], &challenge)
				next[suffix].Add(&evenPart, &oddPart)
			}
			working[oracle] = next
		}
	}

	finalEvaluations := make([]fr.Element, len(working))
	for i := range working {
		finalEvaluations[i] = working[i][0]
	}
	endpoint := composition(finalEvaluations)
	if !runningClaim.Equal(&endpoint) {
		return SumCheckTranscript{}, nil, ErrSumCheckDegree
	}

	return transcript, finalEvaluations, nil
}

// VerifyDegreeFiveSumCheck verifies the fixed six-coefficient encoding, all
// round-sum equations, and the final endpoint equation. endpoint is computed
// by the caller from authenticated folded evaluations, as in Equation
// (endpoint-sumcheck-check).
func VerifyDegreeFiveSumCheck(
	claim fr.Element,
	challenges []fr.Element,
	transcript SumCheckTranscript,
	endpoint fr.Element,
) error {
	if len(challenges) == 0 {
		return fmt.Errorf("%w: empty challenge vector", ErrInvalidSumCheckTranscript)
	}
	if len(transcript.Rounds) != len(challenges) {
		return fmt.Errorf(
			"%w: got %d rounds for %d challenges",
			ErrInvalidSumCheckTranscript, len(transcript.Rounds), len(challenges),
		)
	}

	runningClaim := claim
	for round := range transcript.Rounds {
		coefficients := transcript.Rounds[round]
		if len(coefficients) != SumCheckRoundCoefficients {
			return fmt.Errorf(
				"%w: round %d has %d coefficients, want exactly %d",
				ErrInvalidSumCheckTranscript, round+1, len(coefficients), SumCheckRoundCoefficients,
			)
		}

		atZero := coefficients[0]
		atOne := evaluateCoefficientPolynomial(coefficients, fr.NewElement(1))
		var roundSum fr.Element
		roundSum.Add(&atZero, &atOne)
		if !roundSum.Equal(&runningClaim) {
			return fmt.Errorf("%w: round %d sum check failed", ErrInvalidSumCheckTranscript, round+1)
		}
		runningClaim = evaluateCoefficientPolynomial(coefficients, challenges[round])
	}

	if !runningClaim.Equal(&endpoint) {
		return fmt.Errorf("%w: endpoint check failed", ErrInvalidSumCheckTranscript)
	}
	return nil
}

// EvaluateMultilinear evaluates a little-endian Boolean-hypercube table at an
// arbitrary point by repeated adjacent-pair folding.
func EvaluateMultilinear(evaluations, point []fr.Element) (fr.Element, error) {
	if len(evaluations) == 0 || !isPowerOfTwo(len(evaluations)) {
		return fr.Element{}, fmt.Errorf("%w: evaluation-table length %d is not a power of two", ErrInvalidSumCheckInput, len(evaluations))
	}
	if len(point) >= bits.UintSize || 1<<uint(len(point)) != len(evaluations) {
		return fr.Element{}, fmt.Errorf(
			"%w: point arity %d does not match table length %d",
			ErrInvalidSumCheckInput, len(point), len(evaluations),
		)
	}

	working := append([]fr.Element(nil), evaluations...)
	for _, coordinate := range point {
		var oneMinusCoordinate fr.Element
		oneMinusCoordinate.SetOne().Sub(&oneMinusCoordinate, &coordinate)
		next := make([]fr.Element, len(working)/2)
		for i := range next {
			var evenPart, oddPart fr.Element
			evenPart.Mul(&working[2*i], &oneMinusCoordinate)
			oddPart.Mul(&working[2*i+1], &coordinate)
			next[i].Add(&evenPart, &oddPart)
		}
		working = next
	}
	return working[0], nil
}

// EqualityEvaluationTable returns the Boolean evaluations of
// eq(theta,z)=product_k ((1-theta_k)(1-z_k)+theta_k*z_k), using the same
// little-endian bit convention as Protocol 1.
func EqualityEvaluationTable(theta []fr.Element) ([]fr.Element, error) {
	if len(theta) >= bits.UintSize {
		return nil, fmt.Errorf("%w: equality-table arity %d is too large", ErrInvalidSumCheckInput, len(theta))
	}

	table := make([]fr.Element, 1<<uint(len(theta)))
	for index := range table {
		table[index].SetOne()
		for coordinate := range theta {
			var factor fr.Element
			if index&(1<<uint(coordinate)) != 0 {
				factor = theta[coordinate]
			} else {
				factor.SetOne().Sub(&factor, &theta[coordinate])
			}
			table[index].Mul(&table[index], &factor)
		}
	}
	return table, nil
}

func interpolateDegreeFive(values []fr.Element) []fr.Element {
	coefficients := make([]fr.Element, SumCheckRoundCoefficients)
	for j := 0; j < SumCheckRoundCoefficients; j++ {
		basis := []fr.Element{fr.NewElement(1)}
		denominator := fr.NewElement(1)
		xj := fr.NewElement(uint64(j))

		for k := 0; k < SumCheckRoundCoefficients; k++ {
			if k == j {
				continue
			}
			xk := fr.NewElement(uint64(k))
			var difference fr.Element
			difference.Sub(&xj, &xk)
			denominator.Mul(&denominator, &difference)

			var negativeXk fr.Element
			negativeXk.Neg(&xk)
			nextBasis := make([]fr.Element, len(basis)+1)
			for degree := range basis {
				var constantTerm fr.Element
				constantTerm.Mul(&basis[degree], &negativeXk)
				nextBasis[degree].Add(&nextBasis[degree], &constantTerm)
				nextBasis[degree+1].Add(&nextBasis[degree+1], &basis[degree])
			}
			basis = nextBasis
		}

		var scale fr.Element
		scale.Div(&values[j], &denominator)
		for degree := range basis {
			var term fr.Element
			term.Mul(&basis[degree], &scale)
			coefficients[degree].Add(&coefficients[degree], &term)
		}
	}
	return coefficients
}

func evaluateCoefficientPolynomial(coefficients []fr.Element, point fr.Element) fr.Element {
	var result fr.Element
	for i := len(coefficients) - 1; i >= 0; i-- {
		result.Mul(&result, &point)
		result.Add(&result, &coefficients[i])
	}
	return result
}
