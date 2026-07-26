package dlinkzg

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// ErrSumCheckProverState reports a state-machine call made before its
// prerequisite or after all SumCheck rounds are complete.
var ErrSumCheckProverState = errors.New("dlinkzg: invalid stateful SumCheck prover operation")

// DegreeFiveSumCheckProver incrementally constructs the degree-five SumCheck
// transcript. A round polynomial must be obtained with NextRound before its
// Fiat--Shamir challenge is bound with BindChallenge. The prover owns a deep
// copy of every oracle table supplied to NewDegreeFiveSumCheckProver.
//
// Challenge exclusion rules belong to the outer transcript. In particular,
// BindChallenge deliberately accepts zero and one.
type DegreeFiveSumCheckProver struct {
	composition  SumCheckComposition
	working      [][]fr.Element
	runningClaim fr.Element
	round        int
	roundCount   int

	pendingRound [SumCheckRoundCoefficients]fr.Element
	awaitingBind bool
	complete     bool
}

// NewDegreeFiveSumCheckProver validates the common power-of-two oracle shape,
// checks the initial hypercube claim, and takes an immutable snapshot of the
// oracle tables. At least one SumCheck round is required.
func NewDegreeFiveSumCheckProver(
	claim fr.Element,
	oracleTables [][]fr.Element,
	composition SumCheckComposition,
) (*DegreeFiveSumCheckProver, error) {
	if composition == nil {
		return nil, fmt.Errorf("%w: nil composition", ErrInvalidSumCheckInput)
	}
	if len(oracleTables) == 0 {
		return nil, fmt.Errorf("%w: no multilinear oracle tables", ErrInvalidSumCheckInput)
	}
	tableLength := len(oracleTables[0])
	if tableLength < 2 || !isPowerOfTwo(tableLength) {
		return nil, fmt.Errorf(
			"%w: oracle table length %d is not a power of two at least two",
			ErrInvalidSumCheckInput,
			tableLength,
		)
	}
	roundCount := bits.Len(uint(tableLength)) - 1
	if roundCount <= 0 || roundCount >= bits.UintSize {
		return nil, fmt.Errorf(
			"%w: round count %d is outside the supported range",
			ErrInvalidSumCheckInput,
			roundCount,
		)
	}

	working := make([][]fr.Element, len(oracleTables))
	for oracle := range oracleTables {
		if len(oracleTables[oracle]) != tableLength {
			return nil, fmt.Errorf(
				"%w: oracle table %d has length %d, want %d",
				ErrInvalidSumCheckInput,
				oracle,
				len(oracleTables[oracle]),
				tableLength,
			)
		}
		working[oracle] = append([]fr.Element(nil), oracleTables[oracle]...)
	}

	values := make([]fr.Element, len(working))
	var computedClaim fr.Element
	for index := 0; index < tableLength; index++ {
		for oracle := range working {
			values[oracle] = working[oracle][index]
		}
		term := composition(values)
		computedClaim.Add(&computedClaim, &term)
	}
	if !computedClaim.Equal(&claim) {
		return nil, fmt.Errorf("%w: declared claim does not equal the oracle hypercube sum", ErrInvalidSumCheckClaim)
	}

	return &DegreeFiveSumCheckProver{
		composition:  composition,
		working:      working,
		runningClaim: claim,
		roundCount:   roundCount,
	}, nil
}

// NextRound returns the current round polynomial as exactly six coefficients
// in ascending monomial order. The returned array does not alias prover state.
// BindChallenge must be called before another round can be requested.
func (p *DegreeFiveSumCheckProver) NextRound() ([SumCheckRoundCoefficients]fr.Element, error) {
	var result [SumCheckRoundCoefficients]fr.Element
	if p == nil {
		return result, fmt.Errorf("%w: nil prover", ErrSumCheckProverState)
	}
	if p.complete || p.round >= p.roundCount {
		return result, fmt.Errorf("%w: all rounds are complete", ErrSumCheckProverState)
	}
	if p.awaitingBind {
		return result, fmt.Errorf("%w: round %d is awaiting its challenge", ErrSumCheckProverState, p.round)
	}
	if len(p.working) == 0 || len(p.working[0]) < 2 || len(p.working[0])%2 != 0 {
		return result, fmt.Errorf("%w: malformed working table at round %d", ErrSumCheckProverState, p.round)
	}

	pairCount := len(p.working[0]) / 2
	valuesAtPoints := make([]fr.Element, SumCheckRoundCoefficients)
	oracleValues := make([]fr.Element, len(p.working))
	for pointIndex := 0; pointIndex < SumCheckRoundCoefficients; pointIndex++ {
		point := fr.NewElement(uint64(pointIndex))
		var oneMinusPoint fr.Element
		oneMinusPoint.SetOne().Sub(&oneMinusPoint, &point)
		for suffix := 0; suffix < pairCount; suffix++ {
			for oracle := range p.working {
				var evenPart, oddPart fr.Element
				evenPart.Mul(&p.working[oracle][2*suffix], &oneMinusPoint)
				oddPart.Mul(&p.working[oracle][2*suffix+1], &point)
				oracleValues[oracle].Add(&evenPart, &oddPart)
			}
			term := p.composition(oracleValues)
			valuesAtPoints[pointIndex].Add(&valuesAtPoints[pointIndex], &term)
		}
	}

	coefficients := interpolateDegreeFive(valuesAtPoints)
	if len(coefficients) != SumCheckRoundCoefficients {
		return result, fmt.Errorf("%w: interpolation returned %d coefficients", ErrSumCheckProverState, len(coefficients))
	}
	atZero := coefficients[0]
	atOne := evaluateCoefficientPolynomial(coefficients, fr.One())
	var roundSum fr.Element
	roundSum.Add(&atZero, &atOne)
	if !roundSum.Equal(&p.runningClaim) {
		return result, fmt.Errorf(
			"%w at round %d: polynomial endpoints do not match the running claim",
			ErrInvalidSumCheckClaim,
			p.round+1,
		)
	}

	copy(p.pendingRound[:], coefficients)
	p.awaitingBind = true
	return p.pendingRound, nil
}

// BindChallenge binds the challenge for the round most recently returned by
// NextRound and folds every oracle table. Zero and one are valid inputs here;
// any exclusion policy is enforced by the outer transcript.
func (p *DegreeFiveSumCheckProver) BindChallenge(challenge fr.Element) error {
	if p == nil {
		return fmt.Errorf("%w: nil prover", ErrSumCheckProverState)
	}
	if p.complete || p.round >= p.roundCount {
		return fmt.Errorf("%w: all rounds are complete", ErrSumCheckProverState)
	}
	if !p.awaitingBind {
		return fmt.Errorf("%w: round %d has not been requested", ErrSumCheckProverState, p.round)
	}

	p.runningClaim = evaluateCoefficientPolynomial(p.pendingRound[:], challenge)
	var oneMinusChallenge fr.Element
	oneMinusChallenge.SetOne().Sub(&oneMinusChallenge, &challenge)
	for oracle := range p.working {
		pairCount := len(p.working[oracle]) / 2
		next := make([]fr.Element, pairCount)
		for suffix := 0; suffix < pairCount; suffix++ {
			var evenPart, oddPart fr.Element
			evenPart.Mul(&p.working[oracle][2*suffix], &oneMinusChallenge)
			oddPart.Mul(&p.working[oracle][2*suffix+1], &challenge)
			next[suffix].Add(&evenPart, &oddPart)
		}
		p.working[oracle] = next
	}

	p.awaitingBind = false
	p.round++
	if p.round == p.roundCount {
		p.complete = true
	}
	return nil
}

// FinalEvaluations returns the final folded value of every oracle. It is
// available only after every round challenge has been bound. Repeated calls
// are read-only and return fresh slices. The final composition check detects a
// composition whose individual degree exceeded five along the bound path.
func (p *DegreeFiveSumCheckProver) FinalEvaluations() ([]fr.Element, error) {
	if p == nil {
		return nil, fmt.Errorf("%w: nil prover", ErrSumCheckProverState)
	}
	if !p.complete || p.awaitingBind || p.round != p.roundCount {
		return nil, fmt.Errorf(
			"%w: completed %d of %d rounds",
			ErrSumCheckProverState,
			p.round,
			p.roundCount,
		)
	}

	finalEvaluations := make([]fr.Element, len(p.working))
	for oracle := range p.working {
		if len(p.working[oracle]) != 1 {
			return nil, fmt.Errorf(
				"%w: oracle %d has final length %d",
				ErrSumCheckProverState,
				oracle,
				len(p.working[oracle]),
			)
		}
		finalEvaluations[oracle] = p.working[oracle][0]
	}
	endpoint := p.composition(finalEvaluations)
	if !p.runningClaim.Equal(&endpoint) {
		return nil, ErrSumCheckDegree
	}
	return finalEvaluations, nil
}
