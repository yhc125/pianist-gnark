package dlinkzg

// This file deliberately implements a correctness prototype, not a
// production SNARK. It connects the paper's ProductCheck, degree-five
// SumCheck, translated coefficient-MLE claims, Laurent identity, source link,
// same-set quotients, and final delta-batched pairing. The caller still
// supplies the post-Plonk local residual table and public-coin challenges in
// the clear. A production backend must derive those objects from gnark's
// partitioned Plonk witness and an ordered Fiat--Shamir transcript.

import (
	"errors"
	"fmt"
	"math/big"
	"math/bits"
	"time"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

const (
	// CorrectnessPrototypeNotice is intentionally exported so benchmark
	// adapters cannot silently label this path as a production proof system.
	CorrectnessPrototypeNotice = "centralized correctness prototype: clear post-Plonk residuals, caller-supplied public coins, no U0-U3 party aggregation, and a known-trapdoor test SRS; not production soundness"
	terminalCircuitClaims      = 3
	terminalTreeClaims         = 5
)

var (
	ErrInvalidPrototypeInput     = errors.New("dlinkzg: invalid correctness-prototype input")
	ErrInvalidPrototypeChallenge = errors.New("dlinkzg: invalid correctness-prototype challenge")
	ErrPrototypePIOP             = errors.New("dlinkzg: correctness-prototype PIOP rejected")
	ErrPrototypeOpening          = errors.New("dlinkzg: correctness-prototype opening rejected")
)

// PrototypeInput is the honest prover's clear input. Sources contains the
// three already translated rectangular coefficient sources D_j from Protocol
// 2. LocalResiduals is the post-Plonk R_0 Boolean table; deriving it from a
// gnark SparseR1CS is intentionally outside this prototype.
type PrototypeInput struct {
	T       int
	Sources [terminalCircuitClaims][][]fr.Element
	// CircuitShiftedQueries are z_j=x_j-sigma from Equation
	// (translated-circuit-sources), not the unshifted local-row points x_j.
	CircuitShiftedQueries [terminalCircuitClaims]fr.Element
	LocalResiduals        []fr.Element
	BoundaryFactors       []fr.Element
	BoundaryNumerators    []fr.Element
	BoundaryDenominators  []fr.Element
}

// PrototypeChallenges contains the paper's public coins after the local
// quotient stage. SumCheck is also the terminal partition point r. Production
// code must replace this caller-supplied structure with transcript-derived,
// domain-separated challenges and all stated rejection counters.
type PrototypeChallenges struct {
	Theta      []fr.Element
	SumCheck   []fr.Element
	Zeta       fr.Element
	Xi         fr.Element
	Nu         fr.Element
	ZChallenge fr.Element
	Beta       fr.Element
	Kappa      fr.Element
	Delta      fr.Element
}

// PrototypeStatement is sufficient for this prototype verifier. The three
// length-M clear vectors make verification linear in M and are another reason
// this is not the production succinct statement from the paper.
type PrototypeStatement struct {
	M int
	T int

	SourceCommitments [terminalCircuitClaims]bn254.G1Affine
	TreeCommitments   [2]bn254.G1Affine
	// CircuitShiftedQueries are the public z_j=x_j-sigma values.
	CircuitShiftedQueries [terminalCircuitClaims]fr.Element
	CircuitClaims         [terminalCircuitClaims]fr.Element
	TreeClaims            [terminalTreeClaims]fr.Element

	LocalResiduals       []fr.Element
	BoundaryFactors      []fr.Element
	BoundaryNumerators   []fr.Element
	BoundaryDenominators []fr.Element
}

// PrototypeProof contains the SumCheck transcript and the U0--U3 data needed
// for the scalar Laurent check and final four-pairing equation. Values in the
// second index of PartialValues are ordered as z_ch, beta, beta^{-1}; values
// in LinearValues are ordered as beta, beta^{-1} for h_xi,t_0,t_1,S.
type PrototypeProof struct {
	SumCheck SumCheckTranscript

	PartialCommitments [terminalCircuitClaims]bn254.G1Affine
	PartialValues      [terminalCircuitClaims][3]fr.Element
	HCommitment        bn254.G1Affine
	LaurentCommitment  bn254.G1Affine
	LinearValues       [4][2]fr.Element

	SourceLink cryptodlinkzg.SourceLinkProof
	WN         bn254.G1Affine
}

// PrototypeTimings exposes the two outer phases without pretending that the
// uninstrumented prototype has the paper's complete per-round cost ledger.
type PrototypeTimings struct {
	Prove  time.Duration
	Verify time.Duration
}

// PrototypeRunResult is returned by Run. ProductionReady is always false.
type PrototypeRunResult struct {
	Statement       PrototypeStatement
	Proof           PrototypeProof
	Timings         PrototypeTimings
	Verified        bool
	ProductionReady bool
	Notice          string
}

// Run executes the correctness prototype end to end and verifies its output.
func Run(input PrototypeInput, challenges PrototypeChallenges, srs *cryptodlinkzg.MonomialSRS) (PrototypeRunResult, error) {
	var result PrototypeRunResult
	result.Notice = CorrectnessPrototypeNotice
	result.ProductionReady = false

	started := time.Now()
	statement, proof, err := Prove(input, challenges, srs)
	result.Timings.Prove = time.Since(started)
	if err != nil {
		return result, err
	}
	result.Statement = statement
	result.Proof = proof

	started = time.Now()
	err = Verify(statement, proof, challenges, srs)
	result.Timings.Verify = time.Since(started)
	if err != nil {
		return result, err
	}
	result.Verified = true
	return result, nil
}

// Prove constructs an honest ProductCheck/SumCheck/DLinKZG prototype proof.
func Prove(input PrototypeInput, challenges PrototypeChallenges, srs *cryptodlinkzg.MonomialSRS) (PrototypeStatement, PrototypeProof, error) {
	var statement PrototypeStatement
	var proof PrototypeProof
	m, t, err := validatePrototypeInput(input, challenges, srs)
	if err != nil {
		return statement, proof, err
	}

	productTable, err := BuildProductCheckTable(input.BoundaryFactors)
	if err != nil {
		return statement, proof, err
	}
	if err := VerifyProductCheckTable(productTable); err != nil {
		return statement, proof, fmt.Errorf("%w: %v", ErrPrototypePIOP, err)
	}
	t0, t1, err := SplitProductCheckTable(productTable)
	if err != nil {
		return statement, proof, err
	}

	oracleTables, composition, err := prototypeSumCheckOracles(
		input.LocalResiduals,
		input.BoundaryNumerators,
		input.BoundaryDenominators,
		productTable,
		challenges.Theta,
		challenges.Zeta,
	)
	if err != nil {
		return statement, proof, err
	}
	var zero fr.Element
	claim := sumCompositionTable(oracleTables, composition)
	if !claim.IsZero() {
		return statement, proof, fmt.Errorf("%w: declared zero claim is nonzero", ErrPrototypePIOP)
	}
	proof.SumCheck, _, err = ProveDegreeFiveSumCheck(zero, oracleTables, composition, challenges.SumCheck)
	if err != nil {
		return statement, proof, err
	}

	statement = PrototypeStatement{
		M:                     m,
		T:                     t,
		CircuitShiftedQueries: input.CircuitShiftedQueries,
		LocalResiduals:        cloneElements(input.LocalResiduals),
		BoundaryFactors:       cloneElements(input.BoundaryFactors),
		BoundaryNumerators:    cloneElements(input.BoundaryNumerators),
		BoundaryDenominators:  cloneElements(input.BoundaryDenominators),
	}
	for j := 0; j < terminalCircuitClaims; j++ {
		statement.SourceCommitments[j], err = cryptodlinkzg.CommitRect(input.Sources[j], srs)
		if err != nil {
			return PrototypeStatement{}, PrototypeProof{}, err
		}
	}
	statement.TreeCommitments[0], err = cryptodlinkzg.CommitZ(t0, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	statement.TreeCommitments[1], err = cryptodlinkzg.CommitZ(t1, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}

	queries, scales, err := translatedQueries(input.CircuitShiftedQueries, bits.Len(uint(t))-1)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	partials := make([][]fr.Element, terminalCircuitClaims)
	for j := 0; j < terminalCircuitClaims; j++ {
		partials[j] = foldRows(input.Sources[j], challenges.SumCheck, t)
		claim, mleErr := cryptodlinkzg.MLEEval(partials[j], queries[j])
		if mleErr != nil {
			return PrototypeStatement{}, PrototypeProof{}, mleErr
		}
		statement.CircuitClaims[j].Mul(&claim, &scales[j])
		proof.PartialCommitments[j], err = cryptodlinkzg.CommitZ(partials[j], srs)
		if err != nil {
			return PrototypeStatement{}, PrototypeProof{}, err
		}
	}

	treePoints := productCheckFunctionalPoints(challenges.SumCheck)
	for c := 0; c < terminalTreeClaims; c++ {
		statement.TreeClaims[c], err = evaluateTreeFunctional(t0, t1, treePoints[c])
		if err != nil {
			return PrototypeStatement{}, PrototypeProof{}, err
		}
	}
	var one fr.Element
	one.SetOne()
	if !statement.TreeClaims[4].Equal(&one) {
		return PrototypeStatement{}, PrototypeProof{}, fmt.Errorf("%w: ProductCheck root claim is not one", ErrPrototypePIOP)
	}

	dXi, h := foldedSourceAndLink(input.Sources, challenges.Xi, challenges.ZChallenge, t)
	proof.HCommitment, err = cryptodlinkzg.CommitZ(h, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	laurent := buildLaurentWitness(
		partials,
		h,
		t0,
		t1,
		queries,
		scales,
		treePoints,
		challenges,
		t,
	)
	proof.LaurentCommitment, err = cryptodlinkzg.CommitZ(laurent, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}

	betaInverse := inverse(challenges.Beta)
	for j := 0; j < terminalCircuitClaims; j++ {
		proof.PartialValues[j][0] = cryptodlinkzg.Eval(partials[j], challenges.ZChallenge)
		proof.PartialValues[j][1] = cryptodlinkzg.Eval(partials[j], challenges.Beta)
		proof.PartialValues[j][2] = cryptodlinkzg.Eval(partials[j], betaInverse)
	}
	linearPolynomials := [4][]fr.Element{padPolynomial(h, t), padPolynomial(t0, t), padPolynomial(t1, t), laurent}
	for p := range linearPolynomials {
		proof.LinearValues[p][0] = cryptodlinkzg.Eval(linearPolynomials[p], challenges.Beta)
		proof.LinearValues[p][1] = cryptodlinkzg.Eval(linearPolynomials[p], betaInverse)
	}

	proof.SourceLink, err = cryptodlinkzg.OpenSourceLink(dXi, challenges.Beta, challenges.ZChallenge, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	if !proof.SourceLink.ClaimedValue.Equal(&proof.LinearValues[0][0]) {
		return PrototypeStatement{}, PrototypeProof{}, fmt.Errorf("%w: source link does not equal h_xi(beta)", ErrPrototypeOpening)
	}

	gPoints := []fr.Element{challenges.ZChallenge, challenges.Beta, betaInverse}
	gInputs := make([]cryptodlinkzg.SameSetInput, terminalCircuitClaims)
	for j := range gInputs {
		gInputs[j] = cryptodlinkzg.SameSetInput{
			Polynomial:    partials[j],
			ClaimedValues: cloneElements(proof.PartialValues[j][:]),
		}
	}
	lPoints := []fr.Element{challenges.Beta, betaInverse}
	lInputs := make([]cryptodlinkzg.SameSetInput, len(linearPolynomials))
	for p := range lInputs {
		lInputs[p] = cryptodlinkzg.SameSetInput{
			Polynomial:    linearPolynomials[p],
			ClaimedValues: cloneElements(proof.LinearValues[p][:]),
		}
	}
	nestedBatch, err := cryptodlinkzg.BuildNestedSetQuotient(
		gInputs, gPoints, lInputs, lPoints, challenges.Kappa,
	)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	proof.WN, err = cryptodlinkzg.CommitZ(nestedBatch.Quotient, srs)
	if err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}

	if err := verifyLaurentIdentity(statement, proof, queries, scales, treePoints, challenges); err != nil {
		return PrototypeStatement{}, PrototypeProof{}, err
	}
	return statement, proof, nil
}

// Verify checks the prototype's PIOP transcript, Laurent scalar identity, and
// final delta-batched four-pairing equation. It does not claim production
// succinctness because PrototypeStatement contains clear O(M) tables.
func Verify(statement PrototypeStatement, proof PrototypeProof, challenges PrototypeChallenges, srs *cryptodlinkzg.MonomialSRS) error {
	if err := validatePrototypeStatement(statement, challenges, srs); err != nil {
		return err
	}
	productTable, err := BuildProductCheckTable(statement.BoundaryFactors)
	if err != nil {
		return err
	}
	if err := VerifyProductCheckTable(productTable); err != nil {
		return fmt.Errorf("%w: %v", ErrPrototypePIOP, err)
	}
	for i := 0; i < statement.M; i++ {
		var linked fr.Element
		linked.Mul(&statement.BoundaryFactors[i], &statement.BoundaryDenominators[i])
		if !linked.Equal(&statement.BoundaryNumerators[i]) {
			return fmt.Errorf("%w: boundary link %d failed", ErrPrototypePIOP, i)
		}
	}
	t0, t1, err := SplitProductCheckTable(productTable)
	if err != nil {
		return err
	}
	commitment, err := cryptodlinkzg.CommitZ(t0, srs)
	if err != nil {
		return err
	}
	if !commitment.Equal(&statement.TreeCommitments[0]) {
		return fmt.Errorf("%w: t_0 commitment mismatch", ErrPrototypePIOP)
	}
	commitment, err = cryptodlinkzg.CommitZ(t1, srs)
	if err != nil {
		return err
	}
	if !commitment.Equal(&statement.TreeCommitments[1]) {
		return fmt.Errorf("%w: t_1 commitment mismatch", ErrPrototypePIOP)
	}

	oracleTables, composition, err := prototypeSumCheckOracles(
		statement.LocalResiduals,
		statement.BoundaryNumerators,
		statement.BoundaryDenominators,
		productTable,
		challenges.Theta,
		challenges.Zeta,
	)
	if err != nil {
		return err
	}
	claim := sumCompositionTable(oracleTables, composition)
	if !claim.IsZero() {
		return fmt.Errorf("%w: declared zero claim is nonzero", ErrPrototypePIOP)
	}
	finalEvaluations := make([]fr.Element, len(oracleTables))
	for i := range oracleTables {
		finalEvaluations[i], err = EvaluateMultilinear(oracleTables[i], challenges.SumCheck)
		if err != nil {
			return err
		}
	}
	endpoint := composition(finalEvaluations)
	var zero fr.Element
	if err := VerifyDegreeFiveSumCheck(zero, challenges.SumCheck, proof.SumCheck, endpoint); err != nil {
		return fmt.Errorf("%w: %v", ErrPrototypePIOP, err)
	}

	queries, scales, err := translatedQueries(statement.CircuitShiftedQueries, bits.Len(uint(statement.T))-1)
	if err != nil {
		return err
	}
	treePoints := productCheckFunctionalPoints(challenges.SumCheck)
	for c := 0; c < terminalTreeClaims; c++ {
		want, evalErr := evaluateTreeFunctional(t0, t1, treePoints[c])
		if evalErr != nil {
			return evalErr
		}
		if !want.Equal(&statement.TreeClaims[c]) {
			return fmt.Errorf("%w: tree claim %d mismatch", ErrPrototypePIOP, c)
		}
	}
	var one fr.Element
	one.SetOne()
	if !statement.TreeClaims[4].Equal(&one) {
		return fmt.Errorf("%w: ProductCheck root claim is not one", ErrPrototypePIOP)
	}

	if err := verifyLaurentIdentity(statement, proof, queries, scales, treePoints, challenges); err != nil {
		return err
	}

	gPoints := []fr.Element{challenges.ZChallenge, challenges.Beta, inverse(challenges.Beta)}
	gInterpolants := make([][]fr.Element, terminalCircuitClaims)
	for j := range gInterpolants {
		gInterpolants[j], err = cryptodlinkzg.Interpolate(gPoints, proof.PartialValues[j][:])
		if err != nil {
			return err
		}
	}
	numeratorG, err := cryptodlinkzg.FoldSameSetCommitments(proof.PartialCommitments[:], gInterpolants, challenges.Kappa, srs)
	if err != nil {
		return err
	}

	lPoints := []fr.Element{challenges.Beta, inverse(challenges.Beta)}
	lInterpolants := make([][]fr.Element, len(proof.LinearValues))
	for p := range lInterpolants {
		lInterpolants[p], err = cryptodlinkzg.Interpolate(lPoints, proof.LinearValues[p][:])
		if err != nil {
			return err
		}
	}
	lCommitments := []bn254.G1Affine{
		proof.HCommitment,
		statement.TreeCommitments[0],
		statement.TreeCommitments[1],
		proof.LaurentCommitment,
	}
	numeratorL, err := cryptodlinkzg.FoldSameSetCommitments(lCommitments, lInterpolants, challenges.Kappa, srs)
	if err != nil {
		return err
	}
	innerScale := fr.One()
	for i := 0; i < terminalCircuitClaims; i++ {
		innerScale.Mul(&innerScale, &challenges.Kappa)
	}

	sourceCommitment := foldG1(statement.SourceCommitments[:], challenges.Xi)
	if !proof.SourceLink.ClaimedValue.Equal(&proof.LinearValues[0][0]) {
		return fmt.Errorf("%w: source-link value mismatch", ErrPrototypeOpening)
	}
	batchStatement := cryptodlinkzg.DeltaBatchStatement{
		SourceCommitment: sourceCommitment,
		SourceValue:      proof.SourceLink.ClaimedValue,
		Beta:             challenges.Beta,
		ZChallenge:       challenges.ZChallenge,
		OuterNumerator:   numeratorG,
		InnerNumerator:   numeratorL,
		OuterVanishing:   cryptodlinkzg.VanishingPolynomial(gPoints),
		InnerVanishing:   cryptodlinkzg.VanishingPolynomial(lPoints),
		InnerScale:       innerScale,
	}
	batchProof := cryptodlinkzg.DeltaBatchProof{
		PiZ: proof.SourceLink.PiZ,
		PiY: proof.SourceLink.PiY,
		WN:  proof.WN,
	}
	if err := cryptodlinkzg.VerifyDeltaBatch(batchStatement, batchProof, challenges.Delta, srs); err != nil {
		return fmt.Errorf("%w: %v", ErrPrototypeOpening, err)
	}
	return nil
}

type productCheckPoint struct {
	partition []fr.Element
	selector  fr.Element
}

func validatePrototypeInput(input PrototypeInput, challenges PrototypeChallenges, srs *cryptodlinkzg.MonomialSRS) (int, int, error) {
	m := len(input.BoundaryFactors)
	if m < 2 || !isPowerOfTwo(m) {
		return 0, 0, fmt.Errorf("%w: M=%d must be a power of two at least two", ErrInvalidPrototypeInput, m)
	}
	if len(input.LocalResiduals) != m || len(input.BoundaryNumerators) != m || len(input.BoundaryDenominators) != m {
		return 0, 0, fmt.Errorf("%w: all PIOP tables must have length M", ErrInvalidPrototypeInput)
	}
	if srs == nil || len(srs.G1Rect) < m {
		return 0, 0, fmt.Errorf("%w: SRS has fewer than M Y powers", ErrInvalidPrototypeInput)
	}
	t := input.T
	if t < 4 || !isPowerOfTwo(t) || m > t || len(srs.G1Rect[0]) < t || len(srs.G2Z) < 4 {
		return 0, 0, fmt.Errorf("%w: T=%d must be a power of two at least four with M<=T", ErrInvalidPrototypeInput, t)
	}
	for j := 0; j < terminalCircuitClaims; j++ {
		if len(input.Sources[j]) != m {
			return 0, 0, fmt.Errorf("%w: source %d has %d rows, want M=%d", ErrInvalidPrototypeInput, j, len(input.Sources[j]), m)
		}
		for i := range input.Sources[j] {
			if len(input.Sources[j][i]) == 0 || len(input.Sources[j][i]) > t {
				return 0, 0, fmt.Errorf("%w: source %d row %d has invalid width", ErrInvalidPrototypeInput, j, i)
			}
		}
	}
	for i := 0; i < m; i++ {
		var linked fr.Element
		linked.Mul(&input.BoundaryFactors[i], &input.BoundaryDenominators[i])
		if !linked.Equal(&input.BoundaryNumerators[i]) {
			return 0, 0, fmt.Errorf("%w: boundary link %d failed", ErrInvalidPrototypeInput, i)
		}
	}
	if err := validatePrototypeChallenges(challenges, m); err != nil {
		return 0, 0, err
	}
	if _, _, err := translatedQueries(input.CircuitShiftedQueries, bits.Len(uint(t))-1); err != nil {
		return 0, 0, err
	}
	return m, t, nil
}

func validatePrototypeStatement(statement PrototypeStatement, challenges PrototypeChallenges, srs *cryptodlinkzg.MonomialSRS) error {
	if statement.M < 2 || !isPowerOfTwo(statement.M) || statement.T < 4 || !isPowerOfTwo(statement.T) || statement.M > statement.T {
		return fmt.Errorf("%w: invalid M,T rectangle", ErrInvalidPrototypeInput)
	}
	if srs == nil || len(srs.G1Rect) < statement.M || len(srs.G1Rect[0]) < statement.T || len(srs.G2Z) < 4 {
		return fmt.Errorf("%w: SRS does not cover statement rectangle", ErrInvalidPrototypeInput)
	}
	if len(statement.LocalResiduals) != statement.M || len(statement.BoundaryFactors) != statement.M || len(statement.BoundaryNumerators) != statement.M || len(statement.BoundaryDenominators) != statement.M {
		return fmt.Errorf("%w: statement PIOP tables do not have length M", ErrInvalidPrototypeInput)
	}
	if err := validatePrototypeChallenges(challenges, statement.M); err != nil {
		return err
	}
	_, _, err := translatedQueries(statement.CircuitShiftedQueries, bits.Len(uint(statement.T))-1)
	return err
}

func validatePrototypeChallenges(challenges PrototypeChallenges, m int) error {
	logM := bits.Len(uint(m)) - 1
	if len(challenges.Theta) != logM || len(challenges.SumCheck) != logM {
		return fmt.Errorf("%w: theta and SumCheck must have log2(M) coordinates", ErrInvalidPrototypeChallenge)
	}
	orderedNonzeroChallenges := []struct {
		name  string
		value fr.Element
	}{
		{name: "xi", value: challenges.Xi},
		{name: "nu", value: challenges.Nu},
		{name: "z_ch", value: challenges.ZChallenge},
		{name: "beta", value: challenges.Beta},
		{name: "kappa", value: challenges.Kappa},
		{name: "delta", value: challenges.Delta},
	}
	for _, challenge := range orderedNonzeroChallenges {
		if challenge.value.IsZero() {
			return fmt.Errorf("%w: %s is zero", ErrInvalidPrototypeChallenge, challenge.name)
		}
	}
	one := fr.One()
	minusOne := one
	minusOne.Neg(&minusOne)
	if challenges.Beta.Equal(&one) || challenges.Beta.Equal(&minusOne) {
		return fmt.Errorf("%w: beta is +/-1", ErrInvalidPrototypeChallenge)
	}
	betaInverse := inverse(challenges.Beta)
	zInverse := inverse(challenges.ZChallenge)
	if challenges.ZChallenge.Equal(&challenges.Beta) || challenges.ZChallenge.Equal(&betaInverse) || challenges.Beta.Equal(&zInverse) {
		return fmt.Errorf("%w: z_ch collides with beta or beta^{-1}", ErrInvalidPrototypeChallenge)
	}
	return nil
}

func prototypeSumCheckOracles(localResiduals, numerators, denominators, table, theta []fr.Element, zeta fr.Element) ([][]fr.Element, SumCheckComposition, error) {
	m := len(localResiduals)
	if len(numerators) != m || len(denominators) != m || len(table) != 2*m {
		return nil, nil, fmt.Errorf("%w: malformed PIOP table dimensions", ErrInvalidPrototypeInput)
	}
	equality, err := EqualityEvaluationTable(theta)
	if err != nil {
		return nil, nil, err
	}
	parents := cloneElements(table[m:])
	leaves := cloneElements(table[:m])
	evenChildren := make([]fr.Element, m)
	oddChildren := make([]fr.Element, m)
	for i := 0; i < m; i++ {
		evenChildren[i] = table[2*i]
		oddChildren[i] = table[2*i+1]
	}
	oracles := [][]fr.Element{
		equality,
		cloneElements(localResiduals),
		leaves,
		cloneElements(denominators),
		cloneElements(numerators),
		parents,
		evenChildren,
		oddChildren,
	}
	zetaSquared := zeta
	zetaSquared.Square(&zetaSquared)
	composition := func(values []fr.Element) fr.Element {
		var boundary, tree, residual, term fr.Element
		boundary.Mul(&values[2], &values[3]).Sub(&boundary, &values[4])
		tree.Mul(&values[6], &values[7])
		tree.Sub(&values[5], &tree)
		term.Mul(&zeta, &boundary)
		residual.Add(&values[1], &term)
		term.Mul(&zetaSquared, &tree)
		residual.Add(&residual, &term)
		residual.Mul(&residual, &values[0])
		return residual
	}
	return oracles, composition, nil
}

func sumCompositionTable(oracles [][]fr.Element, composition SumCheckComposition) fr.Element {
	values := make([]fr.Element, len(oracles))
	var sum fr.Element
	for row := range oracles[0] {
		for oracle := range oracles {
			values[oracle] = oracles[oracle][row]
		}
		term := composition(values)
		sum.Add(&sum, &term)
	}
	return sum
}

func translatedQueries(queries [terminalCircuitClaims]fr.Element, logT int) ([terminalCircuitClaims][]fr.Element, [terminalCircuitClaims]fr.Element, error) {
	var points [terminalCircuitClaims][]fr.Element
	var scales [terminalCircuitClaims]fr.Element
	for j := 0; j < terminalCircuitClaims; j++ {
		points[j] = make([]fr.Element, logT)
		scales[j].SetOne()
		power := queries[j]
		for k := 0; k < logT; k++ {
			denominator := fr.One()
			denominator.Add(&denominator, &power)
			if denominator.IsZero() {
				return points, scales, fmt.Errorf("%w: circuit query %d is in B_T", ErrInvalidPrototypeChallenge, j)
			}
			points[j][k].Div(&power, &denominator)
			scales[j].Mul(&scales[j], &denominator)
			power.Square(&power)
		}
	}
	return points, scales, nil
}

func productCheckFunctionalPoints(r []fr.Element) [terminalTreeClaims]productCheckPoint {
	var result [terminalTreeClaims]productCheckPoint
	zero := fr.Element{}
	one := fr.One()
	result[0] = productCheckPoint{partition: cloneElements(r), selector: zero}
	result[1] = productCheckPoint{partition: cloneElements(r), selector: one}
	shiftedZero := make([]fr.Element, len(r))
	shiftedOne := make([]fr.Element, len(r))
	shiftedOne[0].SetOne()
	if len(r) > 1 {
		copy(shiftedZero[1:], r[:len(r)-1])
		copy(shiftedOne[1:], r[:len(r)-1])
	}
	selector := r[len(r)-1]
	result[2] = productCheckPoint{partition: shiftedZero, selector: selector}
	result[3] = productCheckPoint{partition: shiftedOne, selector: selector}
	m := 1 << uint(len(r))
	rootPartition := make([]fr.Element, len(r))
	rootIndex := m - 2
	for k := range rootPartition {
		if rootIndex&(1<<uint(k)) != 0 {
			rootPartition[k].SetOne()
		}
	}
	result[4] = productCheckPoint{partition: rootPartition, selector: one}
	return result
}

func evaluateTreeFunctional(t0, t1 []fr.Element, point productCheckPoint) (fr.Element, error) {
	v0, err := cryptodlinkzg.MLEEval(t0, point.partition)
	if err != nil {
		return fr.Element{}, err
	}
	v1, err := cryptodlinkzg.MLEEval(t1, point.partition)
	if err != nil {
		return fr.Element{}, err
	}
	oneMinus := fr.One()
	oneMinus.Sub(&oneMinus, &point.selector)
	var result, term fr.Element
	result.Mul(&oneMinus, &v0)
	term.Mul(&point.selector, &v1)
	result.Add(&result, &term)
	return result, nil
}

func foldRows(rows [][]fr.Element, point []fr.Element, width int) []fr.Element {
	weights := cryptodlinkzg.EqualityWeights(point)
	result := make([]fr.Element, width)
	for i := range rows {
		addScaledPolynomial(result, rows[i], weights[i])
	}
	return result
}

func foldedSourceAndLink(sources [terminalCircuitClaims][][]fr.Element, xi, zChallenge fr.Element, width int) ([][]fr.Element, []fr.Element) {
	m := len(sources[0])
	dXi := make([][]fr.Element, m)
	h := make([]fr.Element, m)
	for i := 0; i < m; i++ {
		dXi[i] = make([]fr.Element, width)
		power := fr.One()
		for j := 0; j < terminalCircuitClaims; j++ {
			addScaledPolynomial(dXi[i], sources[j][i], power)
			value := cryptodlinkzg.Eval(sources[j][i], zChallenge)
			var term fr.Element
			term.Mul(&power, &value)
			h[i].Add(&h[i], &term)
			power.Mul(&power, &xi)
		}
	}
	return dXi, h
}

func buildLaurentWitness(partials [][]fr.Element, h, t0, t1 []fr.Element, queries [terminalCircuitClaims][]fr.Element, scales [terminalCircuitClaims]fr.Element, treePoints [terminalTreeClaims]productCheckPoint, challenges PrototypeChallenges, width int) []fr.Element {
	result := make([]fr.Element, width-1)
	power := fr.One()
	for j := 0; j < terminalCircuitClaims; j++ {
		psi := padPolynomial(cryptodlinkzg.EqualityWeights(queries[j]), width)
		offDiagonal := cryptodlinkzg.FastOffDiag(padPolynomial(partials[j], width), psi)
		var scale fr.Element
		scale.Mul(&power, &scales[j])
		addScaledPolynomial(result, offDiagonal, scale)
		power.Mul(&power, &challenges.Xi)
	}
	psiR := padPolynomial(cryptodlinkzg.EqualityWeights(challenges.SumCheck), width)
	addScaledPolynomial(result, cryptodlinkzg.FastOffDiag(padPolynomial(h, width), psiR), challenges.Nu)
	a, b := treeWeightPolynomials(treePoints, challenges.Xi, width)
	addScaledPolynomial(result, cryptodlinkzg.FastOffDiag(padPolynomial(t0, width), a), fr.One())
	addScaledPolynomial(result, cryptodlinkzg.FastOffDiag(padPolynomial(t1, width), b), fr.One())
	return result
}

func treeWeightPolynomials(points [terminalTreeClaims]productCheckPoint, xi fr.Element, width int) ([]fr.Element, []fr.Element) {
	a := make([]fr.Element, width)
	b := make([]fr.Element, width)
	power := fieldPower(xi, terminalCircuitClaims)
	for c := 0; c < terminalTreeClaims; c++ {
		weights := cryptodlinkzg.EqualityWeights(points[c].partition)
		oneMinus := fr.One()
		oneMinus.Sub(&oneMinus, &points[c].selector)
		var scaleA, scaleB fr.Element
		scaleA.Mul(&power, &oneMinus)
		scaleB.Mul(&power, &points[c].selector)
		addScaledPolynomial(a, weights, scaleA)
		addScaledPolynomial(b, weights, scaleB)
		power.Mul(&power, &xi)
	}
	return a, b
}

func verifyLaurentIdentity(statement PrototypeStatement, proof PrototypeProof, queries [terminalCircuitClaims][]fr.Element, scales [terminalCircuitClaims]fr.Element, treePoints [terminalTreeClaims]productCheckPoint, challenges PrototypeChallenges) error {
	betaInverse := inverse(challenges.Beta)
	var left, term, symmetric fr.Element
	power := fr.One()
	for j := 0; j < terminalCircuitClaims; j++ {
		psi := cryptodlinkzg.EqualityWeights(queries[j])
		symmetric.Mul(&proof.PartialValues[j][1], valuePointer(cryptodlinkzg.Eval(psi, betaInverse)))
		term.Mul(&proof.PartialValues[j][2], valuePointer(cryptodlinkzg.Eval(psi, challenges.Beta)))
		symmetric.Add(&symmetric, &term)
		var scale fr.Element
		scale.Mul(&power, &scales[j])
		term.Mul(&scale, &symmetric)
		left.Add(&left, &term)
		power.Mul(&power, &challenges.Xi)
	}
	psiR := cryptodlinkzg.EqualityWeights(challenges.SumCheck)
	symmetric.Mul(&proof.LinearValues[0][0], valuePointer(cryptodlinkzg.Eval(psiR, betaInverse)))
	term.Mul(&proof.LinearValues[0][1], valuePointer(cryptodlinkzg.Eval(psiR, challenges.Beta)))
	symmetric.Add(&symmetric, &term)
	term.Mul(&challenges.Nu, &symmetric)
	left.Add(&left, &term)
	a, b := treeWeightPolynomials(treePoints, challenges.Xi, statement.T)
	for p, weights := range [][]fr.Element{a, b} {
		symmetric.Mul(&proof.LinearValues[p+1][0], valuePointer(cryptodlinkzg.Eval(weights, betaInverse)))
		term.Mul(&proof.LinearValues[p+1][1], valuePointer(cryptodlinkzg.Eval(weights, challenges.Beta)))
		symmetric.Add(&symmetric, &term)
		left.Add(&left, &symmetric)
	}

	var aLinear fr.Element
	power.SetOne()
	for j := 0; j < terminalCircuitClaims; j++ {
		term.Mul(&power, &statement.CircuitClaims[j])
		aLinear.Add(&aLinear, &term)
		term.Mul(&power, &proof.PartialValues[j][0])
		term.Mul(&term, &challenges.Nu)
		aLinear.Add(&aLinear, &term)
		power.Mul(&power, &challenges.Xi)
	}
	power = fieldPower(challenges.Xi, terminalCircuitClaims)
	for c := 0; c < terminalTreeClaims; c++ {
		term.Mul(&power, &statement.TreeClaims[c])
		aLinear.Add(&aLinear, &term)
		power.Mul(&power, &challenges.Xi)
	}
	var right fr.Element
	right.Double(&aLinear)
	term.Mul(&challenges.Beta, &proof.LinearValues[3][0])
	right.Add(&right, &term)
	term.Mul(&betaInverse, &proof.LinearValues[3][1])
	right.Add(&right, &term)
	if !left.Equal(&right) {
		return fmt.Errorf("%w: Laurent scalar identity failed", ErrPrototypeOpening)
	}
	return nil
}

func foldG1(commitments []bn254.G1Affine, challenge fr.Element) bn254.G1Affine {
	var result bn254.G1Jac
	power := fr.One()
	for i := range commitments {
		scaled := scaleG1Prototype(commitments[i], power)
		var scaledJac bn254.G1Jac
		scaledJac.FromAffine(&scaled)
		result.AddAssign(&scaledJac)
		power.Mul(&power, &challenge)
	}
	var affine bn254.G1Affine
	affine.FromJacobian(&result)
	return affine
}

func scaleG1Prototype(point bn254.G1Affine, scalar fr.Element) bn254.G1Affine {
	var scalarBig big.Int
	scalar.ToBigIntRegular(&scalarBig)
	var result bn254.G1Affine
	result.ScalarMultiplication(&point, &scalarBig)
	return result
}

func addScaledPolynomial(destination, source []fr.Element, scale fr.Element) {
	limit := len(source)
	if len(destination) < limit {
		limit = len(destination)
	}
	for i := 0; i < limit; i++ {
		var term fr.Element
		term.Mul(&source[i], &scale)
		destination[i].Add(&destination[i], &term)
	}
}

func padPolynomial(polynomial []fr.Element, width int) []fr.Element {
	result := make([]fr.Element, width)
	copy(result, polynomial)
	return result
}

func cloneElements(elements []fr.Element) []fr.Element {
	return append([]fr.Element(nil), elements...)
}

func inverse(value fr.Element) fr.Element {
	var result fr.Element
	result.Inverse(&value)
	return result
}

func fieldPower(base fr.Element, exponent int) fr.Element {
	result := fr.One()
	for i := 0; i < exponent; i++ {
		result.Mul(&result, &base)
	}
	return result
}

func valuePointer(value fr.Element) *fr.Element {
	return &value
}
