package dlinkzg

// This file implements the preserved degree-aware hybrid terminal compiler.
// Circuit partial evaluations stay in the semantic U direction, while the
// partition/ProductCheck functional batch stays at honest degree < M. The
// two opening equations are combined only by the final verifier challenge.

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

const HybridOpeningInstanceDigestDomain = "DLinKZG/hybrid-opening-instance/v1"

type HybridOpeningProof struct {
	U0 HybridU0Message
	U1 HybridU1Message
	U2 HybridU2Message
	U3 HybridU3Message
}

type hybridOpeningPrepared struct {
	m       int
	t       int
	weights []fr.Element
	tree    [dlinkzgOpeningTreeClaims]productCheckPoint
}

type hybridOpeningLocalState struct {
	g                    [dlinkzgOpeningCircuitClaims][]fr.Element
	a                    [dlinkzgOpeningCircuitClaims]fr.Element
	semanticValues       [dlinkzgOpeningCircuitClaims][dlinkzgOpeningCircuitClaims]fr.Element
	pGamma               []fr.Element
	uHat                 fr.Element
	h                    []fr.Element
	t0                   []fr.Element
	t1                   []fr.Element
	sFun                 []fr.Element
	laurentAtBeta        [4]fr.Element
	laurentAtBetaInverse [4]fr.Element
}

func HybridOpeningInstanceDigest(instance DLinkZGOpeningInstance) TranscriptDigest {
	encoded := make([]byte, 0, 1024)
	encoded = appendLengthPrefixed(encoded, []byte(HybridOpeningInstanceDigestDomain))
	encoded = appendUint64(encoded, uint64(instance.LocalDomainSize))
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		encoded = appendG1(encoded, &instance.SourceCommitments[claim])
		encoded = appendField(encoded, &instance.CircuitClaims[claim])
		encoded = appendField(encoded, &instance.SemanticQueryPoints[claim])
	}
	for commitment := range instance.TreeCommitments {
		encoded = appendG1(encoded, &instance.TreeCommitments[commitment])
	}
	for claim := range instance.TreeClaims {
		encoded = appendField(encoded, &instance.TreeClaims[claim])
	}
	encoded = appendUint64(encoded, uint64(len(instance.PartitionPoint)))
	for coordinate := range instance.PartitionPoint {
		encoded = appendField(encoded, &instance.PartitionPoint[coordinate])
	}
	return NewTranscriptDigest(encoded)
}

func BindHybridOpeningTranscriptContext(instance DLinkZGOpeningInstance, context TranscriptContext) TranscriptContext {
	context.PublicStatementDigest = HybridOpeningInstanceDigest(instance)
	return context
}

func ProveHybridOpening(instance DLinkZGOpeningInstance, input DLinkZGOpeningProverInput) (HybridOpeningProof, error) {
	var proof HybridOpeningProof
	prepared, err := prepareHybridOpeningInstance(instance)
	if err != nil {
		return proof, err
	}
	if err := validateDLinkZGOpeningInput(input, dlinkzgOpeningPrepared{m: prepared.m, t: prepared.t}); err != nil {
		return proof, err
	}
	prepared.weights = cryptodlinkzg.EqualityWeights(instance.PartitionPoint)
	transcript, err := NewHybridOpeningTranscript(instance.TranscriptContext)
	if err != nil {
		return proof, err
	}

	locals := make([]hybridOpeningLocalState, prepared.m)
	var partialCommitments [dlinkzgOpeningCircuitClaims]bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			locals[rank].g[claim] = dlinkzgOpeningScalePolynomial(
				input.Sources[claim][rank],
				prepared.weights[rank],
			)
			commitment, commitErr := input.PartySRS[rank].CommitU(locals[rank].g[claim])
			if commitErr != nil {
				return HybridOpeningProof{}, fmt.Errorf("%w: hybrid U0 rank %d claim %d: %v", ErrInvalidDLinkZGOpeningInput, rank, claim, commitErr)
			}
			dlinkzgOpeningAddG1(&partialCommitments[claim], commitment)
		}
	}
	for claim := range proof.U0.PartialCommitments {
		proof.U0.PartialCommitments[claim].FromJacobian(&partialCommitments[claim])
	}
	if err := transcript.AppendU0(proof.U0); err != nil {
		return HybridOpeningProof{}, err
	}

	gammaOut, err := transcript.DeriveGamma()
	if err != nil {
		return HybridOpeningProof{}, err
	}
	alphaOut, err := transcript.DeriveAlphaChallenge(instance.SemanticQueryPoints)
	if err != nil {
		return HybridOpeningProof{}, err
	}
	gamma, alphaChallenge := gammaOut.Value, alphaOut.Value

	aWeights, bWeights := treeWeightPolynomials(prepared.tree, gamma, prepared.m)
	uHatCoefficients := make([]fr.Element, prepared.m)
	var laurentCommitment bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		locals[rank].pGamma = make([]fr.Element, prepared.t)
		gammaPower := fr.One()
		var weightedGammaAtChallenge fr.Element
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			locals[rank].a[claim] = cryptodlinkzg.Eval(locals[rank].g[claim], alphaChallenge)
			proof.U1.PartialAtChallenge[claim].Add(&proof.U1.PartialAtChallenge[claim], &locals[rank].a[claim])
			dlinkzgOpeningAddScaled(locals[rank].pGamma, input.Sources[claim][rank], gammaPower)
			var term fr.Element
			term.Mul(&gammaPower, &locals[rank].a[claim])
			weightedGammaAtChallenge.Add(&weightedGammaAtChallenge, &term)
			gammaPower.Mul(&gammaPower, &gamma)
		}
		if prepared.weights[rank].IsZero() {
			return HybridOpeningProof{}, fmt.Errorf("%w: zero partition equality weight at rank %d", ErrInvalidDLinkZGOpeningInput, rank)
		}
		locals[rank].uHat.Div(&weightedGammaAtChallenge, &prepared.weights[rank])
		uHatCoefficients[rank] = locals[rank].uHat
		locals[rank].h = dlinkzgOpeningMonomial(locals[rank].uHat, rank)
		locals[rank].t0 = dlinkzgOpeningMonomial(input.T0[rank], rank)
		locals[rank].t1 = dlinkzgOpeningMonomial(input.T1[rank], rank)
		locals[rank].sFun = cryptodlinkzg.BuildLocalLaurent(cryptodlinkzg.LocalLaurentInput{
			HXi:  locals[rank].h,
			PsiR: prepared.weights,
			Nu:   fr.One(),
			T0:   locals[rank].t0,
			T1:   locals[rank].t1,
			AXi:  aWeights,
			BXi:  bWeights,
		})
		commitment, commitErr := input.PartySRS[rank].CommitU(locals[rank].sFun)
		if commitErr != nil {
			return HybridOpeningProof{}, fmt.Errorf("%w: hybrid U1 Laurent rank %d: %v", ErrInvalidDLinkZGOpeningInput, rank, commitErr)
		}
		dlinkzgOpeningAddG1(&laurentCommitment, commitment)
	}
	proof.U1.FunctionalCommitment, err = input.CoordinatorSRS.CommitU(uHatCoefficients)
	if err != nil {
		return HybridOpeningProof{}, fmt.Errorf("%w: hybrid functional commitment: %v", ErrInvalidDLinkZGOpeningInput, err)
	}
	proof.U1.LaurentCommitment.FromJacobian(&laurentCommitment)
	if err := transcript.AppendU1(proof.U1); err != nil {
		return HybridOpeningProof{}, err
	}

	betaOut, err := transcript.DeriveBeta()
	if err != nil {
		return HybridOpeningProof{}, err
	}
	beta := betaOut.Value
	betaInverse := dlinkzgOpeningInverse(beta)
	for rank := 0; rank < prepared.m; rank++ {
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			cross := 0
			for semantic := 0; semantic < dlinkzgOpeningCircuitClaims; semantic++ {
				value := cryptodlinkzg.Eval(locals[rank].g[claim], instance.SemanticQueryPoints[semantic])
				locals[rank].semanticValues[claim][semantic] = value
				if semantic == claim {
					continue
				}
				proof.U2.CircuitCrossValues[claim][cross].Add(&proof.U2.CircuitCrossValues[claim][cross], &value)
				cross++
			}
		}
		polynomials := [4][]fr.Element{locals[rank].h, locals[rank].t0, locals[rank].t1, locals[rank].sFun}
		for polynomial := range polynomials {
			locals[rank].laurentAtBeta[polynomial] = cryptodlinkzg.Eval(polynomials[polynomial], beta)
			locals[rank].laurentAtBetaInverse[polynomial] = cryptodlinkzg.Eval(polynomials[polynomial], betaInverse)
			proof.U2.LaurentAtBeta[polynomial].Add(&proof.U2.LaurentAtBeta[polynomial], &locals[rank].laurentAtBeta[polynomial])
			proof.U2.LaurentAtBetaInverse[polynomial].Add(&proof.U2.LaurentAtBetaInverse[polynomial], &locals[rank].laurentAtBetaInverse[polynomial])
		}
	}
	if err := verifyHybridOpeningScalarIdentity(instance, proof.U1, proof.U2, prepared, gamma, beta); err != nil {
		return HybridOpeningProof{}, err
	}
	if err := transcript.AppendU2(proof.U2); err != nil {
		return HybridOpeningProof{}, err
	}

	kappaOut, err := transcript.DeriveKappa()
	if err != nil {
		return HybridOpeningProof{}, err
	}
	kappa := kappaOut.Value
	circuitPoints := hybridCircuitPoints(alphaChallenge, instance.SemanticQueryPoints)
	laurentPoints := []fr.Element{beta, betaInverse}
	var wCirc, wLaur, piU bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		circuitInputs := make([]cryptodlinkzg.SameSetInput, dlinkzgOpeningCircuitClaims)
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			values := make([]fr.Element, 0, len(circuitPoints))
			values = append(values, locals[rank].a[claim])
			values = append(values, locals[rank].semanticValues[claim][:]...)
			circuitInputs[claim] = cryptodlinkzg.SameSetInput{Polynomial: locals[rank].g[claim], ClaimedValues: values}
		}
		circuitBatch, batchErr := cryptodlinkzg.BuildSameSetQuotient(circuitInputs, circuitPoints, kappa)
		if batchErr != nil {
			return HybridOpeningProof{}, fmt.Errorf("%w: hybrid circuit batch rank %d: %v", ErrDLinkZGOpeningRejected, rank, batchErr)
		}
		wCircShare, commitErr := input.PartySRS[rank].CommitU(circuitBatch.Quotient)
		if commitErr != nil {
			return HybridOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&wCirc, wCircShare)

		laurentPolynomials := [4][]fr.Element{locals[rank].h, locals[rank].t0, locals[rank].t1, locals[rank].sFun}
		laurentInputs := make([]cryptodlinkzg.SameSetInput, len(laurentPolynomials))
		for polynomial := range laurentPolynomials {
			laurentInputs[polynomial] = cryptodlinkzg.SameSetInput{
				Polynomial: laurentPolynomials[polynomial],
				ClaimedValues: []fr.Element{
					locals[rank].laurentAtBeta[polynomial],
					locals[rank].laurentAtBetaInverse[polynomial],
				},
			}
		}
		laurentBatch, batchErr := cryptodlinkzg.BuildSameSetQuotient(laurentInputs, laurentPoints, kappa)
		if batchErr != nil {
			return HybridOpeningProof{}, fmt.Errorf("%w: hybrid Laurent batch rank %d: %v", ErrDLinkZGOpeningRejected, rank, batchErr)
		}
		wLaurShare, commitErr := input.PartySRS[rank].CommitU(laurentBatch.Quotient)
		if commitErr != nil {
			return HybridOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&wLaur, wLaurShare)

		qU, remainder := cryptodlinkzg.SyntheticDivision(locals[rank].pGamma, alphaChallenge)
		if !remainder.Equal(&locals[rank].uHat) {
			return HybridOpeningProof{}, fmt.Errorf("%w: hybrid source remainder rank %d", ErrDLinkZGOpeningRejected, rank)
		}
		piUShare, commitErr := input.PartySRS[rank].CommitSemantic(qU)
		if commitErr != nil {
			return HybridOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&piU, piUShare)
	}
	proof.U3.WCirc.FromJacobian(&wCirc)
	proof.U3.WLaur.FromJacobian(&wLaur)
	proof.U3.PiU.FromJacobian(&piU)
	qV, sourceValue := cryptodlinkzg.SyntheticDivision(uHatCoefficients, beta)
	if !sourceValue.Equal(&proof.U2.LaurentAtBeta[0]) {
		return HybridOpeningProof{}, fmt.Errorf("%w: hybrid source-link value", ErrDLinkZGOpeningRejected)
	}
	proof.U3.PiV, err = input.CoordinatorSRS.CommitY(qV)
	if err != nil {
		return HybridOpeningProof{}, err
	}
	if err := transcript.AppendU3(proof.U3); err != nil {
		return HybridOpeningProof{}, err
	}
	if _, err := transcript.DeriveDelta(); err != nil {
		return HybridOpeningProof{}, err
	}
	return proof, nil
}

func VerifyHybridOpening(instance DLinkZGOpeningInstance, proof HybridOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS) error {
	prepared, err := prepareHybridOpeningInstance(instance)
	if err != nil {
		return err
	}
	if verifierSRS == nil || verifierSRS.Validate() != nil {
		return fmt.Errorf("%w: hybrid verifier SRS", ErrInvalidDLinkZGOpeningInstance)
	}
	prepared.weights = cryptodlinkzg.EqualityWeights(instance.PartitionPoint)
	transcript, err := NewHybridOpeningTranscript(instance.TranscriptContext)
	if err != nil {
		return err
	}
	if err := transcript.AppendU0(proof.U0); err != nil {
		return err
	}
	gammaOut, err := transcript.DeriveGamma()
	if err != nil {
		return err
	}
	alphaOut, err := transcript.DeriveAlphaChallenge(instance.SemanticQueryPoints)
	if err != nil {
		return err
	}
	if err := transcript.AppendU1(proof.U1); err != nil {
		return err
	}
	betaOut, err := transcript.DeriveBeta()
	if err != nil {
		return err
	}
	if err := verifyHybridOpeningScalarIdentity(instance, proof.U1, proof.U2, prepared, gammaOut.Value, betaOut.Value); err != nil {
		return err
	}
	if err := transcript.AppendU2(proof.U2); err != nil {
		return err
	}
	kappaOut, err := transcript.DeriveKappa()
	if err != nil {
		return err
	}
	if err := transcript.AppendU3(proof.U3); err != nil {
		return err
	}
	deltaOut, err := transcript.DeriveDelta()
	if err != nil {
		return err
	}
	return verifyHybridOpeningFinal(instance, proof, verifierSRS, alphaOut.Value, betaOut.Value, gammaOut.Value, kappaOut.Value, deltaOut.Value)
}

func prepareHybridOpeningInstance(instance DLinkZGOpeningInstance) (hybridOpeningPrepared, error) {
	var prepared hybridOpeningPrepared
	if instance.LocalDomainSize < 4 || instance.LocalDomainSize > FastLocalPIOPMaxDomainSize || !isPowerOfTwo(instance.LocalDomainSize) {
		return prepared, fmt.Errorf("%w: invalid hybrid T=%d", ErrInvalidDLinkZGOpeningInstance, instance.LocalDomainSize)
	}
	if len(instance.PartitionPoint) == 0 || len(instance.PartitionPoint) >= 63 {
		return prepared, fmt.Errorf("%w: invalid hybrid partition point", ErrInvalidDLinkZGOpeningInstance)
	}
	m64 := uint64(1) << uint(len(instance.PartitionPoint))
	if m64 > uint64(instance.LocalDomainSize) || m64 > uint64(^uint(0)>>1) {
		return prepared, fmt.Errorf("%w: hybrid M exceeds T", ErrInvalidDLinkZGOpeningInstance)
	}
	for i := range instance.SourceCommitments {
		if err := validateTranscriptG1(&instance.SourceCommitments[i]); err != nil {
			return prepared, fmt.Errorf("%w: hybrid source commitment %d", ErrInvalidDLinkZGOpeningInstance, i)
		}
	}
	for i := range instance.TreeCommitments {
		if err := validateTranscriptG1(&instance.TreeCommitments[i]); err != nil {
			return prepared, fmt.Errorf("%w: hybrid tree commitment %d", ErrInvalidDLinkZGOpeningInstance, i)
		}
	}
	if err := validateTranscriptContext(instance.TranscriptContext); err != nil {
		return prepared, err
	}
	if instance.TranscriptContext.PublicStatementDigest != HybridOpeningInstanceDigest(instance) {
		return prepared, fmt.Errorf("%w: hybrid transcript statement digest mismatch", ErrInvalidDLinkZGOpeningInstance)
	}
	for i := 0; i < dlinkzgOpeningCircuitClaims; i++ {
		for j := i + 1; j < dlinkzgOpeningCircuitClaims; j++ {
			if instance.SemanticQueryPoints[i].Equal(&instance.SemanticQueryPoints[j]) {
				return prepared, fmt.Errorf("%w: duplicate hybrid semantic point", ErrInvalidDLinkZGOpeningInstance)
			}
		}
	}
	prepared.m = int(m64)
	prepared.t = instance.LocalDomainSize
	prepared.tree = productCheckFunctionalPoints(instance.PartitionPoint)
	return prepared, nil
}

func hybridCircuitPoints(alphaChallenge fr.Element, semantic [3]fr.Element) []fr.Element {
	return []fr.Element{alphaChallenge, semantic[0], semantic[1], semantic[2]}
}

func verifyHybridOpeningScalarIdentity(instance DLinkZGOpeningInstance, u1 HybridU1Message, u2 HybridU2Message, prepared hybridOpeningPrepared, gamma, beta fr.Element) error {
	betaInverse := dlinkzgOpeningInverse(beta)
	psiRAtBeta := dlinkzgOpeningPsiEval(instance.PartitionPoint, beta)
	psiRAtBetaInverse := dlinkzgOpeningPsiEval(instance.PartitionPoint, betaInverse)
	var left, term fr.Element
	left.Mul(&u2.LaurentAtBeta[0], &psiRAtBetaInverse)
	term.Mul(&u2.LaurentAtBetaInverse[0], &psiRAtBeta)
	left.Add(&left, &term)
	aAtBeta, bAtBeta := dlinkzgOpeningTreeWeightEvals(prepared.tree, gamma, beta)
	aAtBetaInverse, bAtBetaInverse := dlinkzgOpeningTreeWeightEvals(prepared.tree, gamma, betaInverse)
	weights := [2][2]fr.Element{{aAtBeta, aAtBetaInverse}, {bAtBeta, bAtBetaInverse}}
	for index := 0; index < 2; index++ {
		term.Mul(&u2.LaurentAtBeta[index+1], &weights[index][1])
		left.Add(&left, &term)
		term.Mul(&u2.LaurentAtBetaInverse[index+1], &weights[index][0])
		left.Add(&left, &term)
	}
	target := hybridOpeningLinearTarget(instance, u1.PartialAtChallenge, gamma)
	var right fr.Element
	right.Double(&target)
	term.Mul(&beta, &u2.LaurentAtBeta[3])
	right.Add(&right, &term)
	term.Mul(&betaInverse, &u2.LaurentAtBetaInverse[3])
	right.Add(&right, &term)
	if !left.Equal(&right) {
		return ErrDLinkZGOpeningScalarIdentity
	}
	return nil
}

func hybridOpeningLinearTarget(instance DLinkZGOpeningInstance, partials [3]fr.Element, gamma fr.Element) fr.Element {
	var result, term fr.Element
	power := fr.One()
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		term.Mul(&power, &partials[claim])
		result.Add(&result, &term)
		power.Mul(&power, &gamma)
	}
	for claim := 0; claim < dlinkzgOpeningTreeClaims; claim++ {
		term.Mul(&power, &instance.TreeClaims[claim])
		result.Add(&result, &term)
		power.Mul(&power, &gamma)
	}
	return result
}

func verifyHybridOpeningFinal(instance DLinkZGOpeningInstance, proof HybridOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS, alphaChallenge, beta, gamma, kappa, delta fr.Element) error {
	circuitPoints := hybridCircuitPoints(alphaChallenge, instance.SemanticQueryPoints)
	circuitInterpolants := make([][]fr.Element, dlinkzgOpeningCircuitClaims)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		values := make([]fr.Element, 0, len(circuitPoints))
		values = append(values, proof.U1.PartialAtChallenge[claim])
		cross := 0
		for semantic := 0; semantic < dlinkzgOpeningCircuitClaims; semantic++ {
			if semantic == claim {
				values = append(values, instance.CircuitClaims[claim])
				continue
			}
			values = append(values, proof.U2.CircuitCrossValues[claim][cross])
			cross++
		}
		interpolant, err := cryptodlinkzg.Interpolate(circuitPoints, values)
		if err != nil {
			return fmt.Errorf("%w: hybrid circuit interpolant: %v", ErrDLinkZGOpeningRejected, err)
		}
		circuitInterpolants[claim] = interpolant
	}
	circuitNumerator, err := verifierSRS.FoldSameSetCommitmentsU(proof.U0.PartialCommitments[:], circuitInterpolants, kappa)
	if err != nil {
		return fmt.Errorf("%w: hybrid circuit numerator: %v", ErrDLinkZGOpeningRejected, err)
	}

	betaInverse := dlinkzgOpeningInverse(beta)
	laurentPoints := []fr.Element{beta, betaInverse}
	laurentInterpolants := make([][]fr.Element, 4)
	for polynomial := 0; polynomial < 4; polynomial++ {
		interpolant, interpolateErr := cryptodlinkzg.Interpolate(
			laurentPoints,
			[]fr.Element{proof.U2.LaurentAtBeta[polynomial], proof.U2.LaurentAtBetaInverse[polynomial]},
		)
		if interpolateErr != nil {
			return fmt.Errorf("%w: hybrid Laurent interpolant: %v", ErrDLinkZGOpeningRejected, interpolateErr)
		}
		laurentInterpolants[polynomial] = interpolant
	}
	laurentCommitments := []bn254.G1Affine{
		proof.U1.FunctionalCommitment,
		instance.TreeCommitments[0],
		instance.TreeCommitments[1],
		proof.U1.LaurentCommitment,
	}
	laurentNumerator, err := verifierSRS.FoldSameSetCommitmentsU(laurentCommitments, laurentInterpolants, kappa)
	if err != nil {
		return fmt.Errorf("%w: hybrid Laurent numerator: %v", ErrDLinkZGOpeningRejected, err)
	}

	statement := cryptodlinkzg.HybridBatchStatement{
		SourceCommitment: foldG1(instance.SourceCommitments[:], gamma),
		SourceValue:      proof.U2.LaurentAtBeta[0],
		Beta:             beta,
		AlphaChallenge:   alphaChallenge,
		CircuitNumerator: circuitNumerator,
		LaurentNumerator: laurentNumerator,
		CircuitVanishing: cryptodlinkzg.VanishingPolynomial(circuitPoints),
		LaurentVanishing: cryptodlinkzg.VanishingPolynomial(laurentPoints),
	}
	batchProof := cryptodlinkzg.HybridBatchProof{
		PiU:   proof.U3.PiU,
		PiV:   proof.U3.PiV,
		WCirc: proof.U3.WCirc,
		WLaur: proof.U3.WLaur,
	}
	if err := verifierSRS.VerifyHybridBatch(statement, batchProof, delta); err != nil {
		return fmt.Errorf("%w: %v", ErrDLinkZGOpeningRejected, err)
	}
	return nil
}
