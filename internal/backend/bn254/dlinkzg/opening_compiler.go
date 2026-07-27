package dlinkzg

// This file is the centralized executable reference for Protocol 2.  It
// deliberately keeps the party decomposition: every length-T commitment and
// quotient is produced through one PartyRowSRS, while the coordinator uses
// only its O(M) column view and verification uses only VerifierSRS.

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

const (
	dlinkzgOpeningCircuitClaims = 3
	dlinkzgOpeningTreeClaims    = 5

	// DLinkZGOpeningInstanceDigestDomain is the canonical domain separator for
	// the standalone Protocol 2 public instance.
	DLinkZGOpeningInstanceDigestDomain = "DLinKZG/opening-instance/v1"
)

var (
	// ErrInvalidDLinkZGOpeningInstance reports a malformed public Protocol 2
	// statement.  It is distinct from a rejecting, well-formed opening.
	ErrInvalidDLinkZGOpeningInstance = errors.New("dlinkzg: invalid opening instance")
	// ErrInvalidDLinkZGOpeningInput reports malformed rank-local prover state or
	// an SRS role view with the wrong rank, party count, or degree bound.
	ErrInvalidDLinkZGOpeningInput = errors.New("dlinkzg: invalid opening prover input")
	// ErrDLinkZGOpeningScalarIdentity reports failure of the public Laurent
	// identity at beta.
	ErrDLinkZGOpeningScalarIdentity = errors.New("dlinkzg: opening Laurent scalar identity failed")
	// ErrDLinkZGOpeningRejected wraps a failed final five-pairing product.
	ErrDLinkZGOpeningRejected = errors.New("dlinkzg: opening rejected")
)

// DLinkZGOpeningInstance is the complete public input to Protocol 2.  The
// semantic query order is (alpha, omega*alpha, x_star).  LocalDomainSize is T;
// len(PartitionPoint)=log2(M) fixes the manifest-ordered party count.
//
// SourceCommitments bind the unshifted semantic-X rows p_{j,i}(X).  The two
// tree commitments bind t_0 and t_1 in the shared Y^0 Z row.  Neither source
// polynomials nor M-length coefficient vectors occur in the public proof.
type DLinkZGOpeningInstance struct {
	LocalDomainSize int

	SourceCommitments [dlinkzgOpeningCircuitClaims]bn254.G1Affine
	CircuitClaims     [dlinkzgOpeningCircuitClaims]fr.Element
	TreeCommitments   [2]bn254.G1Affine
	TreeClaims        [dlinkzgOpeningTreeClaims]fr.Element

	SemanticQueryPoints [dlinkzgOpeningCircuitClaims]fr.Element
	Shift               fr.Element
	PartitionPoint      []fr.Element
	TranscriptContext   TranscriptContext
}

// DLinkZGOpeningProverInput is manifest ordered: Sources[j][i] is party i's
// untranslated semantic-X polynomial p_{j,i}.  PartySRS[i] must be exactly the
// shift-aware native/semantic role view for rank i.  T0 and T1 are the two
// length-M ProductCheck coefficient arrays.
type DLinkZGOpeningProverInput struct {
	Sources [dlinkzgOpeningCircuitClaims][][]fr.Element
	T0      []fr.Element
	T1      []fr.Element

	PartySRS       []*cryptodlinkzg.PartyRowSRS
	CoordinatorSRS *cryptodlinkzg.CoordinatorSRS
	VerifierSRS    *cryptodlinkzg.VerifierSRS
}

// DLinkZGOpeningProof is exactly the public U0--U3 suffix.  Fiat--Shamir
// challenges, rejection counters, interpolants, and numerator commitments are
// deterministically replayed and are not duplicated here.
type DLinkZGOpeningProof struct {
	U0 U0Message
	U1 U1Message
	U2 U2Message
	U3 U3Message
}

// DLinkZGOpeningInstanceDigest hashes the complete ordered public PCS
// instance, excluding TranscriptContext itself to avoid self-reference.  The
// order is T; the three (source commitment, claim, semantic query) tuples; the
// two tree commitments; the five tree claims; sigma; and r.
//
// In Protocol 3 the global statement remains transitively bound by
// TranscriptContext.PriorTranscriptDigest, which is the completed outer
// transcript.  PublicStatementDigest is reserved for this immediate opening
// instance so a standalone verifier cannot reuse an honest transcript with
// xi- or kappa-cancelling changes to the advertised claims or commitments.
func DLinkZGOpeningInstanceDigest(instance DLinkZGOpeningInstance) TranscriptDigest {
	encoded := make([]byte, 0, 1024)
	encoded = appendLengthPrefixed(encoded, []byte(DLinkZGOpeningInstanceDigestDomain))
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
	encoded = appendField(encoded, &instance.Shift)
	encoded = appendUint64(encoded, uint64(len(instance.PartitionPoint)))
	for coordinate := range instance.PartitionPoint {
		encoded = appendField(encoded, &instance.PartitionPoint[coordinate])
	}
	return NewTranscriptDigest(encoded)
}

// BindDLinkZGOpeningTranscriptContext returns a copy of context whose
// PublicStatementDigest is the canonical immediate opening-instance digest.
// It preserves PriorTranscriptDigest, so a composed Protocol 3 transcript
// continues to bind the verified global statement and outer PIOP prefix.
func BindDLinkZGOpeningTranscriptContext(instance DLinkZGOpeningInstance, context TranscriptContext) TranscriptContext {
	context.PublicStatementDigest = DLinkZGOpeningInstanceDigest(instance)
	return context
}

// ProveDLinkZGOpening executes the honest centralized reference for the four
// distributed phases.  It never materializes a monolithic rectangular SRS and
// never receives either trapdoor.
func ProveDLinkZGOpening(instance DLinkZGOpeningInstance, input DLinkZGOpeningProverInput) (DLinkZGOpeningProof, error) {
	return proveDLinkZGOpeningWithTranscript(instance, input, newDLinkZGOpeningTranscript)
}

// VerifyDLinkZGOpening replays Protocol 2 from its public instance and the
// constant-size verifier SRS, checks the scalar identity, and performs the
// final five-input pairing product.
func VerifyDLinkZGOpening(instance DLinkZGOpeningInstance, proof DLinkZGOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS) error {
	return verifyDLinkZGOpeningWithTranscript(instance, proof, verifierSRS, newDLinkZGOpeningTranscript)
}

type dlinkzgOpeningTranscript interface {
	AppendU0(U0Message) error
	DeriveXi() (ChallengeOut, error)
	DeriveNu() (ChallengeOut, error)
	DeriveZChallenge() (ChallengeOut, error)
	AppendU1(U1Message) error
	DeriveBeta() (ChallengeOut, error)
	AppendU2(U2Message) error
	DeriveKappa() (ChallengeOut, error)
	AppendU3(U3Message) error
	DeriveDelta() (ChallengeOut, error)
}

type dlinkzgOpeningTranscriptFactory func(TranscriptContext) (dlinkzgOpeningTranscript, error)

func newDLinkZGOpeningTranscript(context TranscriptContext) (dlinkzgOpeningTranscript, error) {
	return NewOpeningTranscript(context)
}

type dlinkzgOpeningPrepared struct {
	m       int
	t       int
	weights []fr.Element
	queries [dlinkzgOpeningCircuitClaims][]fr.Element
	scales  [dlinkzgOpeningCircuitClaims]fr.Element
	psiQ    [dlinkzgOpeningCircuitClaims][]fr.Element
	psiR    []fr.Element
	tree    [dlinkzgOpeningTreeClaims]productCheckPoint
}

type dlinkzgOpeningLocalState struct {
	shifted        [dlinkzgOpeningCircuitClaims][]fr.Element
	g              [dlinkzgOpeningCircuitClaims][]fr.Element
	gAtBeta        [dlinkzgOpeningCircuitClaims]fr.Element
	gAtBetaInverse [dlinkzgOpeningCircuitClaims]fr.Element
	d              [dlinkzgOpeningCircuitClaims]fr.Element
	dLink          fr.Element
	h              []fr.Element
	t0             []fr.Element
	t1             []fr.Element
	lAtBeta        [3]fr.Element
	lAtBetaInverse [3]fr.Element
	laurent        []fr.Element
	input          cryptodlinkzg.LocalLaurentInput
}

func proveDLinkZGOpeningWithTranscript(instance DLinkZGOpeningInstance, input DLinkZGOpeningProverInput, transcriptFactory dlinkzgOpeningTranscriptFactory) (DLinkZGOpeningProof, error) {
	var proof DLinkZGOpeningProof
	prepared, err := prepareDLinkZGOpeningInstance(instance)
	if err != nil {
		return proof, err
	}
	if err := validateDLinkZGOpeningInput(input, prepared); err != nil {
		return proof, err
	}
	prepared.weights = cryptodlinkzg.EqualityWeights(instance.PartitionPoint)
	prepared.psiR = append([]fr.Element(nil), prepared.weights...)
	for claim := range prepared.psiQ {
		prepared.psiQ[claim] = cryptodlinkzg.EqualityWeights(prepared.queries[claim])
	}
	if transcriptFactory == nil {
		return proof, fmt.Errorf("%w: nil transcript factory", ErrInvalidDLinkZGOpeningInput)
	}
	transcript, err := transcriptFactory(instance.TranscriptContext)
	if err != nil {
		return proof, err
	}
	if transcript == nil {
		return proof, fmt.Errorf("%w: transcript factory returned nil", ErrInvalidDLinkZGOpeningInput)
	}

	locals := make([]dlinkzgOpeningLocalState, prepared.m)
	var u0Sums [dlinkzgOpeningCircuitClaims]bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		sourceBatch := make([][]fr.Element, dlinkzgOpeningCircuitClaims)
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			sourceBatch[claim] = input.Sources[claim][rank]
		}
		shiftedBatch := cryptodlinkzg.FastTaylorShiftBatch(sourceBatch, instance.Shift)
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			locals[rank].shifted[claim] = shiftedBatch[claim]
			locals[rank].g[claim] = dlinkzgOpeningScalePolynomial(locals[rank].shifted[claim], prepared.weights[rank])
			commitment, commitErr := input.PartySRS[rank].CommitZ(locals[rank].g[claim])
			if commitErr != nil {
				return DLinkZGOpeningProof{}, fmt.Errorf("%w: U0 rank %d source %d: %v", ErrInvalidDLinkZGOpeningInput, rank, claim, commitErr)
			}
			dlinkzgOpeningAddG1(&u0Sums[claim], commitment)
		}
	}
	for claim := range proof.U0.PartialCommitments {
		proof.U0.PartialCommitments[claim].FromJacobian(&u0Sums[claim])
	}
	if err := transcript.AppendU0(proof.U0); err != nil {
		return DLinkZGOpeningProof{}, err
	}

	xiOut, err := transcript.DeriveXi()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	if xiOut.Value.IsZero() {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: xi is zero", ErrInvalidDLinkZGOpeningInput)
	}
	nuOut, err := transcript.DeriveNu()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	if nuOut.Value.IsZero() {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: nu is zero", ErrInvalidDLinkZGOpeningInput)
	}
	zOut, err := transcript.DeriveZChallenge()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	xi, nu, zChallenge := xiOut.Value, nuOut.Value, zOut.Value

	aWeights, bWeights := treeWeightPolynomials(prepared.tree, xi, prepared.t)
	hCoefficients := make([]fr.Element, prepared.m)
	var laurentCommitment bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		xiPower := fr.One()
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			locals[rank].d[claim] = cryptodlinkzg.Eval(locals[rank].shifted[claim], zChallenge)
			var term fr.Element
			term.Mul(&xiPower, &locals[rank].d[claim])
			locals[rank].dLink.Add(&locals[rank].dLink, &term)
			term.Mul(&prepared.weights[rank], &locals[rank].d[claim])
			proof.U1.LinkEvaluations[claim].Add(&proof.U1.LinkEvaluations[claim], &term)
			xiPower.Mul(&xiPower, &xi)
		}
		hCoefficients[rank] = locals[rank].dLink
		locals[rank].h = dlinkzgOpeningMonomial(locals[rank].dLink, rank)
		locals[rank].t0 = dlinkzgOpeningMonomial(input.T0[rank], rank)
		locals[rank].t1 = dlinkzgOpeningMonomial(input.T1[rank], rank)
		locals[rank].input = cryptodlinkzg.LocalLaurentInput{
			G:    locals[rank].g,
			PsiQ: prepared.psiQ,
			P:    prepared.scales,
			Xi:   xi,
			HXi:  locals[rank].h,
			PsiR: prepared.psiR,
			Nu:   nu,
			T0:   locals[rank].t0,
			T1:   locals[rank].t1,
			AXi:  aWeights,
			BXi:  bWeights,
		}
		locals[rank].laurent = cryptodlinkzg.BuildLocalLaurent(locals[rank].input)
		commitment, commitErr := input.PartySRS[rank].CommitZ(locals[rank].laurent)
		if commitErr != nil {
			return DLinkZGOpeningProof{}, fmt.Errorf("%w: U1 Laurent rank %d: %v", ErrInvalidDLinkZGOpeningInput, rank, commitErr)
		}
		dlinkzgOpeningAddG1(&laurentCommitment, commitment)
	}
	proof.U1.LinkCommitment, err = input.CoordinatorSRS.CommitZ(hCoefficients)
	if err != nil {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: U1 link commitment: %v", ErrInvalidDLinkZGOpeningInput, err)
	}
	proof.U1.LaurentCommitment.FromJacobian(&laurentCommitment)
	if err := transcript.AppendU1(proof.U1); err != nil {
		return DLinkZGOpeningProof{}, err
	}

	betaOut, err := transcript.DeriveBeta()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	beta := betaOut.Value
	if err := validateDLinkZGOpeningBeta(beta, zChallenge); err != nil {
		return DLinkZGOpeningProof{}, err
	}
	betaInverse := dlinkzgOpeningInverse(beta)

	localSAtBeta := make([]fr.Element, prepared.m)
	localSAtBetaInverse := make([]fr.Element, prepared.m)
	for rank := 0; rank < prepared.m; rank++ {
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			locals[rank].gAtBeta[claim] = cryptodlinkzg.Eval(locals[rank].g[claim], beta)
			locals[rank].gAtBetaInverse[claim] = cryptodlinkzg.Eval(locals[rank].g[claim], betaInverse)
			proof.U2.PartialAtBeta[claim].Add(&proof.U2.PartialAtBeta[claim], &locals[rank].gAtBeta[claim])
			proof.U2.PartialAtBetaInverse[claim].Add(&proof.U2.PartialAtBetaInverse[claim], &locals[rank].gAtBetaInverse[claim])
		}
		localPolynomials := [3][]fr.Element{locals[rank].h, locals[rank].t0, locals[rank].t1}
		for polynomial := range localPolynomials {
			locals[rank].lAtBeta[polynomial] = cryptodlinkzg.Eval(localPolynomials[polynomial], beta)
			locals[rank].lAtBetaInverse[polynomial] = cryptodlinkzg.Eval(localPolynomials[polynomial], betaInverse)
			proof.U2.BatchAtBeta[polynomial].Add(&proof.U2.BatchAtBeta[polynomial], &locals[rank].lAtBeta[polynomial])
			proof.U2.BatchAtBetaInverse[polynomial].Add(&proof.U2.BatchAtBetaInverse[polynomial], &locals[rank].lAtBetaInverse[polynomial])
		}
		localSAtBeta[rank] = cryptodlinkzg.Eval(locals[rank].laurent, beta)
		proof.U2.BatchAtBeta[3].Add(&proof.U2.BatchAtBeta[3], &localSAtBeta[rank])
	}

	// S(beta^{-1}) is the canonical omitted party value: derive the aggregate
	// from the public target and the other thirteen U2 scalars, rather than
	// reading it from the witness polynomial.
	left := dlinkzgOpeningLaurentLeft(proof.U2, prepared, xi, nu, beta)
	target := dlinkzgOpeningLinearTarget(instance, proof.U1.LinkEvaluations, xi, nu)
	proof.U2.BatchAtBetaInverse[3], err = cryptodlinkzg.DeriveLaurentInverseValue(
		beta,
		left,
		target,
		proof.U2.BatchAtBeta[3],
	)
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}

	// Retain the party decomposition for the L-batch.  Each local inverse is
	// derived by the same identity using its independently determined diagonal;
	// for a valid public instance these shares sum to the canonical aggregate.
	var sumLocalInverse fr.Element
	for rank := 0; rank < prepared.m; rank++ {
		localLeft, localErr := cryptodlinkzg.EvalLocalLaurentLeft(locals[rank].input, beta)
		if localErr != nil {
			return DLinkZGOpeningProof{}, localErr
		}
		localTarget := cryptodlinkzg.LocalLaurentDiagonal(locals[rank].input)
		localSAtBetaInverse[rank], localErr = cryptodlinkzg.DeriveLaurentInverseValue(beta, localLeft, localTarget, localSAtBeta[rank])
		if localErr != nil {
			return DLinkZGOpeningProof{}, localErr
		}
		sumLocalInverse.Add(&sumLocalInverse, &localSAtBetaInverse[rank])
	}
	if !sumLocalInverse.Equal(&proof.U2.BatchAtBetaInverse[3]) {
		return DLinkZGOpeningProof{}, ErrDLinkZGOpeningScalarIdentity
	}
	if err := transcript.AppendU2(proof.U2); err != nil {
		return DLinkZGOpeningProof{}, err
	}
	if err := verifyDLinkZGOpeningScalarIdentity(instance, proof.U1, proof.U2, prepared, xi, nu, beta); err != nil {
		return DLinkZGOpeningProof{}, err
	}

	kappaOut, err := transcript.DeriveKappa()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	kappa := kappaOut.Value
	if kappa.IsZero() {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: kappa is zero", ErrInvalidDLinkZGOpeningInput)
	}
	gPoints := []fr.Element{zChallenge, beta, betaInverse}
	lPoints := []fr.Element{beta, betaInverse}
	var wG, wL, piZ bn254.G1Jac
	for rank := 0; rank < prepared.m; rank++ {
		gInputs := make([]cryptodlinkzg.SameSetInput, dlinkzgOpeningCircuitClaims)
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			var atZ fr.Element
			atZ.Mul(&prepared.weights[rank], &locals[rank].d[claim])
			gInputs[claim] = cryptodlinkzg.SameSetInput{
				Polynomial: locals[rank].g[claim],
				ClaimedValues: []fr.Element{
					atZ,
					locals[rank].gAtBeta[claim],
					locals[rank].gAtBetaInverse[claim],
				},
			}
		}
		gBatch, batchErr := cryptodlinkzg.BuildSameSetQuotient(gInputs, gPoints, kappa)
		if batchErr != nil {
			return DLinkZGOpeningProof{}, fmt.Errorf("%w: local G batch rank %d: %v", ErrDLinkZGOpeningRejected, rank, batchErr)
		}
		wGShare, commitErr := input.PartySRS[rank].CommitZ(gBatch.Quotient)
		if commitErr != nil {
			return DLinkZGOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&wG, wGShare)

		lPolynomials := [4][]fr.Element{locals[rank].h, locals[rank].t0, locals[rank].t1, locals[rank].laurent}
		lInputs := make([]cryptodlinkzg.SameSetInput, len(lPolynomials))
		for polynomial := range lPolynomials {
			values := []fr.Element{localSAtBeta[rank], localSAtBetaInverse[rank]}
			if polynomial < len(locals[rank].lAtBeta) {
				values[0] = locals[rank].lAtBeta[polynomial]
				values[1] = locals[rank].lAtBetaInverse[polynomial]
			}
			lInputs[polynomial] = cryptodlinkzg.SameSetInput{Polynomial: lPolynomials[polynomial], ClaimedValues: values}
		}
		lBatch, batchErr := cryptodlinkzg.BuildSameSetQuotient(lInputs, lPoints, kappa)
		if batchErr != nil {
			return DLinkZGOpeningProof{}, fmt.Errorf("%w: local L batch rank %d: %v", ErrDLinkZGOpeningRejected, rank, batchErr)
		}
		wLShare, commitErr := input.PartySRS[rank].CommitZ(lBatch.Quotient)
		if commitErr != nil {
			return DLinkZGOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&wL, wLShare)

		pXi := make([]fr.Element, prepared.t)
		xiPower := fr.One()
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			dlinkzgOpeningAddScaled(pXi, locals[rank].shifted[claim], xiPower)
			xiPower.Mul(&xiPower, &xi)
		}
		qZ, remainder := cryptodlinkzg.SyntheticDivision(pXi, zChallenge)
		if !remainder.Equal(&locals[rank].dLink) {
			return DLinkZGOpeningProof{}, fmt.Errorf("%w: source remainder rank %d", ErrDLinkZGOpeningRejected, rank)
		}
		piZShare, commitErr := input.PartySRS[rank].CommitRow(qZ)
		if commitErr != nil {
			return DLinkZGOpeningProof{}, commitErr
		}
		dlinkzgOpeningAddG1(&piZ, piZShare)
	}
	proof.U3.WG.FromJacobian(&wG)
	proof.U3.WL.FromJacobian(&wL)
	proof.U3.PiZ.FromJacobian(&piZ)
	qY, sourceValue := cryptodlinkzg.SyntheticDivision(hCoefficients, beta)
	if !sourceValue.Equal(&proof.U2.BatchAtBeta[0]) {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: source link value", ErrDLinkZGOpeningRejected)
	}
	proof.U3.PiY, err = input.CoordinatorSRS.CommitY(qY)
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	if err := transcript.AppendU3(proof.U3); err != nil {
		return DLinkZGOpeningProof{}, err
	}
	deltaOut, err := transcript.DeriveDelta()
	if err != nil {
		return DLinkZGOpeningProof{}, err
	}
	if deltaOut.Value.IsZero() {
		return DLinkZGOpeningProof{}, fmt.Errorf("%w: delta is zero", ErrInvalidDLinkZGOpeningInput)
	}
	return proof, nil
}

func verifyDLinkZGOpeningWithTranscript(instance DLinkZGOpeningInstance, proof DLinkZGOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS, transcriptFactory dlinkzgOpeningTranscriptFactory) error {
	prepared, err := prepareDLinkZGOpeningInstance(instance)
	if err != nil {
		return err
	}
	if verifierSRS == nil {
		return fmt.Errorf("%w: nil verifier SRS", ErrInvalidDLinkZGOpeningInstance)
	}
	if err := verifierSRS.Validate(); err != nil {
		return fmt.Errorf("%w: verifier SRS: %v", ErrInvalidDLinkZGOpeningInstance, err)
	}
	if transcriptFactory == nil {
		return fmt.Errorf("%w: nil transcript factory", ErrInvalidDLinkZGOpeningInstance)
	}
	transcript, err := transcriptFactory(instance.TranscriptContext)
	if err != nil {
		return err
	}
	if transcript == nil {
		return fmt.Errorf("%w: transcript factory returned nil", ErrInvalidDLinkZGOpeningInstance)
	}
	if err := transcript.AppendU0(proof.U0); err != nil {
		return err
	}
	xiOut, err := transcript.DeriveXi()
	if err != nil {
		return err
	}
	nuOut, err := transcript.DeriveNu()
	if err != nil {
		return err
	}
	zOut, err := transcript.DeriveZChallenge()
	if err != nil {
		return err
	}
	if xiOut.Value.IsZero() || nuOut.Value.IsZero() {
		return fmt.Errorf("%w: zero xi or nu", ErrDLinkZGOpeningRejected)
	}
	if err := transcript.AppendU1(proof.U1); err != nil {
		return err
	}
	betaOut, err := transcript.DeriveBeta()
	if err != nil {
		return err
	}
	if err := validateDLinkZGOpeningBeta(betaOut.Value, zOut.Value); err != nil {
		return err
	}
	if err := transcript.AppendU2(proof.U2); err != nil {
		return err
	}
	if err := verifyDLinkZGOpeningScalarIdentity(instance, proof.U1, proof.U2, prepared, xiOut.Value, nuOut.Value, betaOut.Value); err != nil {
		return err
	}
	kappaOut, err := transcript.DeriveKappa()
	if err != nil {
		return err
	}
	if kappaOut.Value.IsZero() {
		return fmt.Errorf("%w: zero kappa", ErrDLinkZGOpeningRejected)
	}
	if err := transcript.AppendU3(proof.U3); err != nil {
		return err
	}
	deltaOut, err := transcript.DeriveDelta()
	if err != nil {
		return err
	}
	if deltaOut.Value.IsZero() {
		return fmt.Errorf("%w: zero delta", ErrDLinkZGOpeningRejected)
	}
	return verifyDLinkZGOpeningFinal(
		instance,
		proof,
		verifierSRS,
		prepared,
		xiOut.Value,
		zOut.Value,
		betaOut.Value,
		kappaOut.Value,
		deltaOut.Value,
	)
}

func prepareDLinkZGOpeningInstance(instance DLinkZGOpeningInstance) (dlinkzgOpeningPrepared, error) {
	var prepared dlinkzgOpeningPrepared
	if instance.LocalDomainSize < 4 || instance.LocalDomainSize > FastLocalPIOPMaxDomainSize || !isPowerOfTwo(instance.LocalDomainSize) {
		return prepared, fmt.Errorf("%w: T=%d is not a supported power of two in [4,%d]", ErrInvalidDLinkZGOpeningInstance, instance.LocalDomainSize, FastLocalPIOPMaxDomainSize)
	}
	if len(instance.PartitionPoint) == 0 || len(instance.PartitionPoint) >= 63 {
		return prepared, fmt.Errorf("%w: invalid partition point length %d", ErrInvalidDLinkZGOpeningInstance, len(instance.PartitionPoint))
	}
	m64 := uint64(1) << uint(len(instance.PartitionPoint))
	if m64 > uint64(instance.LocalDomainSize) || m64 > uint64(^uint(0)>>1) {
		return prepared, fmt.Errorf("%w: M=%d exceeds T=%d", ErrInvalidDLinkZGOpeningInstance, m64, instance.LocalDomainSize)
	}
	for i := range instance.SourceCommitments {
		if err := validateTranscriptG1(&instance.SourceCommitments[i]); err != nil {
			return prepared, fmt.Errorf("%w: source commitment %d", ErrInvalidDLinkZGOpeningInstance, i)
		}
	}
	for i := range instance.TreeCommitments {
		if err := validateTranscriptG1(&instance.TreeCommitments[i]); err != nil {
			return prepared, fmt.Errorf("%w: tree commitment %d", ErrInvalidDLinkZGOpeningInstance, i)
		}
	}
	fieldGroups := []struct {
		name   string
		values []fr.Element
	}{
		{name: "circuit claim", values: instance.CircuitClaims[:]},
		{name: "tree claim", values: instance.TreeClaims[:]},
		{name: "semantic query", values: instance.SemanticQueryPoints[:]},
		{name: "shift", values: []fr.Element{instance.Shift}},
		{name: "partition point", values: instance.PartitionPoint},
	}
	for group := range fieldGroups {
		for index := range fieldGroups[group].values {
			if err := validateTranscriptField(&fieldGroups[group].values[index]); err != nil {
				return prepared, fmt.Errorf("%w: non-canonical %s %d", ErrInvalidDLinkZGOpeningInstance, fieldGroups[group].name, index)
			}
		}
	}
	if err := validateTranscriptContext(instance.TranscriptContext); err != nil {
		return prepared, err
	}
	wantStatementDigest := DLinkZGOpeningInstanceDigest(instance)
	if instance.TranscriptContext.PublicStatementDigest != wantStatementDigest {
		return prepared, fmt.Errorf("%w: transcript public-statement digest does not bind the ordered opening instance", ErrInvalidDLinkZGOpeningInstance)
	}

	prepared.m = int(m64)
	prepared.t = instance.LocalDomainSize
	prepared.tree = productCheckFunctionalPoints(instance.PartitionPoint)
	shiftedQueries := instance.SemanticQueryPoints
	for i := range shiftedQueries {
		shiftedQueries[i].Sub(&shiftedQueries[i], &instance.Shift)
	}
	queries, scales, err := translatedQueries(shiftedQueries, bits.Len(uint(instance.LocalDomainSize))-1)
	if err != nil {
		return prepared, fmt.Errorf("%w: %v", ErrInvalidDLinkZGOpeningInstance, err)
	}
	prepared.queries = queries
	prepared.scales = scales
	return prepared, nil
}

func validateDLinkZGOpeningInput(input DLinkZGOpeningProverInput, prepared dlinkzgOpeningPrepared) error {
	if len(input.T0) != prepared.m || len(input.T1) != prepared.m {
		return fmt.Errorf("%w: t_0,t_1 must both have M=%d coefficients", ErrInvalidDLinkZGOpeningInput, prepared.m)
	}
	if len(input.PartySRS) != prepared.m {
		return fmt.Errorf("%w: got %d party SRS views, want %d", ErrInvalidDLinkZGOpeningInput, len(input.PartySRS), prepared.m)
	}
	if input.CoordinatorSRS == nil || input.VerifierSRS == nil {
		return fmt.Errorf("%w: missing coordinator or verifier SRS", ErrInvalidDLinkZGOpeningInput)
	}
	if err := input.CoordinatorSRS.Validate(); err != nil || input.CoordinatorSRS.Parties != prepared.m {
		return fmt.Errorf("%w: coordinator SRS", ErrInvalidDLinkZGOpeningInput)
	}
	if err := input.VerifierSRS.Validate(); err != nil {
		return fmt.Errorf("%w: verifier SRS", ErrInvalidDLinkZGOpeningInput)
	}
	for rank := 0; rank < prepared.m; rank++ {
		row := input.PartySRS[rank]
		if row == nil || row.Validate() != nil || row.Rank != rank || row.Parties != prepared.m || row.DegreeBound != prepared.t {
			return fmt.Errorf("%w: party SRS at manifest rank %d", ErrInvalidDLinkZGOpeningInput, rank)
		}
	}
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		if len(input.Sources[claim]) != prepared.m {
			return fmt.Errorf("%w: source %d has %d rows, want %d", ErrInvalidDLinkZGOpeningInput, claim, len(input.Sources[claim]), prepared.m)
		}
		for rank := range input.Sources[claim] {
			if len(input.Sources[claim][rank]) > prepared.t {
				return fmt.Errorf("%w: source %d rank %d exceeds T", ErrInvalidDLinkZGOpeningInput, claim, rank)
			}
		}
	}
	return nil
}

func validateDLinkZGOpeningBeta(beta, zChallenge fr.Element) error {
	zero := fr.Element{}
	one := fr.One()
	minusOne := one
	minusOne.Neg(&minusOne)
	if beta.Equal(&zero) || beta.Equal(&one) || beta.Equal(&minusOne) || beta.Equal(&zChallenge) {
		return fmt.Errorf("%w: beta is in the opening exclusion set", ErrDLinkZGOpeningRejected)
	}
	if !zChallenge.IsZero() {
		zInverse := dlinkzgOpeningInverse(zChallenge)
		if beta.Equal(&zInverse) {
			return fmt.Errorf("%w: beta equals z_ch inverse", ErrDLinkZGOpeningRejected)
		}
	}
	return nil
}

func verifyDLinkZGOpeningScalarIdentity(instance DLinkZGOpeningInstance, u1 U1Message, u2 U2Message, prepared dlinkzgOpeningPrepared, xi, nu, beta fr.Element) error {
	left := dlinkzgOpeningLaurentLeft(u2, prepared, xi, nu, beta)
	target := dlinkzgOpeningLinearTarget(instance, u1.LinkEvaluations, xi, nu)
	betaInverse := dlinkzgOpeningInverse(beta)
	var right, term fr.Element
	right.Double(&target)
	term.Mul(&beta, &u2.BatchAtBeta[3])
	right.Add(&right, &term)
	term.Mul(&betaInverse, &u2.BatchAtBetaInverse[3])
	right.Add(&right, &term)
	if !left.Equal(&right) {
		return ErrDLinkZGOpeningScalarIdentity
	}
	return nil
}

func dlinkzgOpeningLaurentLeft(u2 U2Message, prepared dlinkzgOpeningPrepared, xi, nu, beta fr.Element) fr.Element {
	betaInverse := dlinkzgOpeningInverse(beta)
	var left, symmetric, term fr.Element
	xiPower := fr.One()
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		psiAtBeta := dlinkzgOpeningPsiEval(prepared.queries[claim], beta)
		psiAtBetaInverse := dlinkzgOpeningPsiEval(prepared.queries[claim], betaInverse)
		symmetric.Mul(&u2.PartialAtBeta[claim], &psiAtBetaInverse)
		term.Mul(&u2.PartialAtBetaInverse[claim], &psiAtBeta)
		symmetric.Add(&symmetric, &term)
		var scale fr.Element
		scale.Mul(&xiPower, &prepared.scales[claim])
		term.Mul(&scale, &symmetric)
		left.Add(&left, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	psiRAtBeta := dlinkzgOpeningPsiEval(prepared.tree[0].partition, beta)
	psiRAtBetaInverse := dlinkzgOpeningPsiEval(prepared.tree[0].partition, betaInverse)
	symmetric.Mul(&u2.BatchAtBeta[0], &psiRAtBetaInverse)
	term.Mul(&u2.BatchAtBetaInverse[0], &psiRAtBeta)
	symmetric.Add(&symmetric, &term)
	term.Mul(&nu, &symmetric)
	left.Add(&left, &term)
	aAtBeta, bAtBeta := dlinkzgOpeningTreeWeightEvals(prepared.tree, xi, beta)
	aAtBetaInverse, bAtBetaInverse := dlinkzgOpeningTreeWeightEvals(prepared.tree, xi, betaInverse)
	for index, evaluations := range [][2]fr.Element{{aAtBeta, aAtBetaInverse}, {bAtBeta, bAtBetaInverse}} {
		weightsAtBeta := evaluations[0]
		weightsAtBetaInverse := evaluations[1]
		symmetric.Mul(&u2.BatchAtBeta[index+1], &weightsAtBetaInverse)
		term.Mul(&u2.BatchAtBetaInverse[index+1], &weightsAtBeta)
		symmetric.Add(&symmetric, &term)
		left.Add(&left, &symmetric)
	}
	return left
}

func dlinkzgOpeningLinearTarget(instance DLinkZGOpeningInstance, linkValues [dlinkzgOpeningCircuitClaims]fr.Element, xi, nu fr.Element) fr.Element {
	var target, term fr.Element
	xiPower := fr.One()
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		term.Mul(&xiPower, &instance.CircuitClaims[claim])
		target.Add(&target, &term)
		term.Mul(&xiPower, &linkValues[claim]).Mul(&term, &nu)
		target.Add(&target, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	for claim := 0; claim < dlinkzgOpeningTreeClaims; claim++ {
		term.Mul(&xiPower, &instance.TreeClaims[claim])
		target.Add(&target, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	return target
}

// dlinkzgOpeningPsiEval evaluates
// psi_s(Z)=sum_i chi_i(s)Z^i in product form, without materializing its
// 2^len(s) coefficients.  Bit k contributes (1-s_k)+s_k Z^(2^k).
func dlinkzgOpeningPsiEval(point []fr.Element, evaluationPoint fr.Element) fr.Element {
	result := fr.One()
	power := evaluationPoint
	for coordinate := range point {
		oneMinus := fr.One()
		oneMinus.Sub(&oneMinus, &point[coordinate])
		var factor, term fr.Element
		term.Mul(&point[coordinate], &power)
		factor.Add(&oneMinus, &term)
		result.Mul(&result, &factor)
		power.Square(&power)
	}
	return result
}

// dlinkzgOpeningTreeWeightEvals evaluates A_xi and B_xi directly from the
// five ProductCheck points in O(log M), rather than constructing T-wide
// coefficient arrays.  The prover still materializes those arrays once for
// FastOffDiag; the verifier never does.
func dlinkzgOpeningTreeWeightEvals(points [dlinkzgOpeningTreeClaims]productCheckPoint, xi, evaluationPoint fr.Element) (fr.Element, fr.Element) {
	var a, b fr.Element
	xiPower := fieldPower(xi, dlinkzgOpeningCircuitClaims)
	for claim := 0; claim < dlinkzgOpeningTreeClaims; claim++ {
		psi := dlinkzgOpeningPsiEval(points[claim].partition, evaluationPoint)
		oneMinus := fr.One()
		oneMinus.Sub(&oneMinus, &points[claim].selector)
		var term fr.Element
		term.Mul(&xiPower, &oneMinus).Mul(&term, &psi)
		a.Add(&a, &term)
		term.Mul(&xiPower, &points[claim].selector).Mul(&term, &psi)
		b.Add(&b, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	return a, b
}

func verifyDLinkZGOpeningFinal(instance DLinkZGOpeningInstance, proof DLinkZGOpeningProof, verifierSRS *cryptodlinkzg.VerifierSRS, prepared dlinkzgOpeningPrepared, xi, zChallenge, beta, kappa, delta fr.Element) error {
	if err := verifierSRS.Validate(); err != nil {
		return fmt.Errorf("%w: verifier SRS: %v", ErrInvalidDLinkZGOpeningInstance, err)
	}
	betaInverse := dlinkzgOpeningInverse(beta)
	gPoints := []fr.Element{zChallenge, beta, betaInverse}
	gInterpolants := make([][]fr.Element, dlinkzgOpeningCircuitClaims)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		values := []fr.Element{
			proof.U1.LinkEvaluations[claim],
			proof.U2.PartialAtBeta[claim],
			proof.U2.PartialAtBetaInverse[claim],
		}
		interpolant, err := cryptodlinkzg.Interpolate(gPoints, values)
		if err != nil {
			return fmt.Errorf("%w: G interpolant: %v", ErrDLinkZGOpeningRejected, err)
		}
		gInterpolants[claim] = interpolant
	}
	numeratorG, err := verifierSRS.FoldSameSetCommitments(proof.U0.PartialCommitments[:], gInterpolants, kappa)
	if err != nil {
		return fmt.Errorf("%w: G numerator: %v", ErrDLinkZGOpeningRejected, err)
	}

	lPoints := []fr.Element{beta, betaInverse}
	lInterpolants := make([][]fr.Element, 4)
	for polynomial := 0; polynomial < len(lInterpolants); polynomial++ {
		values := []fr.Element{proof.U2.BatchAtBeta[polynomial], proof.U2.BatchAtBetaInverse[polynomial]}
		interpolant, interpolateErr := cryptodlinkzg.Interpolate(lPoints, values)
		if interpolateErr != nil {
			return fmt.Errorf("%w: L interpolant: %v", ErrDLinkZGOpeningRejected, interpolateErr)
		}
		lInterpolants[polynomial] = interpolant
	}
	lCommitments := []bn254.G1Affine{
		proof.U1.LinkCommitment,
		instance.TreeCommitments[0],
		instance.TreeCommitments[1],
		proof.U1.LaurentCommitment,
	}
	numeratorL, err := verifierSRS.FoldSameSetCommitments(lCommitments, lInterpolants, kappa)
	if err != nil {
		return fmt.Errorf("%w: L numerator: %v", ErrDLinkZGOpeningRejected, err)
	}

	sourceCommitment := foldG1(instance.SourceCommitments[:], xi)
	statement := cryptodlinkzg.DeltaBatchStatement{
		SourceCommitment: sourceCommitment,
		SourceValue:      proof.U2.BatchAtBeta[0],
		Beta:             beta,
		ZChallenge:       zChallenge,
		NumeratorG:       numeratorG,
		NumeratorL:       numeratorL,
		VanishingG:       cryptodlinkzg.VanishingPolynomial(gPoints),
		VanishingL:       cryptodlinkzg.VanishingPolynomial(lPoints),
	}
	batchProof := cryptodlinkzg.DeltaBatchProof{
		PiZ: proof.U3.PiZ,
		PiY: proof.U3.PiY,
		WG:  proof.U3.WG,
		WL:  proof.U3.WL,
	}
	if err := verifierSRS.VerifyDeltaBatch(statement, batchProof, delta); err != nil {
		return fmt.Errorf("%w: %v", ErrDLinkZGOpeningRejected, err)
	}
	return nil
}

func dlinkzgOpeningScalePolynomial(polynomial []fr.Element, scale fr.Element) []fr.Element {
	result := make([]fr.Element, len(polynomial))
	for i := range polynomial {
		result[i].Mul(&polynomial[i], &scale)
	}
	return result
}

func dlinkzgOpeningMonomial(coefficient fr.Element, degree int) []fr.Element {
	result := make([]fr.Element, degree+1)
	result[degree] = coefficient
	return result
}

func dlinkzgOpeningAddScaled(destination []fr.Element, source []fr.Element, scale fr.Element) {
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

func dlinkzgOpeningAddG1(destination *bn254.G1Jac, value bn254.G1Affine) {
	var jacobian bn254.G1Jac
	jacobian.FromAffine(&value)
	destination.AddAssign(&jacobian)
}

func dlinkzgOpeningInverse(value fr.Element) fr.Element {
	var result fr.Element
	result.Inverse(&value)
	return result
}
