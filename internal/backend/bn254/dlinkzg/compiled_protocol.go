package dlinkzg

// This file is the centralized executable composition of Protocols 1 and 2.
// It preserves the split roles used by the distributed implementation: the
// prover iterates over manifest-ordered party keys and one coordinator key,
// while CompiledVerify receives only the public metadata and constant-size
// verifier role. No retained W3 table or clear coordinator oracle is included
// in CompiledProof.

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
)

const compiledPublicStatementDomain = "DLinKZG/compiled-public-statement/v1"

var (
	// ErrInvalidCompiledStatement reports a public prefix or outer context
	// which is malformed or is not bound to the selected compiled setup.
	ErrInvalidCompiledStatement = errors.New("dlinkzg: invalid compiled statement")
	// ErrInvalidCompiledProverInput reports a malformed full solution or an
	// online proving role that cannot execute the compiled protocol.
	ErrInvalidCompiledProverInput = errors.New("dlinkzg: invalid compiled prover input")
	// ErrCompiledVerification wraps a well-shaped compiled proof which fails
	// the outer PIOP, endpoint, source-commitment, or opening checks.
	ErrCompiledVerification = errors.New("dlinkzg: compiled proof rejected")
)

// CompiledStatement is the complete per-proof public input. The ordered
// public solution prefix is hashed canonically into OuterContext's
// PublicStatementDigest; it is also used directly to reconstruct PI(alpha)
// and PI(r). The caller owns both fields and they are never mutated.
type CompiledStatement struct {
	PublicSolutionPrefix []fr.Element
	OuterContext         OuterTranscriptContext
}

// CompiledVerifyingKey is the verifier-only role. In particular it contains
// neither a party row SRS, a SparseR1CS, a local table, nor the coordinator's
// O(M) SRS. Metadata authenticates the public fixed commitments and all outer
// transcript parameters.
type CompiledVerifyingKey struct {
	Metadata CompiledSetupMetadata
	Verifier CompiledVerifierSetup
}

// CompiledPublicStatementDigest canonically hashes the ordered public
// solution prefix. The length is explicit, so a zero suffix and a shorter
// vector cannot collide at the encoding layer.
func CompiledPublicStatementDigest(publicSolutionPrefix []fr.Element) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledPublicStatementDomain)
	compiledHashFields(hasher, publicSolutionPrefix)
	return compiledDigest(hasher)
}

// VerifyingKey extracts the authenticated verifier-only role after the cheap
// online boundary checks. The returned value does not expose Parties or the
// Coordinator.
func (setup *CompiledSetup) VerifyingKey() (CompiledVerifyingKey, error) {
	if err := setup.ValidateOnlineRoles(); err != nil {
		return CompiledVerifyingKey{}, err
	}
	vk := CompiledVerifyingKey{Metadata: setup.Metadata, Verifier: setup.Verifier}
	if err := validateCompiledVerifyingKey(&vk); err != nil {
		return CompiledVerifyingKey{}, err
	}
	return vk, nil
}

// CompiledProve executes the complete centralized reference. Fixed adapter
// preprocessing and the split SRS are reused from setup; only solution-to-row
// extraction, local PIOP arithmetic, commitments, the coordinator reduction,
// and Protocol 2 occur on this path.
func CompiledProve(
	setup *CompiledSetup,
	statement CompiledStatement,
	solution []fr.Element,
) (CompiledProof, error) {
	var proof CompiledProof
	if err := validateCompiledProverInputs(setup, statement, solution); err != nil {
		return proof, err
	}

	m := setup.Metadata.PartitionCount
	t := setup.Metadata.LocalDomainSize
	states := make([]*FastLocalPIOPState, m)
	partySRS := make([]*cryptodlinkzg.PartyRowSRS, m)
	publicCommitments := AggregateLocalPIOPPublicCommitments{Parties: m, DomainSize: t}

	// W0: interpolate the three online witness columns and commit in the
	// semantic X coordinate before sampling eta_part, eta_X, gamma.
	for rank := 0; rank < m; rank++ {
		party := &setup.Parties[rank]
		table, _, err := compiledBuildLocalPIOPTableFromSolution(
			party.Preprocessing, party.ConstraintSystem, solution,
		)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d table: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		states[rank], err = PrepareFastLocalPIOPWithFixed(table, party.FastFixed)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d W0 arithmetic: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		w0, err := states[rank].W0()
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d W0: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		partySRS[rank] = party.RowSRS
		for wire := 0; wire < LocalWireCount; wire++ {
			commitment, err := party.RowSRS.CommitSemantic(w0.Wires[wire])
			if err != nil {
				return CompiledProof{}, fmt.Errorf("%w: rank %d W0 commitment %d: %v", ErrInvalidCompiledProverInput, rank, wire, err)
			}
			sourceCommitmentAdd(&publicCommitments.W0[wire], &commitment)
		}
	}
	proof.W0.WitnessCommitments = publicCommitments.W0

	outer, err := NewOuterTranscript(statement.OuterContext)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: outer transcript: %v", ErrInvalidCompiledStatement, err)
	}
	if err := outer.AppendW0(proof.W0); err != nil {
		return CompiledProof{}, err
	}
	initial, err := outer.DeriveInitialChallenges()
	if err != nil {
		return CompiledProof{}, err
	}
	localChallenges := LocalPIOPChallenges{
		EtaPart: initial.EtaPart.Value,
		EtaX:    initial.EtaX.Value,
		Gamma:   initial.Gamma.Value,
	}

	// W1 is lambda-independent by construction.
	for rank := 0; rank < m; rank++ {
		w1, err := states[rank].BuildAccumulator(
			localChallenges.EtaPart, localChallenges.EtaX, localChallenges.Gamma,
		)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d W1 arithmetic: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		commitment, err := partySRS[rank].CommitSemantic(w1.Accumulator)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d W1 commitment: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		sourceCommitmentAdd(&publicCommitments.W1, &commitment)
	}
	proof.W1.AccumulatorCommitment = publicCommitments.W1
	if err := outer.AppendW1(proof.W1); err != nil {
		return CompiledProof{}, err
	}
	lambda, err := outer.DeriveLambda()
	if err != nil {
		return CompiledProof{}, err
	}
	localChallenges.Lambda = lambda.Value

	// W2 completes the exact three quotient chunks per party.
	relations := make([]*LocalPIOPRelation, m)
	for rank := 0; rank < m; rank++ {
		relations[rank], err = states[rank].BuildQuotient(localChallenges.Lambda)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d W2 arithmetic: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		for chunk := range relations[rank].Quotient {
			commitment, err := partySRS[rank].CommitSemantic(relations[rank].Quotient[chunk])
			if err != nil {
				return CompiledProof{}, fmt.Errorf("%w: rank %d W2 commitment %d: %v", ErrInvalidCompiledProverInput, rank, chunk, err)
			}
			sourceCommitmentAdd(&publicCommitments.W2[chunk], &commitment)
		}
	}
	proof.W2.QuotientCommitments = publicCommitments.W2
	if err := outer.AppendW2(proof.W2); err != nil {
		return CompiledProof{}, err
	}
	alphaOut, err := outer.DeriveAlpha()
	if err != nil {
		return CompiledProof{}, err
	}
	alpha := alphaOut.Value

	// W3 remains retained coordinator state. Only its exact 21-field rows are
	// used to construct ProductCheck and SumCheck; no W3 bytes enter proof.
	terminals := make([]LocalTerminalEvaluations, m)
	for rank := 0; rank < m; rank++ {
		terminals[rank], err = relations[rank].TerminalEvaluations(alpha)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d terminal evaluations: %v", ErrInvalidCompiledProverInput, rank, err)
		}
	}
	publicInputAtAlpha, err := setup.Coordinator.PublicInputPlacement.EvaluationsAtAlpha(
		statement.PublicSolutionPrefix, alpha,
	)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: public input at alpha: %v", ErrInvalidCompiledStatement, err)
	}
	if err := outer.MarkRetainedW3Complete(); err != nil {
		return CompiledProof{}, err
	}

	omega := setup.Metadata.LocalDomainGenerator
	xStar := compiledProtocolInverse(omega)
	wireCosets := compiledProtocolWireCosets()
	coordinatorState, err := BuildOuterPIOPCoordinatorStateFromTerminals(
		terminals,
		publicInputAtAlpha,
		OuterPIOPReductionContext{
			DomainSize: t,
			Omega:      omega,
			XStar:      xStar,
			Alpha:      alpha,
			WireCosets: wireCosets,
			Challenges: localChallenges,
		},
	)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: outer reduction: %v", ErrInvalidCompiledProverInput, err)
	}
	_, t0, t1 := coordinatorState.CoordinatorProductCheckWitness()
	proof.ProductCheck.Commitments[0], err = setup.Coordinator.SRS.CommitU(t0)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: t0 commitment: %v", ErrInvalidCompiledProverInput, err)
	}
	proof.ProductCheck.Commitments[1], err = setup.Coordinator.SRS.CommitU(t1)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: t1 commitment: %v", ErrInvalidCompiledProverInput, err)
	}
	if err := outer.AppendProductCheckCommitments(proof.ProductCheck); err != nil {
		return CompiledProof{}, err
	}
	zetaTheta, err := outer.DeriveZetaTheta()
	if err != nil {
		return CompiledProof{}, err
	}
	theta := make([]fr.Element, len(zetaTheta.Theta))
	for coordinate := range zetaTheta.Theta {
		theta[coordinate] = zetaTheta.Theta[coordinate].Value
	}

	// Stateful Fiat--Shamir SumCheck: every polynomial is appended before its
	// challenge is derived and bound into the next fold.
	sumCheckInstance, err := coordinatorState.BuildSumCheckInstance(zetaTheta.Zeta.Value, theta)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: SumCheck instance: %v", ErrInvalidCompiledProverInput, err)
	}
	oracleTables, composition := sumCheckInstance.CoordinatorOracleTables()
	var zero fr.Element
	sumCheckProver, err := NewDegreeFiveSumCheckProver(zero, oracleTables, composition)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: SumCheck prover: %v", ErrInvalidCompiledProverInput, err)
	}
	roundCount := bits.Len(uint(m)) - 1
	proof.SumCheckRounds = make([]OuterSumCheckRoundMessage, roundCount)
	roundChallenges := make([]fr.Element, roundCount)
	sumCheckTranscript := SumCheckTranscript{Rounds: make([][]fr.Element, roundCount)}
	for round := 0; round < roundCount; round++ {
		coefficients, err := sumCheckProver.NextRound()
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: SumCheck round %d: %v", ErrInvalidCompiledProverInput, round, err)
		}
		proof.SumCheckRounds[round].Coefficients = coefficients
		sumCheckTranscript.Rounds[round] = append([]fr.Element(nil), coefficients[:]...)
		if err := outer.AppendSumCheckRound(proof.SumCheckRounds[round]); err != nil {
			return CompiledProof{}, err
		}
		roundChallenge, err := outer.DeriveSumCheckChallenge()
		if err != nil {
			return CompiledProof{}, err
		}
		roundChallenges[round] = roundChallenge.Value
		if err := sumCheckProver.BindChallenge(roundChallenges[round]); err != nil {
			return CompiledProof{}, fmt.Errorf("%w: bind SumCheck round %d: %v", ErrInvalidCompiledProverInput, round, err)
		}
	}
	finalOracles, err := sumCheckProver.FinalEvaluations()
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: SumCheck final evaluations: %v", ErrInvalidCompiledProverInput, err)
	}
	folded := outerPIOPTerminalsFromOracleValues(finalOracles)
	copy(proof.FinalEvaluations.Terminal[:], folded.Flatten())
	treePoints := productCheckFunctionalPoints(roundChallenges)
	for claim := 0; claim < OuterProductCheckClaimCount; claim++ {
		proof.FinalEvaluations.ProductCheck[claim], err = evaluateTreeFunctional(t0, t1, treePoints[claim])
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: tree functional %d: %v", ErrInvalidCompiledProverInput, claim, err)
		}
	}
	if err := outer.AppendFinalEvaluations(proof.FinalEvaluations); err != nil {
		return CompiledProof{}, err
	}
	muOut, err := outer.DeriveSourceCompressionChallenges()
	if err != nil {
		return CompiledProof{}, err
	}
	var mu [LocalCompressedSourceCount]fr.Element
	for group := range muOut {
		mu[group] = muOut[group].Value
	}

	publicInputAtR, err := setup.Coordinator.PublicInputPlacement.FoldAtPartitionPoint(
		statement.PublicSolutionPrefix, alpha, roundChallenges,
	)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: public input at r: %v", ErrInvalidCompiledStatement, err)
	}
	outerSumCheckProof := OuterPIOPSumCheckProof{
		Transcript:      sumCheckTranscript,
		FoldedCircuit:   folded,
		TreeEvaluations: proof.FinalEvaluations.ProductCheck,
	}
	if err := VerifyOuterPIOPSumCheck(
		outerSumCheckProof,
		publicInputAtR,
		sumCheckInstance.EndpointContext(roundChallenges),
	); err != nil {
		return CompiledProof{}, fmt.Errorf("%w: prover endpoint self-check: %v", ErrInvalidCompiledProverInput, err)
	}

	// The three source commitments are public linear combinations of fixed
	// commitments and W0--W2. The clear 13/1/7 sources are retained only as
	// rank-local Protocol 2 prover input.
	sourceCommitments, err := DeriveLocalTerminalSourceCommitments(
		AggregateLocalPIOPCommitmentView{
			Fixed:  setup.Coordinator.FixedCommitments,
			Public: publicCommitments,
		},
		t,
		wireCosets,
		LocalTerminalSourceCommitmentChallenges{
			EtaPart: localChallenges.EtaPart,
			EtaX:    localChallenges.EtaX,
			Gamma:   localChallenges.Gamma,
			Alpha:   alpha,
			Mu:      mu,
		},
	)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: source commitments: %v", ErrInvalidCompiledProverInput, err)
	}
	var sources [LocalCompressedSourceCount][][]fr.Element
	for group := range sources {
		sources[group] = make([][]fr.Element, m)
	}
	for rank := 0; rank < m; rank++ {
		compression, err := CompressLocalTerminalSources(relations[rank], alpha, mu)
		if err != nil {
			return CompiledProof{}, fmt.Errorf("%w: rank %d source compression: %v", ErrInvalidCompiledProverInput, rank, err)
		}
		for group := range sources {
			sources[group][rank] = compression.Polynomials[group]
		}
	}

	openingContext, err := outer.OpeningContext()
	if err != nil {
		return CompiledProof{}, err
	}
	openingInstance := compiledOpeningInstance(
		t,
		alpha,
		omega,
		xStar,
		setup.Metadata.IndexShift.Sigma,
		roundChallenges,
		sourceCommitments.Commitments,
		proof.ProductCheck.Commitments,
		folded,
		proof.FinalEvaluations.ProductCheck,
		mu,
		openingContext,
	)
	openingProof, err := ProveHybridOpening(
		openingInstance,
		DLinkZGOpeningProverInput{
			Sources:        sources,
			T0:             t0,
			T1:             t1,
			PartySRS:       partySRS,
			CoordinatorSRS: setup.Coordinator.SRS,
			VerifierSRS:    setup.Verifier.SRS,
		},
	)
	if err != nil {
		return CompiledProof{}, fmt.Errorf("%w: opening prover: %v", ErrInvalidCompiledProverInput, err)
	}
	proof.U0 = openingProof.U0
	proof.U1 = openingProof.U1
	proof.U2 = openingProof.U2
	proof.U3 = openingProof.U3
	return proof, nil
}

// CompiledVerify replays both Fiat--Shamir layers using only vk, statement,
// and proof. It does not accept or reference party/coordinator tables or keys.
func CompiledVerify(
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	proof CompiledProof,
) error {
	if err := validateCompiledVerifyingKey(&vk); err != nil {
		return err
	}
	if err := validateCompiledStatement(vk.Metadata, vk.Verifier.PublicInputPlacement, statement); err != nil {
		return err
	}
	roundCount := bits.Len(uint(vk.Metadata.PartitionCount)) - 1
	if len(proof.SumCheckRounds) != roundCount {
		return fmt.Errorf("%w: got %d SumCheck rounds, want %d", ErrInvalidCompiledProof, len(proof.SumCheckRounds), roundCount)
	}
	if err := validateCompiledProof(&proof); err != nil {
		return err
	}

	outer, err := NewOuterTranscript(statement.OuterContext)
	if err != nil {
		return fmt.Errorf("%w: outer transcript: %v", ErrInvalidCompiledStatement, err)
	}
	if err := outer.AppendW0(proof.W0); err != nil {
		return compiledVerificationError("W0", err)
	}
	initial, err := outer.DeriveInitialChallenges()
	if err != nil {
		return compiledVerificationError("initial challenges", err)
	}
	if err := outer.AppendW1(proof.W1); err != nil {
		return compiledVerificationError("W1", err)
	}
	lambda, err := outer.DeriveLambda()
	if err != nil {
		return compiledVerificationError("lambda", err)
	}
	if err := outer.AppendW2(proof.W2); err != nil {
		return compiledVerificationError("W2", err)
	}
	alphaOut, err := outer.DeriveAlpha()
	if err != nil {
		return compiledVerificationError("alpha", err)
	}
	alpha := alphaOut.Value
	if err := outer.MarkRetainedW3Complete(); err != nil {
		return compiledVerificationError("retained W3 boundary", err)
	}
	if err := outer.AppendProductCheckCommitments(proof.ProductCheck); err != nil {
		return compiledVerificationError("ProductCheck commitments", err)
	}
	zetaTheta, err := outer.DeriveZetaTheta()
	if err != nil {
		return compiledVerificationError("zeta/theta", err)
	}
	theta := make([]fr.Element, len(zetaTheta.Theta))
	for coordinate := range zetaTheta.Theta {
		theta[coordinate] = zetaTheta.Theta[coordinate].Value
	}
	roundChallenges := make([]fr.Element, roundCount)
	sumCheckTranscript := SumCheckTranscript{Rounds: make([][]fr.Element, roundCount)}
	for round := 0; round < roundCount; round++ {
		if err := outer.AppendSumCheckRound(proof.SumCheckRounds[round]); err != nil {
			return compiledVerificationError(fmt.Sprintf("SumCheck round %d", round), err)
		}
		challenge, err := outer.DeriveSumCheckChallenge()
		if err != nil {
			return compiledVerificationError(fmt.Sprintf("SumCheck challenge %d", round), err)
		}
		roundChallenges[round] = challenge.Value
		sumCheckTranscript.Rounds[round] = append(
			[]fr.Element(nil), proof.SumCheckRounds[round].Coefficients[:]...,
		)
	}
	if err := outer.AppendFinalEvaluations(proof.FinalEvaluations); err != nil {
		return compiledVerificationError("final evaluations", err)
	}
	muOut, err := outer.DeriveSourceCompressionChallenges()
	if err != nil {
		return compiledVerificationError("source compression challenges", err)
	}
	var mu [LocalCompressedSourceCount]fr.Element
	for group := range muOut {
		mu[group] = muOut[group].Value
	}

	folded := compiledTerminalsFromFinalMessage(proof.FinalEvaluations)
	publicInputAtR, err := vk.Verifier.PublicInputPlacement.FoldAtPartitionPoint(
		statement.PublicSolutionPrefix, alpha, roundChallenges,
	)
	if err != nil {
		return fmt.Errorf("%w: public input at r: %v", ErrInvalidCompiledStatement, err)
	}
	omega := vk.Metadata.LocalDomainGenerator
	xStar := compiledProtocolInverse(omega)
	localChallenges := LocalPIOPChallenges{
		EtaPart: initial.EtaPart.Value,
		EtaX:    initial.EtaX.Value,
		Gamma:   initial.Gamma.Value,
		Lambda:  lambda.Value,
	}
	endpointContext := OuterPIOPEndpointContext{
		M:          vk.Metadata.PartitionCount,
		T:          vk.Metadata.LocalDomainSize,
		Omega:      omega,
		XStar:      xStar,
		Alpha:      alpha,
		Zeta:       zetaTheta.Zeta.Value,
		Theta:      theta,
		Point:      roundChallenges,
		WireCosets: compiledProtocolWireCosets(),
		Challenges: localChallenges,
	}
	if err := VerifyOuterPIOPSumCheck(
		OuterPIOPSumCheckProof{
			Transcript:      sumCheckTranscript,
			FoldedCircuit:   folded,
			TreeEvaluations: proof.FinalEvaluations.ProductCheck,
		},
		publicInputAtR,
		endpointContext,
	); err != nil {
		return compiledVerificationError("outer SumCheck endpoint", err)
	}

	publicCommitments := AggregateLocalPIOPPublicCommitments{
		Parties:    vk.Metadata.PartitionCount,
		DomainSize: vk.Metadata.LocalDomainSize,
		W0:         proof.W0.WitnessCommitments,
		W1:         proof.W1.AccumulatorCommitment,
		W2:         proof.W2.QuotientCommitments,
	}
	sourceCommitments, err := DeriveLocalTerminalSourceCommitments(
		AggregateLocalPIOPCommitmentView{
			Fixed:  vk.Verifier.FixedCommitments,
			Public: publicCommitments,
		},
		vk.Metadata.LocalDomainSize,
		compiledProtocolWireCosets(),
		LocalTerminalSourceCommitmentChallenges{
			EtaPart: localChallenges.EtaPart,
			EtaX:    localChallenges.EtaX,
			Gamma:   localChallenges.Gamma,
			Alpha:   alpha,
			Mu:      mu,
		},
	)
	if err != nil {
		return compiledVerificationError("source commitment derivation", err)
	}
	openingContext, err := outer.OpeningContext()
	if err != nil {
		return compiledVerificationError("opening context", err)
	}
	openingInstance := compiledOpeningInstance(
		vk.Metadata.LocalDomainSize,
		alpha,
		omega,
		xStar,
		vk.Metadata.IndexShift.Sigma,
		roundChallenges,
		sourceCommitments.Commitments,
		proof.ProductCheck.Commitments,
		folded,
		proof.FinalEvaluations.ProductCheck,
		mu,
		openingContext,
	)
	if err := VerifyHybridOpening(
		openingInstance,
		HybridOpeningProof{U0: proof.U0, U1: proof.U1, U2: proof.U2, U3: proof.U3},
		vk.Verifier.SRS,
	); err != nil {
		return compiledVerificationError("DLinKZG opening", err)
	}
	return nil
}

func validateCompiledProverInputs(setup *CompiledSetup, statement CompiledStatement, solution []fr.Element) error {
	if err := setup.ValidateOnlineRoles(); err != nil {
		return err
	}
	if err := validateCompiledStatement(setup.Metadata, setup.Coordinator.PublicInputPlacement, statement); err != nil {
		return err
	}
	publicVariables := setup.Coordinator.PublicInputPlacement.PublicVariables()
	if len(solution) < publicVariables {
		return fmt.Errorf("%w: solution has %d entries, shorter than public prefix %d", ErrInvalidCompiledProverInput, len(solution), publicVariables)
	}
	constraintSystem := setup.Parties[0].ConstraintSystem
	layout, err := validateSparseR1CSAdapterStructure(constraintSystem, 0, setup.Metadata.PartitionCount)
	if err != nil || layout != setup.Parties[0].Preprocessing.layout ||
		TranscriptDigest(sparseR1CSAdapterDigest(constraintSystem)) != setup.Metadata.CircuitDigest {
		return fmt.Errorf("%w: retained constraint system is not the authenticated setup circuit", ErrInvalidCompiledProverInput)
	}
	if len(solution) != layout.variables {
		return fmt.Errorf("%w: solution has %d variables, want %d", ErrInvalidCompiledProverInput, len(solution), layout.variables)
	}
	for index := 0; index < publicVariables; index++ {
		if !solution[index].Equal(&statement.PublicSolutionPrefix[index]) {
			return fmt.Errorf("%w: solution/public statement mismatch at index %d", ErrInvalidCompiledProverInput, index)
		}
	}
	return nil
}

// compiledBuildLocalPIOPTableFromSolution is the authenticated-setup online
// path. validateCompiledProverInputs has already checked the one shared
// SparseR1CS digest and layout exactly once. Repeating that O(|CS|) digest for
// every rank would make row extraction artificially O(M|CS|), so each party
// now performs only its solution-dependent row fill and fixed-table copy.
func compiledBuildLocalPIOPTableFromSolution(
	preprocessing *LocalPIOPPreprocessing,
	spr *cs.SparseR1CS,
	solution []fr.Element,
) (table LocalPIOPTable, index LocalPIOPIndex, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			table = LocalPIOPTable{}
			index = LocalPIOPIndex{}
			err = fmt.Errorf("%w: panic while building authenticated online table: %v", ErrInvalidCompiledProverInput, recovered)
		}
	}()
	if preprocessing == nil || spr == nil || len(solution) != preprocessing.layout.variables {
		return LocalPIOPTable{}, LocalPIOPIndex{}, fmt.Errorf("%w: malformed authenticated online table input", ErrInvalidCompiledProverInput)
	}
	table = allocateLocalPIOPAdapterTable(preprocessing.layout.localRows)
	copyLocalPIOPAdapterFixedTable(&table, preprocessing.fixed)
	fillLocalPIOPAdapterWitnessRows(&table, spr, solution, preprocessing.rank, preprocessing.layout)
	return table, preprocessing.index, nil
}

func validateCompiledStatement(
	metadata CompiledSetupMetadata,
	placement *StatementPublicInputPlacement,
	statement CompiledStatement,
) error {
	if placement == nil {
		return fmt.Errorf("%w: nil public-input placement", ErrInvalidCompiledStatement)
	}
	if len(statement.PublicSolutionPrefix) != placement.PublicVariables() {
		return fmt.Errorf(
			"%w: public prefix has length %d, want %d",
			ErrInvalidCompiledStatement, len(statement.PublicSolutionPrefix), placement.PublicVariables(),
		)
	}
	for index := range statement.PublicSolutionPrefix {
		if err := validateTranscriptField(&statement.PublicSolutionPrefix[index]); err != nil {
			return fmt.Errorf("%w: noncanonical public value %d", ErrInvalidCompiledStatement, index)
		}
	}
	wantStatementDigest := CompiledPublicStatementDigest(statement.PublicSolutionPrefix)
	if statement.OuterContext.PublicStatementDigest != wantStatementDigest {
		return fmt.Errorf("%w: public statement digest mismatch", ErrInvalidCompiledStatement)
	}
	if err := validateOuterTranscriptContext(statement.OuterContext); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidCompiledStatement, err)
	}
	context := statement.OuterContext
	if context.SRSDigest != metadata.SRSDigest ||
		context.IndexDigest != metadata.IndexDigest ||
		context.IndexPrefixDigest != metadata.IndexPrefixDigest ||
		context.PartyManifestDigest != metadata.PartyManifestDigest ||
		context.ShiftCounter != metadata.IndexShift.Counter ||
		!context.Shift.Equal(&metadata.IndexShift.Sigma) ||
		context.PartitionCount != uint64(metadata.PartitionCount) ||
		context.LocalDomainSize != uint64(metadata.LocalDomainSize) ||
		!context.LocalDomainGenerator.Equal(&metadata.LocalDomainGenerator) {
		return ErrCompiledSetupMetadataMismatch
	}
	if placement.Partitions() != metadata.PartitionCount ||
		placement.PublicVariables() != metadata.PublicVariableCount ||
		placement.localRows != metadata.LocalDomainSize ||
		!placement.omega.Equal(&metadata.LocalDomainGenerator) {
		return fmt.Errorf("%w: public-input placement metadata mismatch", ErrInvalidCompiledStatement)
	}
	return nil
}

func validateCompiledVerifyingKey(vk *CompiledVerifyingKey) error {
	if vk == nil {
		return fmt.Errorf("%w: nil verifier key", ErrInvalidCompiledSetup)
	}
	metadata := vk.Metadata
	if metadata.PartitionCount < 2 || !isPowerOfTwo(metadata.PartitionCount) ||
		metadata.LocalDomainSize < 4 || !isPowerOfTwo(metadata.LocalDomainSize) ||
		metadata.PartitionCount > metadata.LocalDomainSize ||
		metadata.LocalDomainSize > FastLocalPIOPMaxDomainSize ||
		metadata.PublicVariableCount < 0 || metadata.PublicVariableCount > metadata.LocalDomainSize ||
		metadata.CircuitDigest == (TranscriptDigest{}) ||
		metadata.IndexPrefixDigest == (TranscriptDigest{}) ||
		metadata.IndexDigest == (TranscriptDigest{}) ||
		metadata.SRSDigest == (TranscriptDigest{}) ||
		metadata.PartyManifestDigest == (TranscriptDigest{}) ||
		metadata.SetupDigest == (TranscriptDigest{}) {
		return fmt.Errorf("%w: malformed verifier metadata", ErrInvalidCompiledSetup)
	}
	omega, err := sparseR1CSAdapterRoot(metadata.LocalDomainSize)
	if err != nil || !omega.Equal(&metadata.LocalDomainGenerator) {
		return fmt.Errorf("%w: verifier local domain mismatch", ErrInvalidCompiledSetup)
	}
	if err := VerifyIndexShift(
		metadata.IndexPrefixDigest,
		uint64(metadata.LocalDomainSize),
		metadata.LocalDomainGenerator,
		metadata.IndexShift.Counter,
		metadata.IndexShift.Sigma,
	); err != nil {
		return fmt.Errorf("%w: verifier index shift: %v", ErrInvalidCompiledSetup, err)
	}
	if compiledSetupIndexDigest(metadata.IndexPrefixDigest, metadata.IndexShift) != metadata.IndexDigest {
		return fmt.Errorf("%w: verifier index digest", ErrCompiledSetupMetadataMismatch)
	}
	role := &vk.Verifier
	if role.Metadata != metadata || role.SRS == nil || role.SRS.Validate() != nil ||
		role.PublicInputPlacement == nil || role.PublicInputPlacement.Partitions() != metadata.PartitionCount ||
		role.PublicInputPlacement.PublicVariables() != metadata.PublicVariableCount ||
		role.PublicInputPlacement.localRows != metadata.LocalDomainSize ||
		!role.PublicInputPlacement.omega.Equal(&metadata.LocalDomainGenerator) ||
		role.FixedCommitments.Parties != metadata.PartitionCount ||
		role.FixedCommitments.DomainSize != metadata.LocalDomainSize {
		return fmt.Errorf("%w: malformed verifier role", ErrCompiledSetupMetadataMismatch)
	}
	if err := validateAggregateLocalPIOPFixedCommitments(role.FixedCommitments); err != nil {
		return fmt.Errorf("%w: verifier fixed commitments: %v", ErrInvalidCompiledSetup, err)
	}
	if !compiledVerifierSRSPointsValid(role.SRS) {
		return fmt.Errorf("%w: invalid verifier SRS points", ErrInvalidCompiledSetup)
	}
	if compiledSetupMetadataDigest(metadata, role.FixedCommitments) != metadata.SetupDigest {
		return fmt.Errorf("%w: verifier setup digest", ErrCompiledSetupMetadataMismatch)
	}
	return nil
}

func compiledOpeningInstance(
	t int,
	alpha, omega, xStar, shift fr.Element,
	partitionPoint []fr.Element,
	sourceCommitments [LocalCompressedSourceCount]bn254.G1Affine,
	treeCommitments [2]bn254.G1Affine,
	folded LocalTerminalEvaluations,
	treeClaims [OuterProductCheckClaimCount]fr.Element,
	mu [LocalCompressedSourceCount]fr.Element,
	context TranscriptContext,
) DLinkZGOpeningInstance {
	instance := DLinkZGOpeningInstance{
		LocalDomainSize:   t,
		SourceCommitments: sourceCommitments,
		TreeCommitments:   treeCommitments,
		TreeClaims:        treeClaims,
		Shift:             shift,
		PartitionPoint:    append([]fr.Element(nil), partitionPoint...),
		TranscriptContext: context,
	}
	instance.SemanticQueryPoints[LocalTerminalAtAlpha] = alpha
	instance.SemanticQueryPoints[LocalTerminalAtOmegaAlpha].Mul(&omega, &alpha)
	instance.SemanticQueryPoints[LocalTerminalAtXStar] = xStar
	for group := 0; group < LocalCompressedSourceCount; group++ {
		instance.CircuitClaims[group] = compiledProtocolPowerFold(
			folded.ValuesAt(LocalTerminalPoint(group)), mu[group],
		)
	}
	instance.TranscriptContext = BindHybridOpeningTranscriptContext(instance, context)
	return instance
}

func compiledTerminalsFromFinalMessage(message OuterFinalEvaluationsMessage) LocalTerminalEvaluations {
	var result LocalTerminalEvaluations
	offset := 0
	copy(result.Alpha[:], message.Terminal[offset:offset+LocalTerminalAlphaCount])
	offset += LocalTerminalAlphaCount
	copy(result.OmegaAlpha[:], message.Terminal[offset:offset+LocalTerminalOmegaAlphaCount])
	offset += LocalTerminalOmegaAlphaCount
	copy(result.XStar[:], message.Terminal[offset:offset+LocalTerminalXStarCount])
	return result
}

func compiledProtocolPowerFold(values []fr.Element, challenge fr.Element) fr.Element {
	var result fr.Element
	power := fr.One()
	for index := range values {
		var term fr.Element
		term.Mul(&power, &values[index])
		result.Add(&result, &term)
		power.Mul(&power, &challenge)
	}
	return result
}

func compiledProtocolWireCosets() [LocalWireCount]fr.Element {
	var result [LocalWireCount]fr.Element
	result[LocalWireA].SetOne()
	result[LocalWireB].SetUint64(5)
	result[LocalWireC].Square(&result[LocalWireB])
	return result
}

func compiledProtocolInverse(value fr.Element) fr.Element {
	var result fr.Element
	result.Inverse(&value)
	return result
}

func compiledVerificationError(stage string, err error) error {
	return fmt.Errorf("%w: %s: %w", ErrCompiledVerification, stage, err)
}
