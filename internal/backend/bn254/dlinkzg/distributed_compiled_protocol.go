package dlinkzg

// This file is the role-local M+1 execution of the compiled protocol. MPI
// rank zero owns only CompiledCoordinatorSetup; rank r in [1,M] owns exactly
// party slot r-1. Every rank has the public verifier key and replays both
// Fiat--Shamir transcripts locally. Only rank zero returns a proof.

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

var (
	// ErrInvalidCompiledMPIRole reports a role which is not exclusive, is
	// assigned to the wrong MPI rank, or is not bound to the common VK.
	ErrInvalidCompiledMPIRole = errors.New("dlinkzg: invalid compiled MPI role")
	// ErrCompiledMPIProtocol reports a local arithmetic, payload, transcript,
	// or coordinator assembly failure in the fixed 17-operation execution.
	ErrCompiledMPIProtocol = errors.New("dlinkzg: compiled MPI protocol failed")
)

// CompiledMPIRole is exclusive. Rank zero sets Coordinator only; every other
// rank sets Party only. It intentionally cannot contain a complete setup.
type CompiledMPIRole struct {
	Party       *CompiledPartySetup
	Coordinator *CompiledCoordinatorSetup
}

// compiledMPITrace is test/diagnostic state, never a proof field. It lets the
// fake-network suite assert that all M+1 ranks replayed identical transcripts.
type compiledMPITrace struct {
	Outer   TranscriptDigest
	Opening TranscriptDigest
}

type compiledMPIOuterSuffix struct {
	productCheck    OuterProductCheckCommitmentsMessage
	rounds          []OuterSumCheckRoundMessage
	final           OuterFinalEvaluationsMessage
	t0              []fr.Element
	t1              []fr.Element
	roundChallenges []fr.Element
	mu              [LocalCompressedSourceCount]fr.Element
}

type compiledMPIOpeningPartyState struct {
	local               dlinkzgOpeningLocalState
	localSAtBeta        fr.Element
	localSAtBetaInverse fr.Element
}

// ProvisionCompiledMPIRole extracts one value-copied online role from a setup
// that has already passed full Validate at generation/load time. It performs
// the cheap online boundary check again, returns the common VK, and never
// exposes the setup's complete party slice through the returned role.
func ProvisionCompiledMPIRole(
	setup *CompiledSetup,
	mpiRank uint64,
) (CompiledVerifyingKey, CompiledMPIRole, error) {
	if err := setup.ValidateOnlineRoles(); err != nil {
		return CompiledVerifyingKey{}, CompiledMPIRole{}, err
	}
	if mpiRank > uint64(setup.Metadata.PartitionCount) {
		return CompiledVerifyingKey{}, CompiledMPIRole{}, fmt.Errorf(
			"%w: MPI rank %d exceeds coordinator-plus-parties world M+1=%d",
			ErrInvalidCompiledMPIRole, mpiRank, setup.Metadata.PartitionCount+1,
		)
	}
	vk, err := setup.VerifyingKey()
	if err != nil {
		return CompiledVerifyingKey{}, CompiledMPIRole{}, err
	}
	if mpiRank == 0 {
		coordinator := setup.Coordinator
		return vk, CompiledMPIRole{Coordinator: &coordinator}, nil
	}
	party := setup.Parties[mpiRank-1]
	return vk, CompiledMPIRole{Party: &party}, nil
}

// CompiledProveMPI executes the exact ProtocolMPIChannel schedule. A non-nil
// proof is returned only on coordinator rank zero. If any rank returns an
// error, the caller must abort the whole communicator/job; this fixed 17-op
// protocol does not add a separate failure-propagation collective.
func CompiledProveMPI(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
) (*CompiledProof, error) {
	proof, _, err := compiledProveMPIWithTrace(channel, vk, statement, role, partySolution)
	return proof, err
}

func compiledProveMPIWithTrace(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
) (*CompiledProof, compiledMPITrace, error) {
	var trace compiledMPITrace
	if err := validateCompiledMPIRole(channel, vk, statement, role, partySolution); err != nil {
		return nil, trace, err
	}
	isRoot := channel.IsCoordinator()
	m := vk.Metadata.PartitionCount
	t := vk.Metadata.LocalDomainSize
	partitions := uint64(m)
	outer, err := NewOuterTranscript(statement.OuterContext)
	if err != nil {
		return nil, trace, compiledMPIError("outer transcript", err)
	}
	var publicProof CompiledProof
	var fastState *FastLocalPIOPState
	var relation *LocalPIOPRelation

	// W0: one semantic row commitment share per party, then the aggregate is
	// broadcast so every rank samples the same initial challenge batch.
	w0Share := MPIPayload{}
	if !isRoot {
		table, _, buildErr := compiledBuildLocalPIOPTableFromSolution(
			role.Party.Preprocessing, role.Party.ConstraintSystem, partySolution,
		)
		if buildErr != nil {
			return nil, trace, compiledMPIError("W0 local table", buildErr)
		}
		fastState, err = PrepareFastLocalPIOPWithFixed(table, role.Party.FastFixed)
		if err != nil {
			return nil, trace, compiledMPIError("W0 arithmetic", err)
		}
		w0, stateErr := fastState.W0()
		if stateErr != nil {
			return nil, trace, compiledMPIError("W0 state", stateErr)
		}
		var share OuterW0Message
		for wire := 0; wire < LocalWireCount; wire++ {
			share.WitnessCommitments[wire], err = role.Party.RowSRS.CommitSemantic(w0.Wires[wire])
			if err != nil {
				return nil, trace, compiledMPIError("W0 commitment", err)
			}
		}
		w0Share, err = PackProtocolW0(share)
		if err != nil {
			return nil, trace, compiledMPIError("W0 payload", err)
		}
	}
	w0Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW0, protocolMPIOpAggregate, partitions)
	w0Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW0, w0Shape, w0Share)
	if err != nil {
		return nil, trace, compiledMPIError("W0 aggregate", err)
	}
	w0RootPayload := MPIPayload{}
	if isRoot {
		publicProof.W0, err = UnpackProtocolW0(w0Aggregate)
		if err == nil {
			w0RootPayload, err = PackProtocolW0(publicProof.W0)
		}
		if err != nil {
			return nil, trace, compiledMPIError("W0 root assembly", err)
		}
	}
	w0BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW0, protocolMPIOpBroadcast, partitions)
	w0Public, err := channel.RootBroadcast(ProtocolMPIPhaseW0, w0BroadcastShape, w0RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("W0 broadcast", err)
	}
	publicProof.W0, err = UnpackProtocolW0(w0Public)
	if err != nil {
		return nil, trace, compiledMPIError("W0 decode", err)
	}
	if err := outer.AppendW0(publicProof.W0); err != nil {
		return nil, trace, compiledMPIError("W0 transcript", err)
	}
	initial, err := outer.DeriveInitialChallenges()
	if err != nil {
		return nil, trace, compiledMPIError("initial challenges", err)
	}
	localChallenges := LocalPIOPChallenges{
		EtaPart: initial.EtaPart.Value,
		EtaX:    initial.EtaX.Value,
		Gamma:   initial.Gamma.Value,
	}

	// W1: the local accumulator is fixed before lambda.
	w1Share := MPIPayload{}
	if !isRoot {
		w1, buildErr := fastState.BuildAccumulator(
			localChallenges.EtaPart, localChallenges.EtaX, localChallenges.Gamma,
		)
		if buildErr != nil {
			return nil, trace, compiledMPIError("W1 arithmetic", buildErr)
		}
		commitment, commitErr := role.Party.RowSRS.CommitSemantic(w1.Accumulator)
		if commitErr != nil {
			return nil, trace, compiledMPIError("W1 commitment", commitErr)
		}
		w1Share, err = PackProtocolW1(OuterW1Message{AccumulatorCommitment: commitment})
		if err != nil {
			return nil, trace, compiledMPIError("W1 payload", err)
		}
	}
	w1Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW1, protocolMPIOpAggregate, partitions)
	w1Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW1, w1Shape, w1Share)
	if err != nil {
		return nil, trace, compiledMPIError("W1 aggregate", err)
	}
	w1RootPayload := MPIPayload{}
	if isRoot {
		publicProof.W1, err = UnpackProtocolW1(w1Aggregate)
		if err == nil {
			w1RootPayload, err = PackProtocolW1(publicProof.W1)
		}
		if err != nil {
			return nil, trace, compiledMPIError("W1 root assembly", err)
		}
	}
	w1BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW1, protocolMPIOpBroadcast, partitions)
	w1Public, err := channel.RootBroadcast(ProtocolMPIPhaseW1, w1BroadcastShape, w1RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("W1 broadcast", err)
	}
	publicProof.W1, err = UnpackProtocolW1(w1Public)
	if err != nil {
		return nil, trace, compiledMPIError("W1 decode", err)
	}
	if err := outer.AppendW1(publicProof.W1); err != nil {
		return nil, trace, compiledMPIError("W1 transcript", err)
	}
	lambda, err := outer.DeriveLambda()
	if err != nil {
		return nil, trace, compiledMPIError("lambda", err)
	}
	localChallenges.Lambda = lambda.Value

	// W2: three quotient chunks per party.
	w2Share := MPIPayload{}
	if !isRoot {
		relation, err = fastState.BuildQuotient(localChallenges.Lambda)
		if err != nil {
			return nil, trace, compiledMPIError("W2 arithmetic", err)
		}
		var share OuterW2Message
		for chunk := range relation.Quotient {
			share.QuotientCommitments[chunk], err = role.Party.RowSRS.CommitSemantic(relation.Quotient[chunk])
			if err != nil {
				return nil, trace, compiledMPIError("W2 commitment", err)
			}
		}
		w2Share, err = PackProtocolW2(share)
		if err != nil {
			return nil, trace, compiledMPIError("W2 payload", err)
		}
	}
	w2Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW2, protocolMPIOpAggregate, partitions)
	w2Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW2, w2Shape, w2Share)
	if err != nil {
		return nil, trace, compiledMPIError("W2 aggregate", err)
	}
	w2RootPayload := MPIPayload{}
	if isRoot {
		publicProof.W2, err = UnpackProtocolW2(w2Aggregate)
		if err == nil {
			w2RootPayload, err = PackProtocolW2(publicProof.W2)
		}
		if err != nil {
			return nil, trace, compiledMPIError("W2 root assembly", err)
		}
	}
	w2BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW2, protocolMPIOpBroadcast, partitions)
	w2Public, err := channel.RootBroadcast(ProtocolMPIPhaseW2, w2BroadcastShape, w2RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("W2 broadcast", err)
	}
	publicProof.W2, err = UnpackProtocolW2(w2Public)
	if err != nil {
		return nil, trace, compiledMPIError("W2 decode", err)
	}
	if err := outer.AppendW2(publicProof.W2); err != nil {
		return nil, trace, compiledMPIError("W2 transcript", err)
	}
	alphaOut, err := outer.DeriveAlpha()
	if err != nil {
		return nil, trace, compiledMPIError("alpha", err)
	}
	alpha := alphaOut.Value

	// W3: exact 21-field rows are gathered. The root scatters only each
	// party's t0/t1 coefficients, then broadcasts the complete public suffix.
	w3Share := MPIPayload{}
	if !isRoot {
		terminals, terminalErr := relation.TerminalEvaluations(alpha)
		if terminalErr != nil {
			return nil, trace, compiledMPIError("W3 terminals", terminalErr)
		}
		w3Share, err = PackProtocolW3Gather(ProtocolW3GatherRecord{Terminals: terminals})
		if err != nil {
			return nil, trace, compiledMPIError("W3 gather payload", err)
		}
	}
	w3GatherShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpGather, partitions)
	w3Gathered, err := channel.GatherWorkers(ProtocolMPIPhaseW3, w3GatherShape, w3Share)
	if err != nil {
		return nil, trace, compiledMPIError("W3 gather", err)
	}
	var outerSuffix compiledMPIOuterSuffix
	if isRoot {
		terminalRows := make([]LocalTerminalEvaluations, m)
		for slot := range w3Gathered {
			record, decodeErr := UnpackProtocolW3Gather(w3Gathered[slot])
			if decodeErr != nil {
				return nil, trace, compiledMPIError("W3 gathered row", decodeErr)
			}
			terminalRows[slot] = record.Terminals
		}
		outerSuffix, err = compiledMPICoordinatorBuildOuterSuffix(
			role.Coordinator, statement, terminalRows, localChallenges, alpha, outer,
		)
		if err != nil {
			return nil, trace, compiledMPIError("W3 coordinator reduction", err)
		}
	}
	var scatterPayloads []MPIPayload
	if isRoot {
		scatterPayloads = make([]MPIPayload, m)
		for slot := 0; slot < m; slot++ {
			scatterPayloads[slot], err = PackProtocolW3Scatter(ProtocolW3ScatterRecord{
				T0: outerSuffix.t0[slot], T1: outerSuffix.t1[slot],
			})
			if err != nil {
				return nil, trace, compiledMPIError("W3 scatter assembly", err)
			}
		}
	}
	w3ScatterShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpScatter, partitions)
	w3LocalPairPayload, err := channel.RootScatter(ProtocolMPIPhaseW3, w3ScatterShape, scatterPayloads)
	if err != nil {
		return nil, trace, compiledMPIError("W3 scatter", err)
	}
	var localTreePair ProtocolW3ScatterRecord
	if !isRoot {
		localTreePair, err = UnpackProtocolW3Scatter(w3LocalPairPayload)
		if err != nil {
			return nil, trace, compiledMPIError("W3 scatter decode", err)
		}
	}
	w3RootPayload := MPIPayload{}
	if isRoot {
		w3RootPayload, err = PackProtocolW3Broadcast(ProtocolW3BroadcastRecord{
			ProductCheck:     outerSuffix.productCheck,
			SumCheckRounds:   outerSuffix.rounds,
			FinalEvaluations: outerSuffix.final,
		}, partitions)
		if err != nil {
			return nil, trace, compiledMPIError("W3 broadcast assembly", err)
		}
	}
	w3BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpBroadcast, partitions)
	w3PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseW3, w3BroadcastShape, w3RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("W3 broadcast", err)
	}
	w3Public, err := UnpackProtocolW3Broadcast(w3PublicPayload, partitions)
	if err != nil {
		return nil, trace, compiledMPIError("W3 broadcast decode", err)
	}
	publicProof.ProductCheck = w3Public.ProductCheck
	publicProof.SumCheckRounds = w3Public.SumCheckRounds
	publicProof.FinalEvaluations = w3Public.FinalEvaluations
	var roundChallenges []fr.Element
	var mu [LocalCompressedSourceCount]fr.Element
	if isRoot {
		roundChallenges = outerSuffix.roundChallenges
		mu = outerSuffix.mu
	} else {
		roundChallenges, mu, err = compiledMPIReplayOuterSuffix(
			outer, w3Public.ProductCheck, w3Public.SumCheckRounds, w3Public.FinalEvaluations,
		)
		if err != nil {
			return nil, trace, compiledMPIError("W3 transcript replay", err)
		}
	}
	trace.Outer = TranscriptDigest(outer.Digest())

	// Build the common opening instance. Parties additionally retain only
	// their three local compressed source polynomials.
	publicCommitments := AggregateLocalPIOPPublicCommitments{
		Parties: m, DomainSize: t,
		W0: publicProof.W0.WitnessCommitments,
		W1: publicProof.W1.AccumulatorCommitment,
		W2: publicProof.W2.QuotientCommitments,
	}
	sourceCommitments, err := DeriveLocalTerminalSourceCommitments(
		AggregateLocalPIOPCommitmentView{Fixed: vk.Verifier.FixedCommitments, Public: publicCommitments},
		t,
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
		return nil, trace, compiledMPIError("source commitment derivation", err)
	}
	folded := compiledTerminalsFromFinalMessage(publicProof.FinalEvaluations)
	openingContext, err := outer.OpeningContext()
	if err != nil {
		return nil, trace, compiledMPIError("opening context", err)
	}
	omega := vk.Metadata.LocalDomainGenerator
	openingInstance := compiledOpeningInstance(
		t, alpha, omega, compiledProtocolInverse(omega), vk.Metadata.IndexShift.Sigma,
		roundChallenges, sourceCommitments.Commitments, publicProof.ProductCheck.Commitments,
		folded, publicProof.FinalEvaluations.ProductCheck, mu, openingContext,
	)
	prepared, err := compiledMPIPrepareOpening(openingInstance)
	if err != nil {
		return nil, trace, compiledMPIError("opening instance", err)
	}
	var sources [LocalCompressedSourceCount][]fr.Element
	if !isRoot {
		compression, compressionErr := CompressLocalTerminalSources(relation, alpha, mu)
		if compressionErr != nil {
			return nil, trace, compiledMPIError("local source compression", compressionErr)
		}
		for group := range sources {
			sources[group] = compression.Polynomials[group]
		}
	}
	openingTranscript, err := NewOpeningTranscript(openingInstance.TranscriptContext)
	if err != nil {
		return nil, trace, compiledMPIError("opening transcript", err)
	}

	// U0: weighted translated local source commitments.
	var openingPartyState compiledMPIOpeningPartyState
	u0Share := MPIPayload{}
	if !isRoot {
		var share U0Message
		openingPartyState, share, err = compiledMPIOpeningU0Share(
			role.Party, sources, openingInstance, prepared,
		)
		if err != nil {
			return nil, trace, compiledMPIError("U0 local", err)
		}
		u0Share, err = PackProtocolU0(share)
		if err != nil {
			return nil, trace, compiledMPIError("U0 payload", err)
		}
	}
	u0Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU0, protocolMPIOpAggregate, partitions)
	u0Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseU0, u0Shape, u0Share)
	if err != nil {
		return nil, trace, compiledMPIError("U0 aggregate", err)
	}
	u0RootPayload := MPIPayload{}
	if isRoot {
		publicProof.U0, err = UnpackProtocolU0(u0Aggregate)
		if err == nil {
			u0RootPayload, err = PackProtocolU0(publicProof.U0)
		}
		if err != nil {
			return nil, trace, compiledMPIError("U0 root assembly", err)
		}
	}
	u0BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU0, protocolMPIOpBroadcast, partitions)
	u0PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU0, u0BroadcastShape, u0RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U0 broadcast", err)
	}
	publicProof.U0, err = UnpackProtocolU0(u0PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U0 decode", err)
	}
	if err := openingTranscript.AppendU0(publicProof.U0); err != nil {
		return nil, trace, compiledMPIError("U0 transcript", err)
	}
	xiOut, err := openingTranscript.DeriveXi()
	if err != nil {
		return nil, trace, compiledMPIError("xi", err)
	}
	nuOut, err := openingTranscript.DeriveNu()
	if err != nil {
		return nil, trace, compiledMPIError("nu", err)
	}
	zOut, err := openingTranscript.DeriveZChallenge()
	if err != nil {
		return nil, trace, compiledMPIError("z challenge", err)
	}
	xi, nu, zChallenge := xiOut.Value, nuOut.Value, zOut.Value

	// U1: parties reveal only d_i[3] and one Laurent commitment. The root
	// reconstructs h_xi coefficients and both public commitments.
	u1Share := MPIPayload{}
	if !isRoot {
		d, laurentCommitment, shareErr := compiledMPIOpeningU1Share(
			role.Party, &openingPartyState, localTreePair.T0, localTreePair.T1,
			prepared, xi, nu, zChallenge,
		)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U1 local", shareErr)
		}
		u1Share, err = PackProtocolU1Gather(ProtocolU1GatherRecord{D: d, LaurentCommitment: laurentCommitment})
		if err != nil {
			return nil, trace, compiledMPIError("U1 gather payload", err)
		}
	}
	u1GatherShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU1, protocolMPIOpGather, partitions)
	u1Gathered, err := channel.GatherWorkers(ProtocolMPIPhaseU1, u1GatherShape, u1Share)
	if err != nil {
		return nil, trace, compiledMPIError("U1 gather", err)
	}
	var hCoefficients []fr.Element
	u1RootPayload := MPIPayload{}
	if isRoot {
		hCoefficients = make([]fr.Element, m)
		var message U1Message
		for slot := range u1Gathered {
			record, decodeErr := UnpackProtocolU1Gather(u1Gathered[slot])
			if decodeErr != nil {
				return nil, trace, compiledMPIError("U1 gathered record", decodeErr)
			}
			xiPower := fr.One()
			for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
				var term fr.Element
				term.Mul(&prepared.weights[slot], &record.D[claim])
				message.LinkEvaluations[claim].Add(&message.LinkEvaluations[claim], &term)
				term.Mul(&xiPower, &record.D[claim])
				hCoefficients[slot].Add(&hCoefficients[slot], &term)
				xiPower.Mul(&xiPower, &xi)
			}
			sourceCommitmentAdd(&message.LaurentCommitment, &record.LaurentCommitment)
		}
		message.LinkCommitment, err = role.Coordinator.SRS.CommitZ(hCoefficients)
		if err != nil {
			return nil, trace, compiledMPIError("U1 link commitment", err)
		}
		publicProof.U1 = message
		u1RootPayload, err = PackProtocolU1Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U1 broadcast assembly", err)
		}
	}
	u1BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU1, protocolMPIOpBroadcast, partitions)
	u1PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU1, u1BroadcastShape, u1RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U1 broadcast", err)
	}
	publicProof.U1, err = UnpackProtocolU1Broadcast(u1PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U1 decode", err)
	}
	if err := openingTranscript.AppendU1(publicProof.U1); err != nil {
		return nil, trace, compiledMPIError("U1 transcript", err)
	}
	betaOut, err := openingTranscript.DeriveBeta()
	if err != nil {
		return nil, trace, compiledMPIError("beta", err)
	}
	beta := betaOut.Value
	if err := validateDLinkZGOpeningBeta(beta, zChallenge); err != nil {
		return nil, trace, compiledMPIError("beta exclusion", err)
	}
	betaInverse := dlinkzgOpeningInverse(beta)

	// U2: six g evaluations and S_i(beta) are additive shares. The root
	// evaluates h_xi,t0,t1 itself and derives the canonical S(beta^-1).
	u2Share := MPIPayload{}
	if !isRoot {
		shareValues, shareErr := compiledMPIOpeningU2Share(&openingPartyState, beta)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U2 local", shareErr)
		}
		var record ProtocolU2AggregateRecord
		for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
			record.PartialAtBeta[claim] = shareValues[2*claim]
			record.PartialAtBetaInverse[claim] = shareValues[2*claim+1]
		}
		record.SAtBeta = shareValues[6]
		u2Share, err = PackProtocolU2Aggregate(record)
		if err != nil {
			return nil, trace, compiledMPIError("U2 aggregate payload", err)
		}
	}
	u2Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU2, protocolMPIOpAggregate, partitions)
	u2AggregatePayload, err := channel.AggregateWorkers(ProtocolMPIPhaseU2, u2Shape, u2Share)
	if err != nil {
		return nil, trace, compiledMPIError("U2 aggregate", err)
	}
	u2RootPayload := MPIPayload{}
	if isRoot {
		record, decodeErr := UnpackProtocolU2Aggregate(u2AggregatePayload)
		if decodeErr != nil {
			return nil, trace, compiledMPIError("U2 aggregate decode", decodeErr)
		}
		message := U2Message{
			PartialAtBeta:        record.PartialAtBeta,
			PartialAtBetaInverse: record.PartialAtBetaInverse,
		}
		message.BatchAtBeta[0] = cryptodlinkzg.Eval(hCoefficients, beta)
		message.BatchAtBetaInverse[0] = cryptodlinkzg.Eval(hCoefficients, betaInverse)
		message.BatchAtBeta[1] = cryptodlinkzg.Eval(outerSuffix.t0, beta)
		message.BatchAtBetaInverse[1] = cryptodlinkzg.Eval(outerSuffix.t0, betaInverse)
		message.BatchAtBeta[2] = cryptodlinkzg.Eval(outerSuffix.t1, beta)
		message.BatchAtBetaInverse[2] = cryptodlinkzg.Eval(outerSuffix.t1, betaInverse)
		message.BatchAtBeta[3] = record.SAtBeta
		left := dlinkzgOpeningLaurentLeft(message, prepared, xi, nu, beta)
		target := dlinkzgOpeningLinearTarget(openingInstance, publicProof.U1.LinkEvaluations, xi, nu)
		message.BatchAtBetaInverse[3], err = cryptodlinkzg.DeriveLaurentInverseValue(
			beta, left, target, message.BatchAtBeta[3],
		)
		if err != nil {
			return nil, trace, compiledMPIError("U2 inverse derivation", err)
		}
		publicProof.U2 = message
		u2RootPayload, err = PackProtocolU2Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U2 broadcast assembly", err)
		}
	}
	u2BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU2, protocolMPIOpBroadcast, partitions)
	u2PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU2, u2BroadcastShape, u2RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U2 broadcast", err)
	}
	publicProof.U2, err = UnpackProtocolU2Broadcast(u2PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U2 decode", err)
	}
	if err := openingTranscript.AppendU2(publicProof.U2); err != nil {
		return nil, trace, compiledMPIError("U2 transcript", err)
	}
	if err := verifyDLinkZGOpeningScalarIdentity(
		openingInstance, publicProof.U1, publicProof.U2, prepared, xi, nu, beta,
	); err != nil {
		return nil, trace, compiledMPIError("U2 scalar identity", err)
	}
	kappaOut, err := openingTranscript.DeriveKappa()
	if err != nil {
		return nil, trace, compiledMPIError("kappa", err)
	}
	kappa := kappaOut.Value
	if kappa.IsZero() {
		return nil, trace, compiledMPIError("kappa", ErrDLinkZGOpeningRejected)
	}

	// U3: parties aggregate WG,WL,PiZ; PiY is coordinator-only.
	u3Share := MPIPayload{}
	if !isRoot {
		shares, shareErr := compiledMPIOpeningU3Share(
			role.Party, &openingPartyState, prepared, xi, zChallenge, beta, kappa,
		)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U3 local", shareErr)
		}
		u3Share, err = PackProtocolU3Aggregate(ProtocolU3AggregateRecord{
			WG: shares[0], WL: shares[1], PiZ: shares[2],
		})
		if err != nil {
			return nil, trace, compiledMPIError("U3 aggregate payload", err)
		}
	}
	u3Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU3, protocolMPIOpAggregate, partitions)
	u3AggregatePayload, err := channel.AggregateWorkers(ProtocolMPIPhaseU3, u3Shape, u3Share)
	if err != nil {
		return nil, trace, compiledMPIError("U3 aggregate", err)
	}
	u3RootPayload := MPIPayload{}
	if isRoot {
		record, decodeErr := UnpackProtocolU3Aggregate(u3AggregatePayload)
		if decodeErr != nil {
			return nil, trace, compiledMPIError("U3 aggregate decode", decodeErr)
		}
		message := U3Message{WG: record.WG, WL: record.WL, PiZ: record.PiZ}
		qY, sourceValue := cryptodlinkzg.SyntheticDivision(hCoefficients, beta)
		if !sourceValue.Equal(&publicProof.U2.BatchAtBeta[0]) {
			return nil, trace, compiledMPIError("U3 source-link value", ErrDLinkZGOpeningRejected)
		}
		message.PiY, err = role.Coordinator.SRS.CommitY(qY)
		if err != nil {
			return nil, trace, compiledMPIError("U3 PiY", err)
		}
		publicProof.U3 = message
		u3RootPayload, err = PackProtocolU3Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U3 broadcast assembly", err)
		}
	}
	u3BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU3, protocolMPIOpBroadcast, partitions)
	u3PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU3, u3BroadcastShape, u3RootPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U3 broadcast", err)
	}
	publicProof.U3, err = UnpackProtocolU3Broadcast(u3PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U3 decode", err)
	}
	if err := openingTranscript.AppendU3(publicProof.U3); err != nil {
		return nil, trace, compiledMPIError("U3 transcript", err)
	}
	deltaOut, err := openingTranscript.DeriveDelta()
	if err != nil {
		return nil, trace, compiledMPIError("delta", err)
	}
	if deltaOut.Value.IsZero() {
		return nil, trace, compiledMPIError("delta", ErrDLinkZGOpeningRejected)
	}
	trace.Opening = TranscriptDigest(openingTranscript.Digest())
	if isRoot {
		return &publicProof, trace, nil
	}
	return nil, trace, nil
}

func validateCompiledMPIRole(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
) error {
	if channel == nil {
		return fmt.Errorf("%w: nil channel", ErrInvalidCompiledMPIRole)
	}
	if err := validateCompiledVerifyingKey(&vk); err != nil {
		return err
	}
	if uint64(vk.Metadata.PartitionCount) != channel.PartitionCount() {
		return fmt.Errorf("%w: VK M=%d, channel M=%d", ErrInvalidCompiledMPIRole, vk.Metadata.PartitionCount, channel.PartitionCount())
	}
	if err := validateCompiledStatement(vk.Metadata, vk.Verifier.PublicInputPlacement, statement); err != nil {
		return err
	}
	if channel.IsCoordinator() {
		if role.Coordinator == nil || role.Party != nil || len(partySolution) != 0 {
			return fmt.Errorf("%w: rank zero must own only the coordinator role and no solution", ErrInvalidCompiledMPIRole)
		}
		coordinator := role.Coordinator
		if coordinator.Metadata != vk.Metadata || coordinator.SRS == nil ||
			coordinator.SRS.Validate() != nil || coordinator.SRS.Parties != vk.Metadata.PartitionCount ||
			coordinator.PublicInputPlacement == nil ||
			!compiledPlacementEqual(coordinator.PublicInputPlacement, vk.Verifier.PublicInputPlacement) ||
			!compiledAggregateFixedEqual(coordinator.FixedCommitments, vk.Verifier.FixedCommitments) {
			return fmt.Errorf("%w: coordinator role is not bound to the common VK", ErrInvalidCompiledMPIRole)
		}
		return nil
	}

	if role.Party == nil || role.Coordinator != nil {
		return fmt.Errorf("%w: party rank must own exactly one party role", ErrInvalidCompiledMPIRole)
	}
	slot, ok := channel.PartySlot()
	if !ok || slot > uint64(^uint(0)>>1) {
		return fmt.Errorf("%w: invalid party slot", ErrInvalidCompiledMPIRole)
	}
	party := role.Party
	if party.Rank != int(slot) || party.Metadata != vk.Metadata ||
		party.ConstraintSystem == nil || party.Preprocessing == nil || party.FastFixed == nil || party.RowSRS == nil ||
		party.Preprocessing.rank != int(slot) || party.Preprocessing.world != vk.Metadata.PartitionCount ||
		party.Preprocessing.layout.localRows != vk.Metadata.LocalDomainSize ||
		party.Preprocessing.layout.publicVariables != vk.Metadata.PublicVariableCount ||
		TranscriptDigest(party.Preprocessing.systemDigest) != vk.Metadata.CircuitDigest ||
		party.RowSRS.Rank != int(slot) || party.RowSRS.Parties != vk.Metadata.PartitionCount ||
		party.RowSRS.DegreeBound != vk.Metadata.LocalDomainSize || party.RowSRS.Validate() != nil ||
		party.FixedCommitments.Rank != int(slot) || party.FixedCommitments.Parties != vk.Metadata.PartitionCount ||
		party.FixedCommitments.DomainSize != vk.Metadata.LocalDomainSize {
		return fmt.Errorf("%w: malformed party role for slot %d", ErrInvalidCompiledMPIRole, slot)
	}
	if err := validateStatementPublicInputIndex(party.Preprocessing, int(slot)); err != nil {
		return fmt.Errorf("%w: party index: %v", ErrInvalidCompiledMPIRole, err)
	}
	if !compiledFastFixedShapeMatches(
		party.FastFixed, party.Preprocessing, vk.Metadata.LocalDomainSize, vk.Metadata.LocalDomainGenerator,
	) {
		return fmt.Errorf("%w: party fast-fixed role", ErrInvalidCompiledMPIRole)
	}
	layout, err := validateSparseR1CSAdapterStructure(
		party.ConstraintSystem, int(slot), vk.Metadata.PartitionCount,
	)
	if err != nil || layout != party.Preprocessing.layout ||
		TranscriptDigest(sparseR1CSAdapterDigest(party.ConstraintSystem)) != vk.Metadata.CircuitDigest {
		return fmt.Errorf("%w: party circuit is not the authenticated setup circuit", ErrInvalidCompiledMPIRole)
	}
	if len(partySolution) != layout.variables {
		return fmt.Errorf("%w: party solution has %d variables, want %d", ErrInvalidCompiledMPIRole, len(partySolution), layout.variables)
	}
	for index := range statement.PublicSolutionPrefix {
		if !partySolution[index].Equal(&statement.PublicSolutionPrefix[index]) {
			return fmt.Errorf("%w: party solution/public prefix mismatch at %d", ErrInvalidCompiledMPIRole, index)
		}
	}
	return nil
}

func compiledMPICoordinatorBuildOuterSuffix(
	coordinator *CompiledCoordinatorSetup,
	statement CompiledStatement,
	terminals []LocalTerminalEvaluations,
	localChallenges LocalPIOPChallenges,
	alpha fr.Element,
	outer *OuterTranscript,
) (compiledMPIOuterSuffix, error) {
	var result compiledMPIOuterSuffix
	m := coordinator.Metadata.PartitionCount
	t := coordinator.Metadata.LocalDomainSize
	publicInputAtAlpha, err := coordinator.PublicInputPlacement.EvaluationsAtAlpha(
		statement.PublicSolutionPrefix, alpha,
	)
	if err != nil {
		return result, err
	}
	omega := coordinator.Metadata.LocalDomainGenerator
	state, err := BuildOuterPIOPCoordinatorStateFromTerminals(
		terminals,
		publicInputAtAlpha,
		OuterPIOPReductionContext{
			DomainSize: t,
			Omega:      omega,
			XStar:      compiledProtocolInverse(omega),
			Alpha:      alpha,
			WireCosets: compiledProtocolWireCosets(),
			Challenges: localChallenges,
		},
	)
	if err != nil {
		return result, err
	}
	_, result.t0, result.t1 = state.CoordinatorProductCheckWitness()
	result.productCheck.Commitments[0], err = coordinator.SRS.CommitZ(result.t0)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	result.productCheck.Commitments[1], err = coordinator.SRS.CommitZ(result.t1)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	if err := outer.MarkRetainedW3Complete(); err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	if err := outer.AppendProductCheckCommitments(result.productCheck); err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	zetaTheta, err := outer.DeriveZetaTheta()
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	theta := make([]fr.Element, len(zetaTheta.Theta))
	for coordinate := range theta {
		theta[coordinate] = zetaTheta.Theta[coordinate].Value
	}
	instance, err := state.BuildSumCheckInstance(zetaTheta.Zeta.Value, theta)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	oracles, composition := instance.CoordinatorOracleTables()
	var zero fr.Element
	prover, err := NewDegreeFiveSumCheckProver(zero, oracles, composition)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	rounds := bits.Len(uint(m)) - 1
	result.rounds = make([]OuterSumCheckRoundMessage, rounds)
	result.roundChallenges = make([]fr.Element, rounds)
	sumCheckTranscript := SumCheckTranscript{Rounds: make([][]fr.Element, rounds)}
	for round := 0; round < rounds; round++ {
		coefficients, err := prover.NextRound()
		if err != nil {
			return compiledMPIOuterSuffix{}, err
		}
		result.rounds[round].Coefficients = coefficients
		sumCheckTranscript.Rounds[round] = append([]fr.Element(nil), coefficients[:]...)
		if err := outer.AppendSumCheckRound(result.rounds[round]); err != nil {
			return compiledMPIOuterSuffix{}, err
		}
		challenge, err := outer.DeriveSumCheckChallenge()
		if err != nil {
			return compiledMPIOuterSuffix{}, err
		}
		result.roundChallenges[round] = challenge.Value
		if err := prover.BindChallenge(challenge.Value); err != nil {
			return compiledMPIOuterSuffix{}, err
		}
	}
	finalOracles, err := prover.FinalEvaluations()
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	folded := outerPIOPTerminalsFromOracleValues(finalOracles)
	copy(result.final.Terminal[:], folded.Flatten())
	treePoints := productCheckFunctionalPoints(result.roundChallenges)
	for claim := 0; claim < OuterProductCheckClaimCount; claim++ {
		result.final.ProductCheck[claim], err = evaluateTreeFunctional(result.t0, result.t1, treePoints[claim])
		if err != nil {
			return compiledMPIOuterSuffix{}, err
		}
	}
	if err := outer.AppendFinalEvaluations(result.final); err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	muOut, err := outer.DeriveSourceCompressionChallenges()
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	for group := range muOut {
		result.mu[group] = muOut[group].Value
	}
	publicInputAtR, err := coordinator.PublicInputPlacement.FoldAtPartitionPoint(
		statement.PublicSolutionPrefix, alpha, result.roundChallenges,
	)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	if err := VerifyOuterPIOPSumCheck(
		OuterPIOPSumCheckProof{
			Transcript:      sumCheckTranscript,
			FoldedCircuit:   folded,
			TreeEvaluations: result.final.ProductCheck,
		},
		publicInputAtR,
		instance.EndpointContext(result.roundChallenges),
	); err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	return result, nil
}

func compiledMPIReplayOuterSuffix(
	outer *OuterTranscript,
	productCheck OuterProductCheckCommitmentsMessage,
	rounds []OuterSumCheckRoundMessage,
	final OuterFinalEvaluationsMessage,
) ([]fr.Element, [LocalCompressedSourceCount]fr.Element, error) {
	var mu [LocalCompressedSourceCount]fr.Element
	if err := outer.MarkRetainedW3Complete(); err != nil {
		return nil, mu, err
	}
	if err := outer.AppendProductCheckCommitments(productCheck); err != nil {
		return nil, mu, err
	}
	if _, err := outer.DeriveZetaTheta(); err != nil {
		return nil, mu, err
	}
	challenges := make([]fr.Element, len(rounds))
	for round := range rounds {
		if err := outer.AppendSumCheckRound(rounds[round]); err != nil {
			return nil, mu, err
		}
		challenge, err := outer.DeriveSumCheckChallenge()
		if err != nil {
			return nil, mu, err
		}
		challenges[round] = challenge.Value
	}
	if err := outer.AppendFinalEvaluations(final); err != nil {
		return nil, mu, err
	}
	muOut, err := outer.DeriveSourceCompressionChallenges()
	if err != nil {
		return nil, mu, err
	}
	for group := range muOut {
		mu[group] = muOut[group].Value
	}
	return challenges, mu, nil
}

func compiledMPIPrepareOpening(instance DLinkZGOpeningInstance) (dlinkzgOpeningPrepared, error) {
	prepared, err := prepareDLinkZGOpeningInstance(instance)
	if err != nil {
		return prepared, err
	}
	prepared.weights = cryptodlinkzg.EqualityWeights(instance.PartitionPoint)
	prepared.psiR = append([]fr.Element(nil), prepared.weights...)
	for claim := range prepared.psiQ {
		prepared.psiQ[claim] = cryptodlinkzg.EqualityWeights(prepared.queries[claim])
	}
	return prepared, nil
}

func compiledMPIOpeningU0Share(
	party *CompiledPartySetup,
	sources [LocalCompressedSourceCount][]fr.Element,
	instance DLinkZGOpeningInstance,
	prepared dlinkzgOpeningPrepared,
) (compiledMPIOpeningPartyState, U0Message, error) {
	var state compiledMPIOpeningPartyState
	var share U0Message
	rank := party.Rank
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		if len(sources[claim]) > prepared.t {
			return state, share, fmt.Errorf("%w: source %d exceeds T", ErrCompiledMPIProtocol, claim)
		}
		state.local.shifted[claim] = cryptodlinkzg.FastTaylorShift(sources[claim], instance.Shift)
		state.local.g[claim] = dlinkzgOpeningScalePolynomial(state.local.shifted[claim], prepared.weights[rank])
		commitment, err := party.RowSRS.CommitZ(state.local.g[claim])
		if err != nil {
			return compiledMPIOpeningPartyState{}, U0Message{}, err
		}
		share.PartialCommitments[claim] = commitment
	}
	return state, share, nil
}

func compiledMPIOpeningU1Share(
	party *CompiledPartySetup,
	state *compiledMPIOpeningPartyState,
	t0, t1 fr.Element,
	prepared dlinkzgOpeningPrepared,
	xi, nu, zChallenge fr.Element,
) ([dlinkzgOpeningCircuitClaims]fr.Element, bn254.G1Affine, error) {
	var d [dlinkzgOpeningCircuitClaims]fr.Element
	aWeights, bWeights := treeWeightPolynomials(prepared.tree, xi, prepared.t)
	xiPower := fr.One()
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		state.local.d[claim] = cryptodlinkzg.Eval(state.local.shifted[claim], zChallenge)
		d[claim] = state.local.d[claim]
		var term fr.Element
		term.Mul(&xiPower, &state.local.d[claim])
		state.local.dLink.Add(&state.local.dLink, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	state.local.h = dlinkzgOpeningMonomial(state.local.dLink, party.Rank)
	state.local.t0 = dlinkzgOpeningMonomial(t0, party.Rank)
	state.local.t1 = dlinkzgOpeningMonomial(t1, party.Rank)
	state.local.input = cryptodlinkzg.LocalLaurentInput{
		G: state.local.g, PsiQ: prepared.psiQ, P: prepared.scales, Xi: xi,
		HXi: state.local.h, PsiR: prepared.psiR, Nu: nu,
		T0: state.local.t0, T1: state.local.t1, AXi: aWeights, BXi: bWeights,
	}
	state.local.laurent = cryptodlinkzg.BuildLocalLaurent(state.local.input)
	commitment, err := party.RowSRS.CommitZ(state.local.laurent)
	if err != nil {
		return d, bn254.G1Affine{}, err
	}
	return d, commitment, nil
}

func compiledMPIOpeningU2Share(
	state *compiledMPIOpeningPartyState,
	beta fr.Element,
) ([7]fr.Element, error) {
	var result [7]fr.Element
	betaInverse := dlinkzgOpeningInverse(beta)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		result[2*claim] = cryptodlinkzg.Eval(state.local.g[claim], beta)
		result[2*claim+1] = cryptodlinkzg.Eval(state.local.g[claim], betaInverse)
	}
	state.localSAtBeta = cryptodlinkzg.Eval(state.local.laurent, beta)
	result[6] = state.localSAtBeta
	localLeft, err := cryptodlinkzg.EvalLocalLaurentLeft(state.local.input, beta)
	if err != nil {
		return result, err
	}
	state.localSAtBetaInverse, err = cryptodlinkzg.DeriveLaurentInverseValue(
		beta,
		localLeft,
		cryptodlinkzg.LocalLaurentDiagonal(state.local.input),
		state.localSAtBeta,
	)
	return result, err
}

func compiledMPIOpeningU3Share(
	party *CompiledPartySetup,
	state *compiledMPIOpeningPartyState,
	prepared dlinkzgOpeningPrepared,
	xi, zChallenge, beta, kappa fr.Element,
) ([3]bn254.G1Affine, error) {
	var result [3]bn254.G1Affine
	betaInverse := dlinkzgOpeningInverse(beta)
	gPoints := []fr.Element{zChallenge, beta, betaInverse}
	gInputs := make([]cryptodlinkzg.SameSetInput, dlinkzgOpeningCircuitClaims)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		var atZ fr.Element
		atZ.Mul(&prepared.weights[party.Rank], &state.local.d[claim])
		gInputs[claim] = cryptodlinkzg.SameSetInput{
			Polynomial: state.local.g[claim],
			ClaimedValues: []fr.Element{
				atZ,
				cryptodlinkzg.Eval(state.local.g[claim], beta),
				cryptodlinkzg.Eval(state.local.g[claim], betaInverse),
			},
		}
	}
	gBatch, err := cryptodlinkzg.BuildSameSetQuotient(gInputs, gPoints, kappa)
	if err != nil {
		return result, err
	}
	result[0], err = party.RowSRS.CommitZ(gBatch.Quotient)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}
	lPolynomials := [4][]fr.Element{state.local.h, state.local.t0, state.local.t1, state.local.laurent}
	lInputs := make([]cryptodlinkzg.SameSetInput, len(lPolynomials))
	for polynomial := range lPolynomials {
		values := []fr.Element{
			cryptodlinkzg.Eval(lPolynomials[polynomial], beta),
			cryptodlinkzg.Eval(lPolynomials[polynomial], betaInverse),
		}
		if polynomial == 3 {
			values[0] = state.localSAtBeta
			values[1] = state.localSAtBetaInverse
		}
		lInputs[polynomial] = cryptodlinkzg.SameSetInput{Polynomial: lPolynomials[polynomial], ClaimedValues: values}
	}
	lBatch, err := cryptodlinkzg.BuildSameSetQuotient(lInputs, []fr.Element{beta, betaInverse}, kappa)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}
	result[1], err = party.RowSRS.CommitZ(lBatch.Quotient)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}
	pXi := make([]fr.Element, prepared.t)
	xiPower := fr.One()
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		dlinkzgOpeningAddScaled(pXi, state.local.shifted[claim], xiPower)
		xiPower.Mul(&xiPower, &xi)
	}
	qZ, remainder := cryptodlinkzg.SyntheticDivision(pXi, zChallenge)
	if !remainder.Equal(&state.local.dLink) {
		return [3]bn254.G1Affine{}, fmt.Errorf("%w: local source-link remainder", ErrCompiledMPIProtocol)
	}
	result[2], err = party.RowSRS.CommitRow(qZ)
	return result, err
}

func compiledMPIError(stage string, err error) error {
	return fmt.Errorf("%w: %s: %w", ErrCompiledMPIProtocol, stage, err)
}
