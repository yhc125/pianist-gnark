package dlinkzg

// This file is the role-local M-rank execution of the compiled protocol. MPI
// rank zero owns party slot zero and CompiledCoordinatorSetup; rank r in
// [1,M) owns party slot r. Every rank has the public verifier key and replays
// both Fiat--Shamir transcripts locally. Only rank zero returns a proof.

import (
	"errors"
	"fmt"
	"math/bits"
	"time"

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

// CompiledMPIRole is rank-local. Every rank sets Party; rank zero additionally
// sets Coordinator. It intentionally cannot contain a complete setup.
type CompiledMPIRole struct {
	Party       *CompiledPartySetup
	Coordinator *CompiledCoordinatorSetup
}

// CompiledMPIProver is a setup-validated, rank-local online prover. The
// expensive circuit, SRS, fixed-table, and role checks are performed once by
// NewCompiledMPIProver. Prove still validates every statement and its binding
// to the rank-local solution before executing the transcript.
//
// The role and verifying key used to construct a CompiledMPIProver must be
// treated as immutable for the lifetime of the prover.
type CompiledMPIProver struct {
	channel *ProtocolMPIChannel
	vk      CompiledVerifyingKey
	role    CompiledMPIRole
}

// CompiledMPIPhaseTiming separates elapsed phase time from time spent inside
// transport collectives. On composite rank zero, Active is therefore an
// approximation of P_0 work plus coordinator computation/serialization, and
// Transport includes network service plus waiting for the other workers.
type CompiledMPIPhaseTiming struct {
	Total     time.Duration
	Transport time.Duration
}

// Active returns phase time outside transport calls.
func (timing CompiledMPIPhaseTiming) Active() time.Duration {
	if timing.Transport >= timing.Total {
		return 0
	}
	return timing.Total - timing.Transport
}

// CompiledMPIProfile is diagnostic state and is never serialized into a
// proof. Entries are indexed by ProtocolMPIPhase W0--U3.
type CompiledMPIProfile struct {
	phases [8]CompiledMPIPhaseTiming
}

// Phase returns one W0--U3 timing entry.
func (profile CompiledMPIProfile) Phase(phase ProtocolMPIPhase) (CompiledMPIPhaseTiming, bool) {
	index := int(phase)
	if index < 0 || index >= len(profile.phases) {
		return CompiledMPIPhaseTiming{}, false
	}
	return profile.phases[index], true
}

func (profile *CompiledMPIProfile) recordTotal(phase ProtocolMPIPhase, elapsed time.Duration) {
	if profile != nil {
		profile.phases[int(phase)].Total += elapsed
	}
}

func (profile *CompiledMPIProfile) recordTransport(phase ProtocolMPIPhase, elapsed time.Duration) {
	if profile != nil {
		profile.phases[int(phase)].Transport += elapsed
	}
}

func startCompiledMPITransportProfile(profile *CompiledMPIProfile) time.Time {
	if profile == nil {
		return time.Time{}
	}
	return time.Now()
}

func finishCompiledMPITransportProfile(
	profile *CompiledMPIProfile,
	phase ProtocolMPIPhase,
	started time.Time,
) {
	if profile != nil {
		profile.recordTransport(phase, time.Since(started))
	}
}

// compiledMPITrace is test/diagnostic state, never a proof field. It lets the
// fake-network suite asserts that all M ranks replay identical transcripts.
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
	local              hybridOpeningLocalState
	sources            [dlinkzgOpeningCircuitClaims][]fr.Element
	localCircuitClaims [dlinkzgOpeningCircuitClaims]fr.Element
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
	if mpiRank >= uint64(setup.Metadata.PartitionCount) {
		return CompiledVerifyingKey{}, CompiledMPIRole{}, fmt.Errorf(
			"%w: MPI rank %d exceeds M-rank party world=%d",
			ErrInvalidCompiledMPIRole, mpiRank, setup.Metadata.PartitionCount,
		)
	}
	vk, err := setup.VerifyingKey()
	if err != nil {
		return CompiledVerifyingKey{}, CompiledMPIRole{}, err
	}
	if mpiRank == 0 {
		coordinator := setup.Coordinator
		setup.Parties[0].FastFixed.preprocessExtendedSelectors()
		party := setup.Parties[0]
		return vk, CompiledMPIRole{Party: &party, Coordinator: &coordinator}, nil
	}
	setup.Parties[mpiRank].FastFixed.preprocessExtendedSelectors()
	party := setup.Parties[mpiRank]
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
	prover, err := NewCompiledMPIProver(channel, vk, role)
	if err != nil {
		return nil, err
	}
	return prover.Prove(statement, partySolution)
}

// NewCompiledMPIProver validates the static online role boundary once. It is
// intended to run immediately after setup/key loading and outside repeated
// online proving measurements.
func NewCompiledMPIProver(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	role CompiledMPIRole,
) (*CompiledMPIProver, error) {
	if err := validateCompiledMPIStaticRole(channel, vk, role); err != nil {
		return nil, err
	}
	return &CompiledMPIProver{channel: channel, vk: vk, role: role}, nil
}

// Prove validates the proof-specific statement and solution binding, then
// executes the fixed 17-operation protocol without repeating setup checks.
func (prover *CompiledMPIProver) Prove(
	statement CompiledStatement,
	partySolution []fr.Element,
) (*CompiledProof, error) {
	if prover == nil {
		return nil, fmt.Errorf("%w: nil validated prover", ErrInvalidCompiledMPIRole)
	}
	if err := validateCompiledMPIDynamicInput(
		prover.vk, statement, prover.role, partySolution,
	); err != nil {
		return nil, err
	}
	proof, _, err := compiledProveMPIWithTraceValidatedProfiled(
		prover.channel, prover.vk, statement, prover.role, partySolution, nil,
	)
	return proof, err
}

// ProveProfiled is Prove with diagnostic phase timing. It exists for the
// benchmark adapter; protocol callers should normally use Prove.
func (prover *CompiledMPIProver) ProveProfiled(
	statement CompiledStatement,
	partySolution []fr.Element,
) (*CompiledProof, CompiledMPIProfile, error) {
	var profile CompiledMPIProfile
	if prover == nil {
		return nil, profile, fmt.Errorf("%w: nil validated prover", ErrInvalidCompiledMPIRole)
	}
	if err := validateCompiledMPIDynamicInput(
		prover.vk, statement, prover.role, partySolution,
	); err != nil {
		return nil, profile, err
	}
	proof, _, err := compiledProveMPIWithTraceValidatedProfiled(
		prover.channel, prover.vk, statement, prover.role, partySolution, &profile,
	)
	return proof, profile, err
}

func compiledProveMPIWithTrace(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
) (*CompiledProof, compiledMPITrace, error) {
	var trace compiledMPITrace
	if err := validateCompiledMPIStaticRole(channel, vk, role); err != nil {
		return nil, trace, err
	}
	if err := validateCompiledMPIDynamicInput(vk, statement, role, partySolution); err != nil {
		return nil, trace, err
	}
	return compiledProveMPIWithTraceValidatedProfiled(channel, vk, statement, role, partySolution, nil)
}

func compiledProveMPIWithTraceValidatedProfiled(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
	profile *CompiledMPIProfile,
) (*CompiledProof, compiledMPITrace, error) {
	var trace compiledMPITrace
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
	phaseStarted := time.Now()
	w0Share := MPIPayload{}
	if role.Party != nil {
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
	transportStarted := startCompiledMPITransportProfile(profile)
	w0Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW0, w0Shape, w0Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW0, transportStarted)
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w0Public, err := channel.RootBroadcast(ProtocolMPIPhaseW0, w0BroadcastShape, w0RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW0, transportStarted)
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
	profile.recordTotal(ProtocolMPIPhaseW0, time.Since(phaseStarted))

	// W1: the local accumulator is fixed before lambda.
	phaseStarted = time.Now()
	w1Share := MPIPayload{}
	if role.Party != nil {
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w1Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW1, w1Shape, w1Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW1, transportStarted)
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w1Public, err := channel.RootBroadcast(ProtocolMPIPhaseW1, w1BroadcastShape, w1RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW1, transportStarted)
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
	profile.recordTotal(ProtocolMPIPhaseW1, time.Since(phaseStarted))

	// W2: three quotient chunks per party.
	phaseStarted = time.Now()
	w2Share := MPIPayload{}
	if role.Party != nil {
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w2Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseW2, w2Shape, w2Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW2, transportStarted)
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w2Public, err := channel.RootBroadcast(ProtocolMPIPhaseW2, w2BroadcastShape, w2RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW2, transportStarted)
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
	profile.recordTotal(ProtocolMPIPhaseW2, time.Since(phaseStarted))

	// W3: exact 21-field rows are gathered. The root scatters only each
	// party's t0/t1 coefficients, then broadcasts the complete public suffix.
	phaseStarted = time.Now()
	w3Share := MPIPayload{}
	if role.Party != nil {
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w3Gathered, err := channel.GatherWorkers(ProtocolMPIPhaseW3, w3GatherShape, w3Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW3, transportStarted)
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w3LocalPairPayload, err := channel.RootScatter(ProtocolMPIPhaseW3, w3ScatterShape, scatterPayloads)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW3, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("W3 scatter", err)
	}
	var localTreePair ProtocolW3ScatterRecord
	if role.Party != nil {
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
	transportStarted = startCompiledMPITransportProfile(profile)
	w3PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseW3, w3BroadcastShape, w3RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseW3, transportStarted)
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
	profile.recordTotal(ProtocolMPIPhaseW3, time.Since(phaseStarted))

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
	var sourceValues [LocalCompressedSourceCount]fr.Element
	if role.Party != nil {
		compression, compressionErr := CompressLocalTerminalSources(relation, alpha, mu)
		if compressionErr != nil {
			return nil, trace, compiledMPIError("local source compression", compressionErr)
		}
		for group := range sources {
			sources[group] = compression.Polynomials[group]
			sourceValues[group] = compression.Values[group]
		}
	}
	openingTranscript, err := NewHybridOpeningTranscript(openingInstance.TranscriptContext)
	if err != nil {
		return nil, trace, compiledMPIError("opening transcript", err)
	}

	// U0: weighted local source commitments in the semantic U direction. The
	// degree-aware hybrid deliberately performs no Taylor translation here.
	phaseStarted = time.Now()
	var openingPartyState compiledMPIOpeningPartyState
	u0Share := MPIPayload{}
	if role.Party != nil {
		var share HybridU0Message
		openingPartyState, share, err = compiledMPIOpeningU0Share(
			role.Party, sources, sourceValues, openingInstance, prepared,
		)
		if err != nil {
			return nil, trace, compiledMPIError("U0 local", err)
		}
		u0Share, err = PackHybridProtocolU0(share)
		if err != nil {
			return nil, trace, compiledMPIError("U0 payload", err)
		}
	}
	u0Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU0, protocolMPIOpAggregate, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u0Aggregate, err := channel.AggregateWorkers(ProtocolMPIPhaseU0, u0Shape, u0Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU0, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U0 aggregate", err)
	}
	u0RootPayload := MPIPayload{}
	if isRoot {
		publicProof.U0, err = UnpackHybridProtocolU0(u0Aggregate)
		if err == nil {
			u0RootPayload, err = PackHybridProtocolU0(publicProof.U0)
		}
		if err != nil {
			return nil, trace, compiledMPIError("U0 root assembly", err)
		}
	}
	u0BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU0, protocolMPIOpBroadcast, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u0PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU0, u0BroadcastShape, u0RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU0, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U0 broadcast", err)
	}
	publicProof.U0, err = UnpackHybridProtocolU0(u0PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U0 decode", err)
	}
	if err := openingTranscript.AppendU0(publicProof.U0); err != nil {
		return nil, trace, compiledMPIError("U0 transcript", err)
	}
	gammaOut, err := openingTranscript.DeriveGamma()
	if err != nil {
		return nil, trace, compiledMPIError("gamma", err)
	}
	alphaChallengeOut, err := openingTranscript.DeriveAlphaChallenge(openingInstance.SemanticQueryPoints)
	if err != nil {
		return nil, trace, compiledMPIError("alpha challenge", err)
	}
	gamma, alphaChallenge := gammaOut.Value, alphaChallengeOut.Value
	profile.recordTotal(ProtocolMPIPhaseU0, time.Since(phaseStarted))

	// U1: workers reveal g_j(alpha_ch) and commit to their honest degree-<M
	// functional Laurent witness. The coordinator reconstructs h_gamma.
	phaseStarted = time.Now()
	u1Share := MPIPayload{}
	if role.Party != nil {
		a, laurentCommitment, shareErr := compiledMPIOpeningU1Share(
			role.Party, &openingPartyState, localTreePair.T0, localTreePair.T1,
			prepared, gamma, alphaChallenge,
		)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U1 local", shareErr)
		}
		u1Share, err = PackHybridProtocolU1Gather(HybridProtocolU1GatherRecord{A: a, LaurentCommitment: laurentCommitment})
		if err != nil {
			return nil, trace, compiledMPIError("U1 gather payload", err)
		}
	}
	u1GatherShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU1, protocolMPIOpGather, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u1Gathered, err := channel.GatherWorkers(ProtocolMPIPhaseU1, u1GatherShape, u1Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU1, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U1 gather", err)
	}
	var uHatCoefficients []fr.Element
	u1RootPayload := MPIPayload{}
	if isRoot {
		uHatCoefficients = make([]fr.Element, m)
		var message HybridU1Message
		for slot := range u1Gathered {
			record, decodeErr := UnpackHybridProtocolU1Gather(u1Gathered[slot])
			if decodeErr != nil {
				return nil, trace, compiledMPIError("U1 gathered record", decodeErr)
			}
			gammaPower := fr.One()
			var weightedAtChallenge fr.Element
			for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
				message.PartialAtChallenge[claim].Add(&message.PartialAtChallenge[claim], &record.A[claim])
				var term fr.Element
				term.Mul(&gammaPower, &record.A[claim])
				weightedAtChallenge.Add(&weightedAtChallenge, &term)
				gammaPower.Mul(&gammaPower, &gamma)
			}
			if prepared.weights[slot].IsZero() {
				return nil, trace, compiledMPIError("U1 equality weight", ErrDLinkZGOpeningRejected)
			}
			uHatCoefficients[slot].Div(&weightedAtChallenge, &prepared.weights[slot])
			sourceCommitmentAdd(&message.LaurentCommitment, &record.LaurentCommitment)
		}
		message.FunctionalCommitment, err = role.Coordinator.SRS.CommitU(uHatCoefficients)
		if err != nil {
			return nil, trace, compiledMPIError("U1 functional commitment", err)
		}
		publicProof.U1 = message
		u1RootPayload, err = PackHybridProtocolU1Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U1 broadcast assembly", err)
		}
	}
	u1BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU1, protocolMPIOpBroadcast, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u1PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU1, u1BroadcastShape, u1RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU1, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U1 broadcast", err)
	}
	publicProof.U1, err = UnpackHybridProtocolU1Broadcast(u1PublicPayload)
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
	betaInverse := dlinkzgOpeningInverse(beta)
	profile.recordTotal(ProtocolMPIPhaseU1, time.Since(phaseStarted))

	// U2: workers send only six non-diagonal circuit values. The coordinator
	// derives all length-M functional/Laurent evaluations canonically.
	phaseStarted = time.Now()
	u2Share := MPIPayload{}
	if role.Party != nil {
		record, shareErr := compiledMPIOpeningU2Share(
			&openingPartyState, openingInstance.SemanticQueryPoints, beta,
		)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U2 local", shareErr)
		}
		u2Share, err = PackHybridProtocolU2Aggregate(record)
		if err != nil {
			return nil, trace, compiledMPIError("U2 aggregate payload", err)
		}
	}
	u2Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU2, protocolMPIOpAggregate, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u2AggregatePayload, err := channel.AggregateWorkers(ProtocolMPIPhaseU2, u2Shape, u2Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU2, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U2 aggregate", err)
	}
	u2RootPayload := MPIPayload{}
	if isRoot {
		record, decodeErr := UnpackHybridProtocolU2Aggregate(u2AggregatePayload)
		if decodeErr != nil {
			return nil, trace, compiledMPIError("U2 aggregate decode", decodeErr)
		}
		message := HybridU2Message{CircuitCrossValues: record.CircuitCrossValues}
		message.LaurentAtBeta[0] = cryptodlinkzg.Eval(uHatCoefficients, beta)
		message.LaurentAtBetaInverse[0] = cryptodlinkzg.Eval(uHatCoefficients, betaInverse)
		message.LaurentAtBeta[1] = cryptodlinkzg.Eval(outerSuffix.t0, beta)
		message.LaurentAtBetaInverse[1] = cryptodlinkzg.Eval(outerSuffix.t0, betaInverse)
		message.LaurentAtBeta[2] = cryptodlinkzg.Eval(outerSuffix.t1, beta)
		message.LaurentAtBetaInverse[2] = cryptodlinkzg.Eval(outerSuffix.t1, betaInverse)
		aWeights, bWeights := treeWeightPolynomials(prepared.tree, gamma, prepared.m)
		message.LaurentAtBeta[3] = compiledMPIHybridLaurentValue(
			uHatCoefficients, outerSuffix.t0, outerSuffix.t1,
			prepared.weights, aWeights, bWeights, beta,
		)
		message.LaurentAtBetaInverse[3] = compiledMPIHybridLaurentValue(
			uHatCoefficients, outerSuffix.t0, outerSuffix.t1,
			prepared.weights, aWeights, bWeights, betaInverse,
		)
		publicProof.U2 = message
		u2RootPayload, err = PackHybridProtocolU2Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U2 broadcast assembly", err)
		}
	}
	u2BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU2, protocolMPIOpBroadcast, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u2PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU2, u2BroadcastShape, u2RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU2, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U2 broadcast", err)
	}
	publicProof.U2, err = UnpackHybridProtocolU2Broadcast(u2PublicPayload)
	if err != nil {
		return nil, trace, compiledMPIError("U2 decode", err)
	}
	if err := openingTranscript.AppendU2(publicProof.U2); err != nil {
		return nil, trace, compiledMPIError("U2 transcript", err)
	}
	if err := verifyHybridOpeningScalarIdentity(
		openingInstance, publicProof.U1, publicProof.U2, prepared, gamma, beta,
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
	profile.recordTotal(ProtocolMPIPhaseU2, time.Since(phaseStarted))

	// U3: workers aggregate the semantic same-set quotient, the length-M
	// Laurent same-set quotient, and PiU. PiV is coordinator-only.
	phaseStarted = time.Now()
	u3Share := MPIPayload{}
	if role.Party != nil {
		shares, shareErr := compiledMPIOpeningU3Share(
			role.Party, &openingPartyState, openingInstance.SemanticQueryPoints,
			alphaChallenge, beta, kappa,
		)
		if shareErr != nil {
			return nil, trace, compiledMPIError("U3 local", shareErr)
		}
		u3Share, err = PackHybridProtocolU3Aggregate(HybridProtocolU3AggregateRecord{
			WCirc: shares[0], WLaur: shares[1], PiU: shares[2],
		})
		if err != nil {
			return nil, trace, compiledMPIError("U3 aggregate payload", err)
		}
	}
	u3Shape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU3, protocolMPIOpAggregate, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u3AggregatePayload, err := channel.AggregateWorkers(ProtocolMPIPhaseU3, u3Shape, u3Share)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU3, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U3 aggregate", err)
	}
	u3RootPayload := MPIPayload{}
	if isRoot {
		record, decodeErr := UnpackHybridProtocolU3Aggregate(u3AggregatePayload)
		if decodeErr != nil {
			return nil, trace, compiledMPIError("U3 aggregate decode", decodeErr)
		}
		message := HybridU3Message{WCirc: record.WCirc, WLaur: record.WLaur, PiU: record.PiU}
		qY, sourceValue := cryptodlinkzg.SyntheticDivision(uHatCoefficients, beta)
		if !sourceValue.Equal(&publicProof.U2.LaurentAtBeta[0]) {
			return nil, trace, compiledMPIError("U3 source-link value", ErrDLinkZGOpeningRejected)
		}
		message.PiV, err = role.Coordinator.SRS.CommitY(qY)
		if err != nil {
			return nil, trace, compiledMPIError("U3 PiV", err)
		}
		publicProof.U3 = message
		u3RootPayload, err = PackHybridProtocolU3Broadcast(message)
		if err != nil {
			return nil, trace, compiledMPIError("U3 broadcast assembly", err)
		}
	}
	u3BroadcastShape, _ := protocolMPIExpectedShape(ProtocolMPIPhaseU3, protocolMPIOpBroadcast, partitions)
	transportStarted = startCompiledMPITransportProfile(profile)
	u3PublicPayload, err := channel.RootBroadcast(ProtocolMPIPhaseU3, u3BroadcastShape, u3RootPayload)
	finishCompiledMPITransportProfile(profile, ProtocolMPIPhaseU3, transportStarted)
	if err != nil {
		return nil, trace, compiledMPIError("U3 broadcast", err)
	}
	publicProof.U3, err = UnpackHybridProtocolU3Broadcast(u3PublicPayload)
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
	profile.recordTotal(ProtocolMPIPhaseU3, time.Since(phaseStarted))
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
	if err := validateCompiledMPIStaticRole(channel, vk, role); err != nil {
		return err
	}
	return validateCompiledMPIDynamicInput(vk, statement, role, partySolution)
}

func validateCompiledMPIStaticRole(
	channel *ProtocolMPIChannel,
	vk CompiledVerifyingKey,
	role CompiledMPIRole,
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
	slot, ok := channel.PartySlot()
	if !ok || slot > uint64(^uint(0)>>1) {
		return fmt.Errorf("%w: invalid party slot", ErrInvalidCompiledMPIRole)
	}
	if role.Party == nil {
		return fmt.Errorf("%w: rank %d has no party role", ErrInvalidCompiledMPIRole, channel.Rank())
	}
	if channel.IsCoordinator() {
		if role.Coordinator == nil {
			return fmt.Errorf("%w: rank zero must own coordinator and party roles", ErrInvalidCompiledMPIRole)
		}
		coordinator := role.Coordinator
		if coordinator.Metadata != vk.Metadata || coordinator.SRS == nil ||
			coordinator.SRS.Validate() != nil || coordinator.SRS.Parties != vk.Metadata.PartitionCount ||
			coordinator.PublicInputPlacement == nil ||
			!compiledPlacementEqual(coordinator.PublicInputPlacement, vk.Verifier.PublicInputPlacement) ||
			!compiledAggregateFixedEqual(coordinator.FixedCommitments, vk.Verifier.FixedCommitments) {
			return fmt.Errorf("%w: coordinator role is not bound to the common VK", ErrInvalidCompiledMPIRole)
		}
	} else if role.Coordinator != nil {
		return fmt.Errorf("%w: non-root party rank owns coordinator role", ErrInvalidCompiledMPIRole)
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
	return nil
}

func validateCompiledMPIDynamicInput(
	vk CompiledVerifyingKey,
	statement CompiledStatement,
	role CompiledMPIRole,
	partySolution []fr.Element,
) error {
	if err := validateCompiledStatement(vk.Metadata, vk.Verifier.PublicInputPlacement, statement); err != nil {
		return err
	}
	if role.Party == nil {
		return fmt.Errorf("%w: rank has no party role", ErrInvalidCompiledMPIRole)
	}
	party := role.Party
	layout := party.Preprocessing.layout
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
	result.productCheck.Commitments[0], err = coordinator.SRS.CommitU(result.t0)
	if err != nil {
		return compiledMPIOuterSuffix{}, err
	}
	result.productCheck.Commitments[1], err = coordinator.SRS.CommitU(result.t1)
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

func compiledMPIPrepareOpening(instance DLinkZGOpeningInstance) (hybridOpeningPrepared, error) {
	prepared, err := prepareHybridOpeningInstance(instance)
	if err != nil {
		return prepared, err
	}
	prepared.weights = cryptodlinkzg.EqualityWeights(instance.PartitionPoint)
	return prepared, nil
}

func compiledMPIOpeningU0Share(
	party *CompiledPartySetup,
	sources [LocalCompressedSourceCount][]fr.Element,
	sourceValues [LocalCompressedSourceCount]fr.Element,
	_ DLinkZGOpeningInstance,
	prepared hybridOpeningPrepared,
) (compiledMPIOpeningPartyState, HybridU0Message, error) {
	var state compiledMPIOpeningPartyState
	var share HybridU0Message
	rank := party.Rank
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		if len(sources[claim]) > prepared.t {
			return state, share, fmt.Errorf("%w: source %d exceeds T", ErrCompiledMPIProtocol, claim)
		}
		state.sources[claim] = append([]fr.Element(nil), sources[claim]...)
		state.local.g[claim] = dlinkzgOpeningScalePolynomial(state.sources[claim], prepared.weights[rank])
		state.localCircuitClaims[claim].Mul(&prepared.weights[rank], &sourceValues[claim])
		commitment, err := party.RowSRS.CommitU(state.local.g[claim])
		if err != nil {
			return compiledMPIOpeningPartyState{}, HybridU0Message{}, err
		}
		share.PartialCommitments[claim] = commitment
	}
	return state, share, nil
}

func compiledMPIOpeningU1Share(
	party *CompiledPartySetup,
	state *compiledMPIOpeningPartyState,
	t0, t1 fr.Element,
	prepared hybridOpeningPrepared,
	gamma, alphaChallenge fr.Element,
) ([dlinkzgOpeningCircuitClaims]fr.Element, bn254.G1Affine, error) {
	var a [dlinkzgOpeningCircuitClaims]fr.Element
	aWeights, bWeights := treeWeightPolynomials(prepared.tree, gamma, prepared.m)
	state.local.pGamma = make([]fr.Element, prepared.t)
	gammaPower := fr.One()
	var weightedAtChallenge fr.Element
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		state.local.a[claim] = cryptodlinkzg.Eval(state.local.g[claim], alphaChallenge)
		a[claim] = state.local.a[claim]
		dlinkzgOpeningAddScaled(state.local.pGamma, state.sources[claim], gammaPower)
		var term fr.Element
		term.Mul(&gammaPower, &state.local.a[claim])
		weightedAtChallenge.Add(&weightedAtChallenge, &term)
		gammaPower.Mul(&gammaPower, &gamma)
	}
	if prepared.weights[party.Rank].IsZero() {
		return a, bn254.G1Affine{}, fmt.Errorf("%w: zero partition equality weight", ErrCompiledMPIProtocol)
	}
	state.local.uHat.Div(&weightedAtChallenge, &prepared.weights[party.Rank])
	state.local.h = dlinkzgOpeningMonomial(state.local.uHat, party.Rank)
	state.local.t0 = dlinkzgOpeningMonomial(t0, party.Rank)
	state.local.t1 = dlinkzgOpeningMonomial(t1, party.Rank)
	state.local.sFun = cryptodlinkzg.BuildLocalLaurentWithDomain(cryptodlinkzg.LocalLaurentInput{
		HXi: state.local.h, PsiR: prepared.weights, Nu: fr.One(),
		T0: state.local.t0, T1: state.local.t1, AXi: aWeights, BXi: bWeights,
	}, party.FastFixed.convolutionDomain)
	commitment, err := party.RowSRS.CommitU(state.local.sFun)
	if err != nil {
		return a, bn254.G1Affine{}, err
	}
	return a, commitment, nil
}

func compiledMPIOpeningU2Share(
	state *compiledMPIOpeningPartyState,
	semanticPoints [dlinkzgOpeningCircuitClaims]fr.Element,
	beta fr.Element,
) (HybridProtocolU2AggregateRecord, error) {
	var result HybridProtocolU2AggregateRecord
	betaInverse := dlinkzgOpeningInverse(beta)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		cross := 0
		for semantic := 0; semantic < dlinkzgOpeningCircuitClaims; semantic++ {
			value := cryptodlinkzg.Eval(state.local.g[claim], semanticPoints[semantic])
			state.local.semanticValues[claim][semantic] = value
			if semantic == claim {
				if !value.Equal(&state.localCircuitClaims[claim]) {
					return HybridProtocolU2AggregateRecord{}, fmt.Errorf("%w: local semantic claim %d", ErrCompiledMPIProtocol, claim)
				}
				continue
			}
			result.CircuitCrossValues[claim][cross] = value
			cross++
		}
	}
	localPolynomials := [4][]fr.Element{state.local.h, state.local.t0, state.local.t1, state.local.sFun}
	for polynomial := range localPolynomials {
		state.local.laurentAtBeta[polynomial] = cryptodlinkzg.Eval(localPolynomials[polynomial], beta)
		state.local.laurentAtBetaInverse[polynomial] = cryptodlinkzg.Eval(localPolynomials[polynomial], betaInverse)
	}
	return result, nil
}

func compiledMPIHybridLaurentValue(
	h, t0, t1, psiR, aWeights, bWeights []fr.Element,
	point fr.Element,
) fr.Element {
	var result fr.Element
	terms := [3][]fr.Element{h, t0, t1}
	weights := [3][]fr.Element{psiR, aWeights, bWeights}
	for term := range terms {
		spans := compiledMPIHybridOffDiagonalSpans(weights[term], point)
		limit := len(terms[term])
		if len(spans) < limit {
			limit = len(spans)
		}
		for index := 0; index < limit; index++ {
			var contribution fr.Element
			contribution.Mul(&terms[term][index], &spans[index])
			result.Add(&result, &contribution)
		}
	}
	return result
}

// compiledMPIHybridOffDiagonalSpans returns L_i(z)+R_i(z), where
// L_{i+1}=zL_i+v_i and R_{i-1}=v_i+zR_i. Thus one dot product evaluates
// OffDiag(c,V)(z) in O(M) without constructing its coefficient vector.
func compiledMPIHybridOffDiagonalSpans(weights []fr.Element, point fr.Element) []fr.Element {
	spans := make([]fr.Element, len(weights))
	var left fr.Element
	for index := 0; index < len(weights); index++ {
		spans[index].Set(&left)
		left.Mul(&left, &point)
		left.Add(&left, &weights[index])
	}
	var right fr.Element
	for index := len(weights) - 1; index >= 0; index-- {
		spans[index].Add(&spans[index], &right)
		right.Mul(&right, &point)
		right.Add(&right, &weights[index])
	}
	return spans
}

func compiledMPIOpeningU3Share(
	party *CompiledPartySetup,
	state *compiledMPIOpeningPartyState,
	semanticPoints [dlinkzgOpeningCircuitClaims]fr.Element,
	alphaChallenge, beta, kappa fr.Element,
) ([3]bn254.G1Affine, error) {
	var result [3]bn254.G1Affine
	betaInverse := dlinkzgOpeningInverse(beta)
	circuitPoints := hybridCircuitPoints(alphaChallenge, semanticPoints)
	circuitInputs := make([]cryptodlinkzg.SameSetInput, dlinkzgOpeningCircuitClaims)
	for claim := 0; claim < dlinkzgOpeningCircuitClaims; claim++ {
		values := make([]fr.Element, 0, len(circuitPoints))
		values = append(values, state.local.a[claim])
		values = append(values, state.local.semanticValues[claim][:]...)
		circuitInputs[claim] = cryptodlinkzg.SameSetInput{
			Polynomial:    state.local.g[claim],
			ClaimedValues: values,
		}
	}
	circuitBatch, err := cryptodlinkzg.BuildSameSetQuotient(circuitInputs, circuitPoints, kappa)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}
	result[0], err = party.RowSRS.CommitU(circuitBatch.Quotient)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}

	laurentPolynomials := [4][]fr.Element{state.local.h, state.local.t0, state.local.t1, state.local.sFun}
	laurentInputs := make([]cryptodlinkzg.SameSetInput, len(laurentPolynomials))
	for polynomial := range laurentPolynomials {
		laurentInputs[polynomial] = cryptodlinkzg.SameSetInput{
			Polynomial: laurentPolynomials[polynomial],
			ClaimedValues: []fr.Element{
				state.local.laurentAtBeta[polynomial],
				state.local.laurentAtBetaInverse[polynomial],
			},
		}
	}
	laurentBatch, err := cryptodlinkzg.BuildSameSetQuotient(
		laurentInputs, []fr.Element{beta, betaInverse}, kappa,
	)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}
	result[1], err = party.RowSRS.CommitU(laurentBatch.Quotient)
	if err != nil {
		return [3]bn254.G1Affine{}, err
	}

	qU, remainder := cryptodlinkzg.SyntheticDivision(state.local.pGamma, alphaChallenge)
	if !remainder.Equal(&state.local.uHat) {
		return [3]bn254.G1Affine{}, fmt.Errorf("%w: local source-link remainder", ErrCompiledMPIProtocol)
	}
	result[2], err = party.RowSRS.CommitSemantic(qU)
	return result, err
}

func compiledMPIError(stage string, err error) error {
	return fmt.Errorf("%w: %s: %w", ErrCompiledMPIProtocol, stage, err)
}
