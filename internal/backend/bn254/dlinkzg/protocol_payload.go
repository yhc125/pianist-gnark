package dlinkzg

// This file is the typed boundary between the distributed compiled protocol
// and ProtocolMPIChannel.  MPIPayload deliberately remains a small transport
// type; these helpers fix the semantic order of every W0--W3,U0--U3 record so
// protocol code never indexes an untyped field or point slice directly.

import (
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// ProtocolW3GatherRecord is one party's retained terminal row.
type ProtocolW3GatherRecord struct {
	Terminals LocalTerminalEvaluations
}

// ProtocolW3ScatterRecord is the rank-local ProductCheck coefficient pair
// returned by the coordinator after the retained terminal gather.
type ProtocolW3ScatterRecord struct {
	T0 fr.Element
	T1 fr.Element
}

// ProtocolW3BroadcastRecord is the complete public W3 suffix.  Its field
// order is all six SumCheck coefficients for rounds 0..log2(M)-1, followed by
// the 21 terminal and five ProductCheck endpoint evaluations.
type ProtocolW3BroadcastRecord struct {
	ProductCheck     OuterProductCheckCommitmentsMessage
	SumCheckRounds   []OuterSumCheckRoundMessage
	FinalEvaluations OuterFinalEvaluationsMessage
}

// ProtocolU1GatherRecord is one party's U1 record. D contains its three
// unweighted evaluations at z_ch; the coordinator applies the partition
// equality weight. LaurentCommitment is the party's additive commitment share.
type ProtocolU1GatherRecord struct {
	D                 [LocalCompressedSourceCount]fr.Element
	LaurentCommitment bn254.G1Affine
}

// ProtocolU2AggregateRecord is one party's seven additive U2 values. The
// transport order is (g_0(beta),g_0(beta^-1),...,g_2(beta^-1),S(beta)).
// S(beta^-1) is deliberately absent and is derived by the coordinator.
type ProtocolU2AggregateRecord struct {
	PartialAtBeta        [LocalCompressedSourceCount]fr.Element
	PartialAtBetaInverse [LocalCompressedSourceCount]fr.Element
	SAtBeta              fr.Element
}

// ProtocolU3AggregateRecord is one party's additive quotient share. PiY is
// coordinator-only and consequently occurs only in the U3 broadcast.
type ProtocolU3AggregateRecord struct {
	WN  bn254.G1Affine
	PiZ bn254.G1Affine
}

// PackProtocolW0 packs either a party W0 share or the aggregate W0
// broadcast. Both operations have the same semantic inventory.
func PackProtocolW0(message OuterW0Message) (MPIPayload, error) {
	return newTypedProtocolPayload("W0", nil, message.WitnessCommitments[:], MPIPayloadShape{G1: 3})
}

// UnpackProtocolW0 validates and decodes either W0 payload.
func UnpackProtocolW0(payload MPIPayload) (OuterW0Message, error) {
	validated, err := validateTypedProtocolPayload("W0", payload, MPIPayloadShape{G1: 3})
	if err != nil {
		return OuterW0Message{}, err
	}
	var message OuterW0Message
	copy(message.WitnessCommitments[:], validated.G1)
	return message, nil
}

// PackProtocolW1 packs either a party W1 share or the aggregate W1
// broadcast.
func PackProtocolW1(message OuterW1Message) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"W1", nil, []bn254.G1Affine{message.AccumulatorCommitment}, MPIPayloadShape{G1: 1},
	)
}

// UnpackProtocolW1 validates and decodes either W1 payload.
func UnpackProtocolW1(payload MPIPayload) (OuterW1Message, error) {
	validated, err := validateTypedProtocolPayload("W1", payload, MPIPayloadShape{G1: 1})
	if err != nil {
		return OuterW1Message{}, err
	}
	return OuterW1Message{AccumulatorCommitment: validated.G1[0]}, nil
}

// PackProtocolW2 packs either a party W2 share or the aggregate W2
// broadcast.
func PackProtocolW2(message OuterW2Message) (MPIPayload, error) {
	return newTypedProtocolPayload("W2", nil, message.QuotientCommitments[:], MPIPayloadShape{G1: 3})
}

// UnpackProtocolW2 validates and decodes either W2 payload.
func UnpackProtocolW2(payload MPIPayload) (OuterW2Message, error) {
	validated, err := validateTypedProtocolPayload("W2", payload, MPIPayloadShape{G1: 3})
	if err != nil {
		return OuterW2Message{}, err
	}
	var message OuterW2Message
	copy(message.QuotientCommitments[:], validated.G1)
	return message, nil
}

// PackProtocolW3Gather packs one party's canonical 13/1/7 terminal
// inventory in alpha, omega*alpha, x_star order.
func PackProtocolW3Gather(record ProtocolW3GatherRecord) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"W3 gather", record.Terminals.Flatten(), nil,
		MPIPayloadShape{Fields: OuterTerminalEvaluationCount},
	)
}

// UnpackProtocolW3Gather validates and decodes one retained W3 row.
func UnpackProtocolW3Gather(payload MPIPayload) (ProtocolW3GatherRecord, error) {
	validated, err := validateTypedProtocolPayload(
		"W3 gather", payload, MPIPayloadShape{Fields: OuterTerminalEvaluationCount},
	)
	if err != nil {
		return ProtocolW3GatherRecord{}, err
	}
	var terminals LocalTerminalEvaluations
	offset := 0
	offset += copy(terminals.Alpha[:], validated.Fields[offset:])
	offset += copy(terminals.OmegaAlpha[:], validated.Fields[offset:])
	copy(terminals.XStar[:], validated.Fields[offset:])
	return ProtocolW3GatherRecord{Terminals: terminals}, nil
}

// PackProtocolW3Scatter packs the local (t_0[i],t_1[i]) pair.
func PackProtocolW3Scatter(message ProtocolW3ScatterRecord) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"W3 scatter", []fr.Element{message.T0, message.T1}, nil, MPIPayloadShape{Fields: 2},
	)
}

// UnpackProtocolW3Scatter validates and decodes a local coefficient
// pair.
func UnpackProtocolW3Scatter(payload MPIPayload) (ProtocolW3ScatterRecord, error) {
	validated, err := validateTypedProtocolPayload("W3 scatter", payload, MPIPayloadShape{Fields: 2})
	if err != nil {
		return ProtocolW3ScatterRecord{}, err
	}
	return ProtocolW3ScatterRecord{T0: validated.Fields[0], T1: validated.Fields[1]}, nil
}

// PackProtocolW3Broadcast packs the public ProductCheck commitments,
// SumCheck rounds, and final evaluations for exactly M partitions.
func PackProtocolW3Broadcast(message ProtocolW3BroadcastRecord, partitions uint64) (MPIPayload, error) {
	rounds, err := protocolPayloadRoundCount(partitions)
	if err != nil {
		return MPIPayload{}, err
	}
	if len(message.SumCheckRounds) != rounds {
		return MPIPayload{}, fmt.Errorf(
			"%w: W3 broadcast has %d SumCheck rounds, want %d for M=%d",
			ErrMPIPayload, len(message.SumCheckRounds), rounds, partitions,
		)
	}
	fields := make([]fr.Element, 0,
		SumCheckRoundCoefficients*rounds+OuterTerminalEvaluationCount+OuterProductCheckClaimCount,
	)
	for round := range message.SumCheckRounds {
		fields = append(fields, message.SumCheckRounds[round].Coefficients[:]...)
	}
	fields = append(fields, message.FinalEvaluations.Terminal[:]...)
	fields = append(fields, message.FinalEvaluations.ProductCheck[:]...)
	shape, err := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpBroadcast, partitions)
	if err != nil {
		return MPIPayload{}, err
	}
	return newTypedProtocolPayload("W3 broadcast", fields, message.ProductCheck.Commitments[:], shape)
}

// UnpackProtocolW3Broadcast validates and decodes the public W3 suffix
// for exactly M partitions. SumCheckRounds is newly allocated.
func UnpackProtocolW3Broadcast(payload MPIPayload, partitions uint64) (ProtocolW3BroadcastRecord, error) {
	rounds, err := protocolPayloadRoundCount(partitions)
	if err != nil {
		return ProtocolW3BroadcastRecord{}, err
	}
	shape, err := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpBroadcast, partitions)
	if err != nil {
		return ProtocolW3BroadcastRecord{}, err
	}
	validated, err := validateTypedProtocolPayload("W3 broadcast", payload, shape)
	if err != nil {
		return ProtocolW3BroadcastRecord{}, err
	}
	message := ProtocolW3BroadcastRecord{SumCheckRounds: make([]OuterSumCheckRoundMessage, rounds)}
	offset := 0
	for round := range message.SumCheckRounds {
		offset += copy(message.SumCheckRounds[round].Coefficients[:], validated.Fields[offset:])
	}
	offset += copy(message.FinalEvaluations.Terminal[:], validated.Fields[offset:])
	copy(message.FinalEvaluations.ProductCheck[:], validated.Fields[offset:])
	copy(message.ProductCheck.Commitments[:], validated.G1)
	return message, nil
}

// PackProtocolU0 packs either a party U0 share or the aggregate U0
// broadcast.
func PackProtocolU0(message U0Message) (MPIPayload, error) {
	return newTypedProtocolPayload("U0", nil, message.PartialCommitments[:], MPIPayloadShape{G1: 3})
}

// UnpackProtocolU0 validates and decodes either U0 payload.
func UnpackProtocolU0(payload MPIPayload) (U0Message, error) {
	validated, err := validateTypedProtocolPayload("U0", payload, MPIPayloadShape{G1: 3})
	if err != nil {
		return U0Message{}, err
	}
	var message U0Message
	copy(message.PartialCommitments[:], validated.G1)
	return message, nil
}

// PackProtocolU1Gather packs one party's local d values and Laurent
// commitment share.
func PackProtocolU1Gather(message ProtocolU1GatherRecord) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"U1 gather", message.D[:], []bn254.G1Affine{message.LaurentCommitment},
		MPIPayloadShape{Fields: 3, G1: 1},
	)
}

// UnpackProtocolU1Gather validates and decodes one local U1 record.
func UnpackProtocolU1Gather(payload MPIPayload) (ProtocolU1GatherRecord, error) {
	validated, err := validateTypedProtocolPayload("U1 gather", payload, MPIPayloadShape{Fields: 3, G1: 1})
	if err != nil {
		return ProtocolU1GatherRecord{}, err
	}
	var message ProtocolU1GatherRecord
	copy(message.D[:], validated.Fields)
	message.LaurentCommitment = validated.G1[0]
	return message, nil
}

// PackProtocolU1Broadcast packs the public U1 transcript message.
func PackProtocolU1Broadcast(message U1Message) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"U1 broadcast", message.LinkEvaluations[:],
		[]bn254.G1Affine{message.LinkCommitment, message.LaurentCommitment},
		MPIPayloadShape{Fields: 3, G1: 2},
	)
}

// UnpackProtocolU1Broadcast validates and decodes public U1.
func UnpackProtocolU1Broadcast(payload MPIPayload) (U1Message, error) {
	validated, err := validateTypedProtocolPayload("U1 broadcast", payload, MPIPayloadShape{Fields: 3, G1: 2})
	if err != nil {
		return U1Message{}, err
	}
	var message U1Message
	copy(message.LinkEvaluations[:], validated.Fields)
	message.LinkCommitment = validated.G1[0]
	message.LaurentCommitment = validated.G1[1]
	return message, nil
}

// PackProtocolU2Aggregate packs one party's seven additive values in
// the canonical interleaved order.
func PackProtocolU2Aggregate(message ProtocolU2AggregateRecord) (MPIPayload, error) {
	fields := make([]fr.Element, 0, 7)
	for polynomial := range message.PartialAtBeta {
		fields = append(fields, message.PartialAtBeta[polynomial], message.PartialAtBetaInverse[polynomial])
	}
	fields = append(fields, message.SAtBeta)
	return newTypedProtocolPayload("U2 aggregate", fields, nil, MPIPayloadShape{Fields: 7})
}

// UnpackProtocolU2Aggregate validates and decodes one local U2 share.
func UnpackProtocolU2Aggregate(payload MPIPayload) (ProtocolU2AggregateRecord, error) {
	validated, err := validateTypedProtocolPayload("U2 aggregate", payload, MPIPayloadShape{Fields: 7})
	if err != nil {
		return ProtocolU2AggregateRecord{}, err
	}
	var message ProtocolU2AggregateRecord
	offset := 0
	for polynomial := range message.PartialAtBeta {
		message.PartialAtBeta[polynomial] = validated.Fields[offset]
		message.PartialAtBetaInverse[polynomial] = validated.Fields[offset+1]
		offset += 2
	}
	message.SAtBeta = validated.Fields[offset]
	return message, nil
}

// PackProtocolU2Broadcast packs U2 in the same interleaved order used
// by CompiledProof.MarshalBinary: each beta value immediately precedes its
// beta-inverse value, first for g_0..g_2 and then h_xi,t_0,t_1,S^lin.
func PackProtocolU2Broadcast(message U2Message) (MPIPayload, error) {
	fields := make([]fr.Element, 0, 14)
	for polynomial := range message.PartialAtBeta {
		fields = append(fields, message.PartialAtBeta[polynomial], message.PartialAtBetaInverse[polynomial])
	}
	for polynomial := range message.BatchAtBeta {
		fields = append(fields, message.BatchAtBeta[polynomial], message.BatchAtBetaInverse[polynomial])
	}
	return newTypedProtocolPayload("U2 broadcast", fields, nil, MPIPayloadShape{Fields: 14})
}

// UnpackProtocolU2Broadcast validates and decodes public U2.
func UnpackProtocolU2Broadcast(payload MPIPayload) (U2Message, error) {
	validated, err := validateTypedProtocolPayload("U2 broadcast", payload, MPIPayloadShape{Fields: 14})
	if err != nil {
		return U2Message{}, err
	}
	var message U2Message
	offset := 0
	for polynomial := range message.PartialAtBeta {
		message.PartialAtBeta[polynomial] = validated.Fields[offset]
		message.PartialAtBetaInverse[polynomial] = validated.Fields[offset+1]
		offset += 2
	}
	for polynomial := range message.BatchAtBeta {
		message.BatchAtBeta[polynomial] = validated.Fields[offset]
		message.BatchAtBetaInverse[polynomial] = validated.Fields[offset+1]
		offset += 2
	}
	return message, nil
}

// PackProtocolU3Aggregate packs one party's nested quotient and source-link
// quotient in WN,PiZ order.
func PackProtocolU3Aggregate(message ProtocolU3AggregateRecord) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"U3 aggregate", nil, []bn254.G1Affine{message.WN, message.PiZ},
		MPIPayloadShape{G1: 2},
	)
}

// UnpackProtocolU3Aggregate validates and decodes one local U3 share.
func UnpackProtocolU3Aggregate(payload MPIPayload) (ProtocolU3AggregateRecord, error) {
	validated, err := validateTypedProtocolPayload("U3 aggregate", payload, MPIPayloadShape{G1: 2})
	if err != nil {
		return ProtocolU3AggregateRecord{}, err
	}
	return ProtocolU3AggregateRecord{WN: validated.G1[0], PiZ: validated.G1[1]}, nil
}

// PackProtocolU3Broadcast packs public U3 in WN,PiZ,PiY order.
func PackProtocolU3Broadcast(message U3Message) (MPIPayload, error) {
	return newTypedProtocolPayload(
		"U3 broadcast", nil,
		[]bn254.G1Affine{message.WN, message.PiZ, message.PiY},
		MPIPayloadShape{G1: 3},
	)
}

// UnpackProtocolU3Broadcast validates and decodes public U3.
func UnpackProtocolU3Broadcast(payload MPIPayload) (U3Message, error) {
	validated, err := validateTypedProtocolPayload("U3 broadcast", payload, MPIPayloadShape{G1: 3})
	if err != nil {
		return U3Message{}, err
	}
	return U3Message{WN: validated.G1[0], PiZ: validated.G1[1], PiY: validated.G1[2]}, nil
}

func protocolPayloadRoundCount(partitions uint64) (int, error) {
	if partitions < 2 || partitions&(partitions-1) != 0 {
		return 0, fmt.Errorf(
			"%w: partition count %d is not a power of two at least two", ErrMPIPayload, partitions,
		)
	}
	return bits.Len64(partitions) - 1, nil
}

func newTypedProtocolPayload(
	label string,
	fields []fr.Element,
	points []bn254.G1Affine,
	want MPIPayloadShape,
) (MPIPayload, error) {
	return validateTypedProtocolPayload(label, MPIPayload{Fields: fields, G1: points}, want)
}

// validateTypedProtocolPayload always returns fresh backing slices. Besides
// enforcing the semantic shape, it catches malformed in-memory field values
// and curve points before they reach either arithmetic aggregation or the wire
// encoder.
func validateTypedProtocolPayload(label string, payload MPIPayload, want MPIPayloadShape) (MPIPayload, error) {
	if payload.Shape() != want {
		return MPIPayload{}, protocolPayloadShapeError(label, payload.Shape(), want)
	}
	result := MPIPayload{
		Fields: append([]fr.Element(nil), payload.Fields...),
		G1:     append([]bn254.G1Affine(nil), payload.G1...),
	}
	for index := range result.Fields {
		if err := validateTranscriptField(&result.Fields[index]); err != nil {
			return MPIPayload{}, fmt.Errorf("%w: %s field %d: %v", ErrMPIPayload, label, index, err)
		}
	}
	for index := range result.G1 {
		if err := validateTranscriptG1(&result.G1[index]); err != nil {
			return MPIPayload{}, fmt.Errorf("%w: %s G1 element %d: %v", ErrMPIPayload, label, index, err)
		}
	}
	return result, nil
}
