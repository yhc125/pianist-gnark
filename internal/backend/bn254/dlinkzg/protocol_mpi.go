package dlinkzg

// This file contains the paper-shaped MPI transport. It is deliberately
// separate from mpi.go: MPIChannel remains the compatibility transport used by
// the older U0--U3 smoke test, while ProtocolMPIChannel fixes the exact
// coordinator-plus-parties execution model used by the DLinKZG protocol.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"sync"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

const protocolMPIHeaderSize = mpiHeaderSize

var (
	// ErrProtocolMPIConfiguration reports a world that is not one coordinator
	// followed by a power-of-two number (at least two) of proving parties.
	ErrProtocolMPIConfiguration = errors.New("dlinkzg: invalid protocol MPI configuration")
	// ErrProtocolMPISchedule reports a phase or operation that is not the next
	// operation in the fixed W0--W3,U0--U3 transport schedule.
	ErrProtocolMPISchedule = errors.New("dlinkzg: invalid protocol MPI schedule")
	// ErrProtocolMPIRecord reports a malformed or unexpected protocol header.
	ErrProtocolMPIRecord = errors.New("dlinkzg: invalid protocol MPI record")
)

// ProtocolMPIPhase names the eight worker/coordinator communication phases.
// Rank zero is always the coordinator. MPI rank r in [1,M] is party slot r-1.
type ProtocolMPIPhase uint8

const (
	ProtocolMPIPhaseW0 ProtocolMPIPhase = iota
	ProtocolMPIPhaseW1
	ProtocolMPIPhaseW2
	ProtocolMPIPhaseW3
	ProtocolMPIPhaseU0
	ProtocolMPIPhaseU1
	ProtocolMPIPhaseU2
	ProtocolMPIPhaseU3
)

func (phase ProtocolMPIPhase) String() string {
	switch phase {
	case ProtocolMPIPhaseW0:
		return "DLinKZG/W0"
	case ProtocolMPIPhaseW1:
		return "DLinKZG/W1"
	case ProtocolMPIPhaseW2:
		return "DLinKZG/W2"
	case ProtocolMPIPhaseW3:
		return "DLinKZG/W3"
	case ProtocolMPIPhaseU0:
		return "DLinKZG/U0"
	case ProtocolMPIPhaseU1:
		return "DLinKZG/U1"
	case ProtocolMPIPhaseU2:
		return "DLinKZG/U2"
	case ProtocolMPIPhaseU3:
		return "DLinKZG/U3"
	default:
		return fmt.Sprintf("DLinKZG/invalid-protocol-phase-%d", uint8(phase))
	}
}

func (phase ProtocolMPIPhase) valid() bool {
	return phase >= ProtocolMPIPhaseW0 && phase <= ProtocolMPIPhaseU3
}

type protocolMPIOperation byte

const (
	protocolMPIOpAggregate protocolMPIOperation = iota + 1
	protocolMPIOpGather
	protocolMPIOpScatter
	protocolMPIOpBroadcast
)

func (operation protocolMPIOperation) String() string {
	switch operation {
	case protocolMPIOpAggregate:
		return "aggregate-workers"
	case protocolMPIOpGather:
		return "gather-workers"
	case protocolMPIOpScatter:
		return "root-scatter"
	case protocolMPIOpBroadcast:
		return "root-broadcast"
	default:
		return fmt.Sprintf("invalid-operation-%d", byte(operation))
	}
}

func (operation protocolMPIOperation) valid() bool {
	return operation >= protocolMPIOpAggregate && operation <= protocolMPIOpBroadcast
}

var protocolMPISchedule = [8][]protocolMPIOperation{
	ProtocolMPIPhaseW0: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
	ProtocolMPIPhaseW1: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
	ProtocolMPIPhaseW2: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
	ProtocolMPIPhaseW3: {protocolMPIOpGather, protocolMPIOpScatter, protocolMPIOpBroadcast},
	ProtocolMPIPhaseU0: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
	ProtocolMPIPhaseU1: {protocolMPIOpGather, protocolMPIOpBroadcast},
	ProtocolMPIPhaseU2: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
	ProtocolMPIPhaseU3: {protocolMPIOpAggregate, protocolMPIOpBroadcast},
}

// ProtocolMPIPhaseAccounting separates payload bytes from the canonical
// protocol framing. Counts are successful local calls, not cluster-wide sums.
type ProtocolMPIPhaseAccounting struct {
	Aggregations     uint64 `json:"aggregations"`
	Gathers          uint64 `json:"gathers"`
	Scatters         uint64 `json:"scatters"`
	Broadcasts       uint64 `json:"broadcasts"`
	PayloadBytesSent uint64 `json:"payload_bytes_sent"`
	PayloadBytesRecv uint64 `json:"payload_bytes_received"`
	WireBytesSent    uint64 `json:"wire_bytes_sent"`
	WireBytesRecv    uint64 `json:"wire_bytes_received"`
}

// ProtocolMPIAccounting is one process's immutable communication snapshot.
// WorldSize is M+1 and PartitionCount is M.
type ProtocolMPIAccounting struct {
	Rank           uint64                        `json:"rank"`
	WorldSize      uint64                        `json:"mpi_world_size"`
	PartitionCount uint64                        `json:"partitions"`
	Operations     uint64                        `json:"operations"`
	Total          ProtocolMPIPhaseAccounting    `json:"total"`
	Phases         [8]ProtocolMPIPhaseAccounting `json:"phases"`
}

type protocolMPIRecordHeader struct {
	Operation protocolMPIOperation
	Phase     ProtocolMPIPhase
	Sequence  uint32
	Shape     MPIPayloadShape
}

// ProtocolMPIChannel is a serial, deterministic transport for the exact
// M-party protocol. It is not safe for concurrent calls on the same channel.
// Different MPI ranks must execute the same fixed operation schedule.
type ProtocolMPIChannel struct {
	transport  mpiByteTransport
	rank       uint64
	worldSize  uint64
	partitions uint64
	sequence   uint32
	phase      ProtocolMPIPhase
	phaseStep  int
	complete   bool

	accountingMu sync.Mutex
	accounting   ProtocolMPIAccounting
}

// NewProtocolMPIChannel binds the paper-shaped channel to the initialized
// simpleMPI world.
func NewProtocolMPIChannel() (*ProtocolMPIChannel, error) {
	return newProtocolMPIChannel(simpleMPITransport{rank: mpi.SelfRank, size: mpi.WorldSize})
}

func newProtocolMPIChannel(transport mpiByteTransport) (*ProtocolMPIChannel, error) {
	if transport == nil {
		return nil, fmt.Errorf("%w: nil transport", ErrProtocolMPIConfiguration)
	}
	rank, worldSize := transport.Rank(), transport.Size()
	if worldSize < 3 || rank >= worldSize {
		return nil, fmt.Errorf(
			"%w: rank %d in world size %d",
			ErrProtocolMPIConfiguration, rank, worldSize,
		)
	}
	partitions := worldSize - 1
	if partitions < 2 || partitions&(partitions-1) != 0 {
		return nil, fmt.Errorf(
			"%w: world size %d gives non-power-of-two partition count %d",
			ErrProtocolMPIConfiguration, worldSize, partitions,
		)
	}
	return &ProtocolMPIChannel{
		transport:  transport,
		rank:       rank,
		worldSize:  worldSize,
		partitions: partitions,
		phase:      ProtocolMPIPhaseW0,
		accounting: ProtocolMPIAccounting{
			Rank:           rank,
			WorldSize:      worldSize,
			PartitionCount: partitions,
		},
	}, nil
}

// Rank returns the MPI rank. Rank zero is the coordinator.
func (channel *ProtocolMPIChannel) Rank() uint64 { return channel.rank }

// WorldSize returns M+1: the coordinator plus M parties.
func (channel *ProtocolMPIChannel) WorldSize() uint64 { return channel.worldSize }

// IsCoordinator reports whether this process is coordinator rank zero.
func (channel *ProtocolMPIChannel) IsCoordinator() bool { return channel.rank == 0 }

// PartitionCount returns M, excluding coordinator rank zero.
func (channel *ProtocolMPIChannel) PartitionCount() uint64 { return channel.partitions }

// PartySlot maps MPI rank r in [1,M] to canonical party slot r-1.
func (channel *ProtocolMPIChannel) PartySlot() (uint64, bool) {
	if channel.IsCoordinator() {
		return 0, false
	}
	return channel.rank - 1, true
}

// AggregateWorkers adds one fixed-shape share from every party. The
// coordinator supplies an empty workerShare and contributes no additive share.
// Only the coordinator receives the sum; parties receive an empty payload.
func (channel *ProtocolMPIChannel) AggregateWorkers(
	phase ProtocolMPIPhase,
	shape MPIPayloadShape,
	workerShare MPIPayload,
) (MPIPayload, error) {
	if err := channel.prepareOperation(phase, protocolMPIOpAggregate, shape); err != nil {
		return MPIPayload{}, err
	}
	header := channel.operationHeader(phase, protocolMPIOpAggregate, shape)

	if !channel.IsCoordinator() {
		if workerShare.Shape() != shape {
			return MPIPayload{}, protocolPayloadShapeError("worker aggregate share", workerShare.Shape(), shape)
		}
		encoded, err := encodeMPIPayload(workerShare)
		if err != nil {
			return MPIPayload{}, err
		}
		if err := channel.sendHeader(0, header); err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: protocol aggregate header send: %w", err)
		}
		if err := channel.sendPayload(phase, 0, encoded); err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: protocol aggregate payload send: %w", err)
		}
		channel.finishOperation(phase, protocolMPIOpAggregate)
		return MPIPayload{}, nil
	}

	if !emptyMPIPayload(workerShare) {
		return MPIPayload{}, fmt.Errorf("%w: coordinator supplied an aggregate share", ErrMPIPayload)
	}
	result := MPIPayload{
		Fields: make([]fr.Element, shape.Fields),
		G1:     make([]bn254.G1Affine, shape.G1),
	}
	for rank := uint64(1); rank < channel.worldSize; rank++ {
		if err := channel.receiveExpectedHeader(rank, header); err != nil {
			return MPIPayload{}, fmt.Errorf("%w: aggregate header from rank %d: %v", ErrProtocolMPIRecord, rank, err)
		}
		share, err := channel.receivePayload(phase, rank, shape)
		if err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: protocol aggregate payload from rank %d: %w", rank, err)
		}
		addMPIPayload(&result, share)
	}
	channel.finishOperation(phase, protocolMPIOpAggregate)
	return result, nil
}

// GatherWorkers receives one record from every party without combining it.
// On the coordinator, result[slot] is the record sent by MPI rank slot+1,
// independent of arrival order. Parties return nil.
func (channel *ProtocolMPIChannel) GatherWorkers(
	phase ProtocolMPIPhase,
	shape MPIPayloadShape,
	workerPayload MPIPayload,
) ([]MPIPayload, error) {
	if err := channel.prepareOperation(phase, protocolMPIOpGather, shape); err != nil {
		return nil, err
	}
	header := channel.operationHeader(phase, protocolMPIOpGather, shape)

	if !channel.IsCoordinator() {
		if workerPayload.Shape() != shape {
			return nil, protocolPayloadShapeError("worker gather record", workerPayload.Shape(), shape)
		}
		encoded, err := encodeMPIPayload(workerPayload)
		if err != nil {
			return nil, err
		}
		if err := channel.sendHeader(0, header); err != nil {
			return nil, fmt.Errorf("dlinkzg: protocol gather header send: %w", err)
		}
		if err := channel.sendPayload(phase, 0, encoded); err != nil {
			return nil, fmt.Errorf("dlinkzg: protocol gather payload send: %w", err)
		}
		channel.finishOperation(phase, protocolMPIOpGather)
		return nil, nil
	}

	if !emptyMPIPayload(workerPayload) {
		return nil, fmt.Errorf("%w: coordinator supplied a gather record", ErrMPIPayload)
	}
	result := make([]MPIPayload, channel.partitions)
	for rank := uint64(1); rank < channel.worldSize; rank++ {
		if err := channel.receiveExpectedHeader(rank, header); err != nil {
			return nil, fmt.Errorf("%w: gather header from rank %d: %v", ErrProtocolMPIRecord, rank, err)
		}
		record, err := channel.receivePayload(phase, rank, shape)
		if err != nil {
			return nil, fmt.Errorf("dlinkzg: protocol gather payload from rank %d: %w", rank, err)
		}
		result[rank-1] = record
	}
	channel.finishOperation(phase, protocolMPIOpGather)
	return result, nil
}

// RootScatter sends exactly one fixed-shape payload to each party. The
// coordinator supplies M payloads ordered by party slot. Party rank r receives
// only rootPayloads[r-1]. The coordinator receives an empty payload.
func (channel *ProtocolMPIChannel) RootScatter(
	phase ProtocolMPIPhase,
	shape MPIPayloadShape,
	rootPayloads []MPIPayload,
) (MPIPayload, error) {
	if err := channel.prepareOperation(phase, protocolMPIOpScatter, shape); err != nil {
		return MPIPayload{}, err
	}
	header := channel.operationHeader(phase, protocolMPIOpScatter, shape)

	if channel.IsCoordinator() {
		if uint64(len(rootPayloads)) != channel.partitions {
			return MPIPayload{}, fmt.Errorf(
				"%w: scatter has %d payloads, want %d",
				ErrMPIPayload, len(rootPayloads), channel.partitions,
			)
		}
		encoded := make([][]byte, len(rootPayloads))
		for slot := range rootPayloads {
			if rootPayloads[slot].Shape() != shape {
				return MPIPayload{}, protocolPayloadShapeError(
					fmt.Sprintf("scatter slot %d", slot), rootPayloads[slot].Shape(), shape,
				)
			}
			var err error
			encoded[slot], err = encodeMPIPayload(rootPayloads[slot])
			if err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: protocol scatter slot %d: %w", slot, err)
			}
		}
		for rank := uint64(1); rank < channel.worldSize; rank++ {
			if err := channel.sendHeader(rank, header); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: protocol scatter header to rank %d: %w", rank, err)
			}
			if err := channel.sendPayload(phase, rank, encoded[rank-1]); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: protocol scatter payload to rank %d: %w", rank, err)
			}
		}
		channel.finishOperation(phase, protocolMPIOpScatter)
		return MPIPayload{}, nil
	}

	if len(rootPayloads) != 0 {
		return MPIPayload{}, fmt.Errorf("%w: party supplied root scatter payloads", ErrMPIPayload)
	}
	if err := channel.receiveExpectedHeader(0, header); err != nil {
		return MPIPayload{}, fmt.Errorf("%w: scatter header: %v", ErrProtocolMPIRecord, err)
	}
	result, err := channel.receivePayload(phase, 0, shape)
	if err != nil {
		return MPIPayload{}, fmt.Errorf("dlinkzg: protocol scatter payload: %w", err)
	}
	channel.finishOperation(phase, protocolMPIOpScatter)
	return result, nil
}

// RootBroadcast sends one public fixed-shape payload from the coordinator to
// all parties and returns that payload on every rank.
func (channel *ProtocolMPIChannel) RootBroadcast(
	phase ProtocolMPIPhase,
	shape MPIPayloadShape,
	rootPayload MPIPayload,
) (MPIPayload, error) {
	if err := channel.prepareOperation(phase, protocolMPIOpBroadcast, shape); err != nil {
		return MPIPayload{}, err
	}
	header := channel.operationHeader(phase, protocolMPIOpBroadcast, shape)

	if channel.IsCoordinator() {
		if rootPayload.Shape() != shape {
			return MPIPayload{}, protocolPayloadShapeError("root broadcast payload", rootPayload.Shape(), shape)
		}
		encoded, err := encodeMPIPayload(rootPayload)
		if err != nil {
			return MPIPayload{}, err
		}
		for rank := uint64(1); rank < channel.worldSize; rank++ {
			if err := channel.sendHeader(rank, header); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: protocol broadcast header to rank %d: %w", rank, err)
			}
			if err := channel.sendPayload(phase, rank, encoded); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: protocol broadcast payload to rank %d: %w", rank, err)
			}
		}
		result := rootPayload.clone()
		channel.finishOperation(phase, protocolMPIOpBroadcast)
		return result, nil
	}

	if !emptyMPIPayload(rootPayload) {
		return MPIPayload{}, fmt.Errorf("%w: party supplied a root broadcast payload", ErrMPIPayload)
	}
	if err := channel.receiveExpectedHeader(0, header); err != nil {
		return MPIPayload{}, fmt.Errorf("%w: broadcast header: %v", ErrProtocolMPIRecord, err)
	}
	result, err := channel.receivePayload(phase, 0, shape)
	if err != nil {
		return MPIPayload{}, fmt.Errorf("dlinkzg: protocol broadcast payload: %w", err)
	}
	channel.finishOperation(phase, protocolMPIOpBroadcast)
	return result, nil
}

// Accounting returns an immutable copy of this rank's communication ledger.
func (channel *ProtocolMPIChannel) Accounting() ProtocolMPIAccounting {
	channel.accountingMu.Lock()
	defer channel.accountingMu.Unlock()
	return channel.accounting
}

func (channel *ProtocolMPIChannel) prepareOperation(
	phase ProtocolMPIPhase,
	operation protocolMPIOperation,
	shape MPIPayloadShape,
) error {
	if channel.complete {
		return fmt.Errorf("%w: protocol is complete", ErrProtocolMPISchedule)
	}
	if !phase.valid() {
		return fmt.Errorf("%w: %s", ErrProtocolMPISchedule, phase)
	}
	if phase != channel.phase {
		return fmt.Errorf(
			"%w: got phase %s, want %s",
			ErrProtocolMPISchedule, phase, channel.phase,
		)
	}
	schedule := protocolMPISchedule[int(channel.phase)]
	if channel.phaseStep < 0 || channel.phaseStep >= len(schedule) {
		return fmt.Errorf("%w: invalid internal phase step", ErrProtocolMPISchedule)
	}
	expectedOperation := schedule[channel.phaseStep]
	if operation != expectedOperation {
		return fmt.Errorf(
			"%w: phase %s got %s, want %s",
			ErrProtocolMPISchedule, phase, operation, expectedOperation,
		)
	}
	expectedShape, err := protocolMPIExpectedShape(phase, operation, channel.partitions)
	if err != nil {
		return err
	}
	if shape != expectedShape {
		return protocolPayloadShapeError(
			fmt.Sprintf("%s %s", phase, operation), shape, expectedShape,
		)
	}
	if _, err := mpiPayloadByteSize(shape); err != nil {
		return err
	}
	if channel.sequence == math.MaxUint32 {
		return fmt.Errorf("%w: sequence exhausted", ErrProtocolMPISchedule)
	}
	return nil
}

func (channel *ProtocolMPIChannel) operationHeader(
	phase ProtocolMPIPhase,
	operation protocolMPIOperation,
	shape MPIPayloadShape,
) protocolMPIRecordHeader {
	return protocolMPIRecordHeader{
		Operation: operation,
		Phase:     phase,
		Sequence:  channel.sequence,
		Shape:     shape,
	}
}

func (channel *ProtocolMPIChannel) finishOperation(
	phase ProtocolMPIPhase,
	operation protocolMPIOperation,
) {
	channel.recordCall(phase, func(accounting *ProtocolMPIPhaseAccounting) {
		switch operation {
		case protocolMPIOpAggregate:
			accounting.Aggregations++
		case protocolMPIOpGather:
			accounting.Gathers++
		case protocolMPIOpScatter:
			accounting.Scatters++
		case protocolMPIOpBroadcast:
			accounting.Broadcasts++
		}
	})
	channel.sequence++
	channel.accountingMu.Lock()
	channel.accounting.Operations++
	channel.accountingMu.Unlock()

	channel.phaseStep++
	if channel.phaseStep == len(protocolMPISchedule[int(channel.phase)]) {
		channel.phaseStep = 0
		if channel.phase == ProtocolMPIPhaseU3 {
			channel.complete = true
		} else {
			channel.phase++
		}
	}
}

func protocolMPIExpectedShape(
	phase ProtocolMPIPhase,
	operation protocolMPIOperation,
	partitions uint64,
) (MPIPayloadShape, error) {
	switch phase {
	case ProtocolMPIPhaseW0:
		return MPIPayloadShape{G1: 3}, nil
	case ProtocolMPIPhaseW1:
		return MPIPayloadShape{G1: 1}, nil
	case ProtocolMPIPhaseW2:
		return MPIPayloadShape{G1: 3}, nil
	case ProtocolMPIPhaseW3:
		switch operation {
		case protocolMPIOpGather:
			return MPIPayloadShape{Fields: OuterTerminalEvaluationCount}, nil
		case protocolMPIOpScatter:
			return MPIPayloadShape{Fields: 2}, nil
		case protocolMPIOpBroadcast:
			rounds := bits.Len64(partitions) - 1
			return MPIPayloadShape{
				Fields: SumCheckRoundCoefficients*rounds +
					OuterTerminalEvaluationCount + OuterProductCheckClaimCount,
				G1: 2,
			}, nil
		}
	case ProtocolMPIPhaseU0:
		return MPIPayloadShape{G1: 3}, nil
	case ProtocolMPIPhaseU1:
		switch operation {
		case protocolMPIOpGather:
			return MPIPayloadShape{Fields: 3, G1: 1}, nil
		case protocolMPIOpBroadcast:
			return MPIPayloadShape{Fields: 3, G1: 2}, nil
		}
	case ProtocolMPIPhaseU2:
		if operation == protocolMPIOpAggregate {
			return MPIPayloadShape{Fields: 7}, nil
		}
		if operation == protocolMPIOpBroadcast {
			return MPIPayloadShape{Fields: 14}, nil
		}
	case ProtocolMPIPhaseU3:
		if operation == protocolMPIOpAggregate {
			return MPIPayloadShape{G1: 3}, nil
		}
		if operation == protocolMPIOpBroadcast {
			return MPIPayloadShape{G1: 4}, nil
		}
	}
	return MPIPayloadShape{}, fmt.Errorf(
		"%w: no shape for %s %s",
		ErrProtocolMPISchedule, phase, operation,
	)
}

func emptyMPIPayload(payload MPIPayload) bool {
	return len(payload.Fields) == 0 && len(payload.G1) == 0
}

func protocolPayloadShapeError(label string, got, want MPIPayloadShape) error {
	return fmt.Errorf("%w: %s shape %+v, want %+v", ErrMPIPayload, label, got, want)
}

func (channel *ProtocolMPIChannel) sendHeader(rank uint64, header protocolMPIRecordHeader) error {
	encoded, err := encodeProtocolMPIHeader(header)
	if err != nil {
		return err
	}
	if err := channel.transport.SendBytes(encoded, rank); err != nil {
		return err
	}
	channel.recordWire(header.Phase, true, uint64(len(encoded)), 0)
	return nil
}

func (channel *ProtocolMPIChannel) receiveExpectedHeader(
	rank uint64,
	expected protocolMPIRecordHeader,
) error {
	encoded, err := channel.transport.ReceiveBytes(protocolMPIHeaderSize, rank)
	if err != nil {
		return err
	}
	channel.recordWire(expected.Phase, false, uint64(len(encoded)), 0)
	header, err := decodeProtocolMPIHeader(encoded)
	if err != nil {
		return err
	}
	if header != expected {
		return fmt.Errorf("header %+v, want %+v", header, expected)
	}
	return nil
}

func (channel *ProtocolMPIChannel) sendPayload(
	phase ProtocolMPIPhase,
	rank uint64,
	encoded []byte,
) error {
	if err := channel.transport.SendBytes(encoded, rank); err != nil {
		return err
	}
	channel.recordWire(phase, true, uint64(len(encoded)), uint64(len(encoded)))
	return nil
}

func (channel *ProtocolMPIChannel) receivePayload(
	phase ProtocolMPIPhase,
	rank uint64,
	shape MPIPayloadShape,
) (MPIPayload, error) {
	size, err := mpiPayloadByteSize(shape)
	if err != nil {
		return MPIPayload{}, err
	}
	encoded, err := channel.transport.ReceiveBytes(size, rank)
	if err != nil {
		return MPIPayload{}, err
	}
	channel.recordWire(phase, false, uint64(len(encoded)), uint64(len(encoded)))
	return decodeMPIPayload(encoded, shape)
}

func (channel *ProtocolMPIChannel) recordCall(
	phase ProtocolMPIPhase,
	update func(*ProtocolMPIPhaseAccounting),
) {
	channel.accountingMu.Lock()
	defer channel.accountingMu.Unlock()
	update(&channel.accounting.Phases[int(phase)])
	update(&channel.accounting.Total)
}

func (channel *ProtocolMPIChannel) recordWire(
	phase ProtocolMPIPhase,
	sent bool,
	wire uint64,
	payload uint64,
) {
	channel.recordCall(phase, func(accounting *ProtocolMPIPhaseAccounting) {
		if sent {
			accounting.WireBytesSent += wire
			accounting.PayloadBytesSent += payload
		} else {
			accounting.WireBytesRecv += wire
			accounting.PayloadBytesRecv += payload
		}
	})
}

func encodeProtocolMPIHeader(header protocolMPIRecordHeader) ([]byte, error) {
	if !header.Operation.valid() {
		return nil, fmt.Errorf("%w: operation %d", ErrProtocolMPIRecord, header.Operation)
	}
	if !header.Phase.valid() {
		return nil, fmt.Errorf("%w: phase %d", ErrProtocolMPIRecord, header.Phase)
	}
	if _, err := mpiPayloadByteSize(header.Shape); err != nil {
		return nil, err
	}
	encoded := make([]byte, protocolMPIHeaderSize)
	copy(encoded[0:4], "DLPM")
	encoded[4] = mpiRecordVersion
	encoded[5] = byte(header.Operation)
	encoded[6] = byte(header.Phase)
	// encoded[7] is reserved and canonically zero.
	binary.BigEndian.PutUint32(encoded[8:12], header.Sequence)
	binary.BigEndian.PutUint32(encoded[12:16], uint32(header.Shape.Fields))
	binary.BigEndian.PutUint32(encoded[16:20], uint32(header.Shape.G1))
	return encoded, nil
}

func decodeProtocolMPIHeader(encoded []byte) (protocolMPIRecordHeader, error) {
	if len(encoded) != protocolMPIHeaderSize {
		return protocolMPIRecordHeader{}, fmt.Errorf(
			"%w: header length %d",
			ErrProtocolMPIRecord, len(encoded),
		)
	}
	if !bytes.Equal(encoded[0:4], []byte("DLPM")) || encoded[4] != mpiRecordVersion || encoded[7] != 0 {
		return protocolMPIRecordHeader{}, fmt.Errorf("%w: header prefix", ErrProtocolMPIRecord)
	}
	header := protocolMPIRecordHeader{
		Operation: protocolMPIOperation(encoded[5]),
		Phase:     ProtocolMPIPhase(encoded[6]),
		Sequence:  binary.BigEndian.Uint32(encoded[8:12]),
		Shape: MPIPayloadShape{
			Fields: int(binary.BigEndian.Uint32(encoded[12:16])),
			G1:     int(binary.BigEndian.Uint32(encoded[16:20])),
		},
	}
	canonical, err := encodeProtocolMPIHeader(header)
	if err != nil {
		return protocolMPIRecordHeader{}, err
	}
	if !bytes.Equal(canonical, encoded) {
		return protocolMPIRecordHeader{}, fmt.Errorf("%w: non-canonical header", ErrProtocolMPIRecord)
	}
	return header, nil
}
