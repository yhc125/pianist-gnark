package dlinkzg

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

const (
	mpiRecordVersion byte = 1
	mpiHeaderSize         = 20
)

var (
	// ErrMPIConfiguration reports an invalid rank or world size.
	ErrMPIConfiguration = errors.New("dlinkzg: invalid MPI configuration")
	// ErrMPIPhase reports an invalid or out-of-order U0--U3 phase.
	ErrMPIPhase = errors.New("dlinkzg: invalid MPI phase")
	// ErrMPIRecord reports a malformed, misordered, or non-canonical wire record.
	ErrMPIRecord = errors.New("dlinkzg: invalid MPI record")
	// ErrMPIPayload reports a payload whose shape or group elements are invalid.
	ErrMPIPayload = errors.New("dlinkzg: invalid MPI payload")
)

// MPIPhase names the four distributed opening responses. The names match the
// U0--U3 records consumed by OpeningTranscript; the transport does not derive
// challenges or construct Coordinator-only values.
type MPIPhase uint8

const (
	MPIPhaseU0 MPIPhase = iota
	MPIPhaseU1
	MPIPhaseU2
	MPIPhaseU3
)

func (p MPIPhase) String() string {
	switch p {
	case MPIPhaseU0:
		return "DLinKZG/U0"
	case MPIPhaseU1:
		return "DLinKZG/U1"
	case MPIPhaseU2:
		return "DLinKZG/U2"
	case MPIPhaseU3:
		return "DLinKZG/U3"
	default:
		return fmt.Sprintf("DLinKZG/invalid-%d", uint8(p))
	}
}

func (p MPIPhase) valid() bool {
	return p >= MPIPhaseU0 && p <= MPIPhaseU3
}

// MPIPayload is a fixed-shape additive share. Fields and G1 points retain
// their displayed order. RootAggregate adds the corresponding entries in
// ascending rank order.
type MPIPayload struct {
	Fields []fr.Element
	G1     []bn254.G1Affine
}

// MPIPayloadShape fixes the number of field and compressed G1 encodings in a
// record. It is supplied explicitly for broadcasts because non-root parties
// do not possess the root payload before receiving it.
type MPIPayloadShape struct {
	Fields int
	G1     int
}

// Shape returns the fixed wire shape of p.
func (p MPIPayload) Shape() MPIPayloadShape {
	return MPIPayloadShape{Fields: len(p.Fields), G1: len(p.G1)}
}

func (p MPIPayload) clone() MPIPayload {
	return MPIPayload{
		Fields: append([]fr.Element(nil), p.Fields...),
		G1:     append([]bn254.G1Affine(nil), p.G1...),
	}
}

// MPIPhaseAccounting separates protocol payload bytes from framing and
// barrier bytes. Wire bytes are the bytes successfully handed to or returned
// by simpleMPI. Calls count successful local operations, not cluster-wide
// operations.
type MPIPhaseAccounting struct {
	Barriers         uint64 `json:"barriers"`
	Aggregations     uint64 `json:"aggregations"`
	Broadcasts       uint64 `json:"broadcasts"`
	PayloadBytesSent uint64 `json:"payload_bytes_sent"`
	PayloadBytesRecv uint64 `json:"payload_bytes_received"`
	WireBytesSent    uint64 `json:"wire_bytes_sent"`
	WireBytesRecv    uint64 `json:"wire_bytes_received"`
}

// MPIAccounting is a point-in-time copy of one party's communication ledger.
// The four phase entries are indexed by MPIPhaseU0 through MPIPhaseU3.
type MPIAccounting struct {
	Rank       uint64                `json:"rank"`
	WorldSize  uint64                `json:"world_size"`
	Operations uint64                `json:"operations"`
	Total      MPIPhaseAccounting    `json:"total"`
	Phases     [4]MPIPhaseAccounting `json:"phases"`
}

type mpiOperation byte

const (
	mpiOpBarrier mpiOperation = iota + 1
	mpiOpBarrierAck
	mpiOpAggregate
	mpiOpBroadcast
)

type mpiRecordHeader struct {
	Operation mpiOperation
	Phase     MPIPhase
	Sequence  uint32
	Shape     MPIPayloadShape
}

type mpiByteTransport interface {
	Rank() uint64
	Size() uint64
	SendBytes(buf []byte, rank uint64) error
	ReceiveBytes(size uint64, rank uint64) ([]byte, error)
}

type simpleMPITransport struct {
	rank uint64
	size uint64
}

func (t simpleMPITransport) Rank() uint64 { return t.rank }
func (t simpleMPITransport) Size() uint64 { return t.size }
func (simpleMPITransport) SendBytes(buf []byte, rank uint64) error {
	return mpi.SendBytes(buf, rank)
}
func (simpleMPITransport) ReceiveBytes(size uint64, rank uint64) ([]byte, error) {
	return mpi.ReceiveBytes(size, rank)
}

type localMPITransport struct{}

func (localMPITransport) Rank() uint64 { return 0 }
func (localMPITransport) Size() uint64 { return 1 }
func (localMPITransport) SendBytes([]byte, uint64) error {
	return errors.New("dlinkzg: local MPI transport cannot send")
}
func (localMPITransport) ReceiveBytes(uint64, uint64) ([]byte, error) {
	return nil, errors.New("dlinkzg: local MPI transport cannot receive")
}

// MPIChannel is a deterministic root/worker schedule over simpleMPI's byte
// streams. Every wire operation carries a phase, operation type, sequence,
// and fixed payload shape. Root receives aggregate shares in ascending rank
// order and broadcasts in the same order.
//
// MPIChannel is intentionally not safe for concurrent method calls. A
// DLinKZG party must execute the public U0--U3 schedule serially.
type MPIChannel struct {
	transport mpiByteTransport
	rank      uint64
	size      uint64
	sequence  uint32
	started   bool
	phase     MPIPhase

	accountingMu sync.Mutex
	accounting   MPIAccounting
}

// NewMPIChannel binds a channel to the simpleMPI world initialized by the
// benchmark executable.
func NewMPIChannel() (*MPIChannel, error) {
	return newMPIChannel(simpleMPITransport{rank: mpi.SelfRank, size: mpi.WorldSize})
}

// NewLocalMPIChannel returns a no-network, WorldSize=1 channel. It is useful
// for deterministic unit tests and serial benchmark checks without mutating
// simpleMPI's process-global state.
func NewLocalMPIChannel() *MPIChannel {
	channel, err := newMPIChannel(localMPITransport{})
	if err != nil {
		panic(err)
	}
	return channel
}

func newMPIChannel(transport mpiByteTransport) (*MPIChannel, error) {
	if transport == nil {
		return nil, fmt.Errorf("%w: nil transport", ErrMPIConfiguration)
	}
	rank, size := transport.Rank(), transport.Size()
	if size == 0 || rank >= size {
		return nil, fmt.Errorf("%w: rank %d in world size %d", ErrMPIConfiguration, rank, size)
	}
	return &MPIChannel{
		transport: transport,
		rank:      rank,
		size:      size,
		accounting: MPIAccounting{
			Rank:      rank,
			WorldSize: size,
		},
	}, nil
}

// Rank returns this party's rank. Rank zero is the Coordinator/root.
func (c *MPIChannel) Rank() uint64 { return c.rank }

// WorldSize returns the fixed number of parties in the channel.
func (c *MPIChannel) WorldSize() uint64 { return c.size }

// Barrier performs a deterministic two-way barrier. Workers send one header;
// root receives headers in ascending rank order and then acknowledges workers
// in ascending rank order. The phase and operation sequence are authenticated
// structurally by the fixed header (cryptographic transcript binding remains
// the responsibility of OpeningTranscript).
func (c *MPIChannel) Barrier(phase MPIPhase) error {
	if err := c.beginOperation(phase); err != nil {
		return err
	}
	header := mpiRecordHeader{
		Operation: mpiOpBarrier,
		Phase:     phase,
		Sequence:  c.sequence,
	}
	ack := header
	ack.Operation = mpiOpBarrierAck

	if c.size > 1 {
		if c.rank == 0 {
			for rank := uint64(1); rank < c.size; rank++ {
				if _, err := c.receiveHeader(rank, header); err != nil {
					return fmt.Errorf("%w: barrier receive from rank %d: %v", ErrMPIRecord, rank, err)
				}
			}
			for rank := uint64(1); rank < c.size; rank++ {
				if err := c.sendHeader(rank, ack); err != nil {
					return fmt.Errorf("dlinkzg: barrier acknowledge rank %d: %w", rank, err)
				}
			}
		} else {
			if err := c.sendHeader(0, header); err != nil {
				return fmt.Errorf("dlinkzg: barrier send: %w", err)
			}
			if _, err := c.receiveHeader(0, ack); err != nil {
				return fmt.Errorf("%w: barrier acknowledge: %v", ErrMPIRecord, err)
			}
		}
	}

	c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Barriers++ })
	c.finishOperation()
	return nil
}

// RootAggregate sends one fixed-shape additive share from every worker to
// rank zero. Root includes its local share and returns the component-wise sum;
// workers return an empty payload after their send completes. No aggregate is
// implicitly broadcast.
func (c *MPIChannel) RootAggregate(phase MPIPhase, local MPIPayload) (MPIPayload, error) {
	if err := c.beginOperation(phase); err != nil {
		return MPIPayload{}, err
	}
	encoded, err := encodeMPIPayload(local)
	if err != nil {
		return MPIPayload{}, err
	}
	header := mpiRecordHeader{
		Operation: mpiOpAggregate,
		Phase:     phase,
		Sequence:  c.sequence,
		Shape:     local.Shape(),
	}

	if c.size == 1 {
		result := local.clone()
		c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Aggregations++ })
		c.finishOperation()
		return result, nil
	}

	if c.rank != 0 {
		if err := c.sendHeader(0, header); err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: aggregate header send: %w", err)
		}
		if err := c.sendPayload(phase, 0, encoded); err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: aggregate payload send: %w", err)
		}
		c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Aggregations++ })
		c.finishOperation()
		return MPIPayload{}, nil
	}

	result := local.clone()
	for rank := uint64(1); rank < c.size; rank++ {
		if _, err := c.receiveHeader(rank, header); err != nil {
			return MPIPayload{}, fmt.Errorf("%w: aggregate header from rank %d: %v", ErrMPIRecord, rank, err)
		}
		share, err := c.receivePayload(phase, rank, header.Shape)
		if err != nil {
			return MPIPayload{}, fmt.Errorf("dlinkzg: aggregate payload from rank %d: %w", rank, err)
		}
		addMPIPayload(&result, share)
	}

	c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Aggregations++ })
	c.finishOperation()
	return result, nil
}

// RootBroadcast sends rootPayload from rank zero to every worker and returns
// that payload on every rank. All callers provide the same fixed shape;
// workers may pass an empty rootPayload. Broadcast does not add shares.
func (c *MPIChannel) RootBroadcast(
	phase MPIPhase,
	shape MPIPayloadShape,
	rootPayload MPIPayload,
) (MPIPayload, error) {
	if err := c.beginOperation(phase); err != nil {
		return MPIPayload{}, err
	}
	if _, err := mpiPayloadByteSize(shape); err != nil {
		return MPIPayload{}, err
	}
	header := mpiRecordHeader{
		Operation: mpiOpBroadcast,
		Phase:     phase,
		Sequence:  c.sequence,
		Shape:     shape,
	}

	if c.rank == 0 {
		if rootPayload.Shape() != shape {
			return MPIPayload{}, fmt.Errorf(
				"%w: root broadcast shape %+v, want %+v",
				ErrMPIPayload, rootPayload.Shape(), shape,
			)
		}
		encoded, err := encodeMPIPayload(rootPayload)
		if err != nil {
			return MPIPayload{}, err
		}
		for rank := uint64(1); rank < c.size; rank++ {
			if err := c.sendHeader(rank, header); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: broadcast header to rank %d: %w", rank, err)
			}
			if err := c.sendPayload(phase, rank, encoded); err != nil {
				return MPIPayload{}, fmt.Errorf("dlinkzg: broadcast payload to rank %d: %w", rank, err)
			}
		}
		result := rootPayload.clone()
		c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Broadcasts++ })
		c.finishOperation()
		return result, nil
	}

	if _, err := c.receiveHeader(0, header); err != nil {
		return MPIPayload{}, fmt.Errorf("%w: broadcast header: %v", ErrMPIRecord, err)
	}
	result, err := c.receivePayload(phase, 0, shape)
	if err != nil {
		return MPIPayload{}, fmt.Errorf("dlinkzg: broadcast payload: %w", err)
	}
	c.recordCall(phase, func(a *MPIPhaseAccounting) { a.Broadcasts++ })
	c.finishOperation()
	return result, nil
}

// Accounting returns an immutable copy of the current communication ledger.
func (c *MPIChannel) Accounting() MPIAccounting {
	c.accountingMu.Lock()
	defer c.accountingMu.Unlock()
	return c.accounting
}

func (c *MPIChannel) beginOperation(phase MPIPhase) error {
	if !phase.valid() {
		return fmt.Errorf("%w: %s", ErrMPIPhase, phase)
	}
	if !c.started {
		if phase != MPIPhaseU0 {
			return fmt.Errorf("%w: first phase is %s, want %s", ErrMPIPhase, phase, MPIPhaseU0)
		}
		c.started = true
		c.phase = phase
		return nil
	}
	if phase < c.phase {
		return fmt.Errorf("%w: phase regressed from %s to %s", ErrMPIPhase, c.phase, phase)
	}
	if phase > c.phase+1 {
		return fmt.Errorf("%w: phase jumped from %s to %s", ErrMPIPhase, c.phase, phase)
	}
	if phase > c.phase {
		c.phase = phase
	}
	return nil
}

func (c *MPIChannel) finishOperation() {
	c.sequence++
	c.accountingMu.Lock()
	c.accounting.Operations++
	c.accountingMu.Unlock()
}

func (c *MPIChannel) sendHeader(rank uint64, header mpiRecordHeader) error {
	encoded, err := encodeMPIHeader(header)
	if err != nil {
		return err
	}
	if err := c.transport.SendBytes(encoded, rank); err != nil {
		return err
	}
	c.recordWire(header.Phase, true, uint64(len(encoded)), 0)
	return nil
}

func (c *MPIChannel) receiveHeader(rank uint64, expected mpiRecordHeader) (mpiRecordHeader, error) {
	encoded, err := c.transport.ReceiveBytes(mpiHeaderSize, rank)
	if err != nil {
		return mpiRecordHeader{}, err
	}
	c.recordWire(expected.Phase, false, uint64(len(encoded)), 0)
	header, err := decodeMPIHeader(encoded)
	if err != nil {
		return mpiRecordHeader{}, err
	}
	if header != expected {
		return mpiRecordHeader{}, fmt.Errorf("header %+v, want %+v", header, expected)
	}
	return header, nil
}

func (c *MPIChannel) sendPayload(phase MPIPhase, rank uint64, encoded []byte) error {
	if err := c.transport.SendBytes(encoded, rank); err != nil {
		return err
	}
	c.recordWire(phase, true, uint64(len(encoded)), uint64(len(encoded)))
	return nil
}

func (c *MPIChannel) receivePayload(phase MPIPhase, rank uint64, shape MPIPayloadShape) (MPIPayload, error) {
	size, err := mpiPayloadByteSize(shape)
	if err != nil {
		return MPIPayload{}, err
	}
	encoded, err := c.transport.ReceiveBytes(size, rank)
	if err != nil {
		return MPIPayload{}, err
	}
	c.recordWire(phase, false, uint64(len(encoded)), uint64(len(encoded)))
	return decodeMPIPayload(encoded, shape)
}

func (c *MPIChannel) recordCall(phase MPIPhase, update func(*MPIPhaseAccounting)) {
	c.accountingMu.Lock()
	defer c.accountingMu.Unlock()
	update(&c.accounting.Phases[int(phase)])
	update(&c.accounting.Total)
}

func (c *MPIChannel) recordWire(phase MPIPhase, sent bool, wire, payload uint64) {
	c.recordCall(phase, func(a *MPIPhaseAccounting) {
		if sent {
			a.WireBytesSent += wire
			a.PayloadBytesSent += payload
		} else {
			a.WireBytesRecv += wire
			a.PayloadBytesRecv += payload
		}
	})
}

func encodeMPIHeader(header mpiRecordHeader) ([]byte, error) {
	if !header.Phase.valid() {
		return nil, fmt.Errorf("%w: %s", ErrMPIPhase, header.Phase)
	}
	if header.Operation < mpiOpBarrier || header.Operation > mpiOpBroadcast {
		return nil, fmt.Errorf("%w: operation %d", ErrMPIRecord, header.Operation)
	}
	if _, err := mpiPayloadByteSize(header.Shape); err != nil {
		return nil, err
	}
	if (header.Operation == mpiOpBarrier || header.Operation == mpiOpBarrierAck) && header.Shape != (MPIPayloadShape{}) {
		return nil, fmt.Errorf("%w: barrier has nonempty payload shape", ErrMPIRecord)
	}

	encoded := make([]byte, mpiHeaderSize)
	copy(encoded[0:4], "DLMP")
	encoded[4] = mpiRecordVersion
	encoded[5] = byte(header.Operation)
	encoded[6] = byte(header.Phase)
	// encoded[7] is reserved and canonically zero.
	binary.BigEndian.PutUint32(encoded[8:12], header.Sequence)
	binary.BigEndian.PutUint32(encoded[12:16], uint32(header.Shape.Fields))
	binary.BigEndian.PutUint32(encoded[16:20], uint32(header.Shape.G1))
	return encoded, nil
}

func decodeMPIHeader(encoded []byte) (mpiRecordHeader, error) {
	if len(encoded) != mpiHeaderSize {
		return mpiRecordHeader{}, fmt.Errorf("%w: header length %d", ErrMPIRecord, len(encoded))
	}
	if !bytes.Equal(encoded[0:4], []byte("DLMP")) || encoded[4] != mpiRecordVersion || encoded[7] != 0 {
		return mpiRecordHeader{}, fmt.Errorf("%w: header prefix", ErrMPIRecord)
	}
	header := mpiRecordHeader{
		Operation: mpiOperation(encoded[5]),
		Phase:     MPIPhase(encoded[6]),
		Sequence:  binary.BigEndian.Uint32(encoded[8:12]),
		Shape: MPIPayloadShape{
			Fields: int(binary.BigEndian.Uint32(encoded[12:16])),
			G1:     int(binary.BigEndian.Uint32(encoded[16:20])),
		},
	}
	if _, err := encodeMPIHeader(header); err != nil {
		return mpiRecordHeader{}, err
	}
	return header, nil
}

func mpiPayloadByteSize(shape MPIPayloadShape) (uint64, error) {
	if shape.Fields < 0 || shape.G1 < 0 {
		return 0, fmt.Errorf("%w: negative shape %+v", ErrMPIPayload, shape)
	}
	fields, points := uint64(shape.Fields), uint64(shape.G1)
	if fields > math.MaxUint32 || points > math.MaxUint32 {
		return 0, fmt.Errorf("%w: shape exceeds uint32 %+v", ErrMPIPayload, shape)
	}
	if fields > (math.MaxUint64-points*bn254.SizeOfG1AffineCompressed)/fr.Bytes {
		return 0, fmt.Errorf("%w: encoded size overflow %+v", ErrMPIPayload, shape)
	}
	return fields*fr.Bytes + points*bn254.SizeOfG1AffineCompressed, nil
}

func encodeMPIPayload(payload MPIPayload) ([]byte, error) {
	size, err := mpiPayloadByteSize(payload.Shape())
	if err != nil {
		return nil, err
	}
	if size > uint64(math.MaxInt) {
		return nil, fmt.Errorf("%w: encoded payload too large", ErrMPIPayload)
	}
	encoded := make([]byte, 0, int(size))
	for i := range payload.Fields {
		value := payload.Fields[i].Bytes()
		encoded = append(encoded, value[:]...)
	}
	for i := range payload.G1 {
		if !payload.G1[i].IsOnCurve() || !payload.G1[i].IsInSubGroup() {
			return nil, fmt.Errorf("%w: G1 element %d", ErrMPIPayload, i)
		}
		point := payload.G1[i].Bytes()
		encoded = append(encoded, point[:]...)
	}
	return encoded, nil
}

func decodeMPIPayload(encoded []byte, shape MPIPayloadShape) (MPIPayload, error) {
	size, err := mpiPayloadByteSize(shape)
	if err != nil {
		return MPIPayload{}, err
	}
	if uint64(len(encoded)) != size {
		return MPIPayload{}, fmt.Errorf("%w: encoded length %d, want %d", ErrMPIPayload, len(encoded), size)
	}

	result := MPIPayload{
		Fields: make([]fr.Element, shape.Fields),
		G1:     make([]bn254.G1Affine, shape.G1),
	}
	offset := 0
	for i := range result.Fields {
		fieldBytes := encoded[offset : offset+fr.Bytes]
		result.Fields[i].SetBytes(fieldBytes)
		canonical := result.Fields[i].Bytes()
		if !bytes.Equal(canonical[:], fieldBytes) {
			return MPIPayload{}, fmt.Errorf("%w: non-canonical field element %d", ErrMPIPayload, i)
		}
		offset += fr.Bytes
	}
	for i := range result.G1 {
		pointBytes := encoded[offset : offset+bn254.SizeOfG1AffineCompressed]
		read, err := result.G1[i].SetBytes(pointBytes)
		if err != nil || read != bn254.SizeOfG1AffineCompressed {
			return MPIPayload{}, fmt.Errorf("%w: G1 element %d", ErrMPIPayload, i)
		}
		canonical := result.G1[i].Bytes()
		if !bytes.Equal(canonical[:], pointBytes) {
			return MPIPayload{}, fmt.Errorf("%w: non-canonical G1 element %d", ErrMPIPayload, i)
		}
		offset += bn254.SizeOfG1AffineCompressed
	}
	return result, nil
}

func addMPIPayload(destination *MPIPayload, share MPIPayload) {
	for i := range destination.Fields {
		destination.Fields[i].Add(&destination.Fields[i], &share.Fields[i])
	}
	for i := range destination.G1 {
		destination.G1[i].Add(&destination.G1[i], &share.G1[i])
	}
}
