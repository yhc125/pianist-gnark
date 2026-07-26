package dlinkzg

import (
	"errors"
	"fmt"
	"math/big"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

func TestLocalMPIChannelScheduleAggregationBroadcastAndAccounting(t *testing.T) {
	channel := NewLocalMPIChannel()
	if channel.Rank() != 0 || channel.WorldSize() != 1 {
		t.Fatalf("local channel is rank %d in world %d", channel.Rank(), channel.WorldSize())
	}

	if err := channel.Barrier(MPIPhaseU0); err != nil {
		t.Fatalf("U0 barrier: %v", err)
	}
	local := MPIPayload{
		Fields: []fr.Element{fr.NewElement(7), fr.NewElement(11)},
		G1:     []bn254.G1Affine{mpiTestG1(3)},
	}
	aggregate, err := channel.RootAggregate(MPIPhaseU0, local)
	if err != nil {
		t.Fatalf("U0 aggregate: %v", err)
	}
	assertMPIPayloadEqual(t, aggregate, local)

	broadcast, err := channel.RootBroadcast(MPIPhaseU0, local.Shape(), aggregate)
	if err != nil {
		t.Fatalf("U0 broadcast: %v", err)
	}
	assertMPIPayloadEqual(t, broadcast, local)

	for _, phase := range []MPIPhase{MPIPhaseU1, MPIPhaseU2, MPIPhaseU3} {
		if err := channel.Barrier(phase); err != nil {
			t.Fatalf("%s barrier: %v", phase, err)
		}
	}

	stats := channel.Accounting()
	if stats.Rank != 0 || stats.WorldSize != 1 || stats.Operations != 6 {
		t.Fatalf("unexpected local accounting header: %+v", stats)
	}
	if stats.Total.Barriers != 4 || stats.Total.Aggregations != 1 || stats.Total.Broadcasts != 1 {
		t.Fatalf("unexpected local operation accounting: %+v", stats.Total)
	}
	if stats.Total.WireBytesSent != 0 || stats.Total.WireBytesRecv != 0 ||
		stats.Total.PayloadBytesSent != 0 || stats.Total.PayloadBytesRecv != 0 {
		t.Fatalf("WorldSize=1 must not report network traffic: %+v", stats.Total)
	}
	if stats.Phases[MPIPhaseU0].Barriers != 1 || stats.Phases[MPIPhaseU3].Barriers != 1 {
		t.Fatalf("phase ledger not populated: %+v", stats.Phases)
	}
}

func TestMPIChannelRejectsPhaseRegressionAndJump(t *testing.T) {
	channel := NewLocalMPIChannel()
	if err := channel.Barrier(MPIPhaseU1); !errors.Is(err, ErrMPIPhase) {
		t.Fatalf("starting at U1: got %v, want ErrMPIPhase", err)
	}
	if err := channel.Barrier(MPIPhaseU0); err != nil {
		t.Fatalf("starting at U0: %v", err)
	}
	if err := channel.Barrier(MPIPhaseU2); !errors.Is(err, ErrMPIPhase) {
		t.Fatalf("jumping U0 to U2: got %v, want ErrMPIPhase", err)
	}
	if err := channel.Barrier(MPIPhaseU1); err != nil {
		t.Fatalf("advancing U0 to U1: %v", err)
	}
	if err := channel.Barrier(MPIPhaseU0); !errors.Is(err, ErrMPIPhase) {
		t.Fatalf("regressing U1 to U0: got %v, want ErrMPIPhase", err)
	}
}

func TestMPIChannelThreeRankAggregateAndBroadcast(t *testing.T) {
	const worldSize = 3
	network := newFakeMPINetwork(worldSize)
	channels := make([]*MPIChannel, worldSize)
	for rank := 0; rank < worldSize; rank++ {
		channel, err := newMPIChannel(network.transport(uint64(rank)))
		if err != nil {
			t.Fatalf("new channel %d: %v", rank, err)
		}
		channels[rank] = channel
	}

	type rankResult struct {
		payload MPIPayload
		err     error
	}
	results := make([]rankResult, worldSize)
	var wg sync.WaitGroup
	for rank := 0; rank < worldSize; rank++ {
		rank := rank
		wg.Add(1)
		go func() {
			defer wg.Done()
			channel := channels[rank]
			if err := channel.Barrier(MPIPhaseU0); err != nil {
				results[rank].err = fmt.Errorf("barrier: %w", err)
				return
			}
			local := MPIPayload{
				Fields: []fr.Element{fr.NewElement(uint64(rank + 1))},
				G1:     []bn254.G1Affine{mpiTestG1(int64(rank + 1))},
			}
			aggregate, err := channel.RootAggregate(MPIPhaseU0, local)
			if err != nil {
				results[rank].err = fmt.Errorf("aggregate: %w", err)
				return
			}
			broadcast, err := channel.RootBroadcast(MPIPhaseU0, local.Shape(), aggregate)
			if err != nil {
				results[rank].err = fmt.Errorf("broadcast: %w", err)
				return
			}
			results[rank].payload = broadcast
		}()
	}
	wg.Wait()

	want := MPIPayload{
		Fields: []fr.Element{fr.NewElement(6)},
		G1:     []bn254.G1Affine{mpiTestG1(6)},
	}
	for rank := range results {
		if results[rank].err != nil {
			t.Fatalf("rank %d: %v", rank, results[rank].err)
		}
		assertMPIPayloadEqual(t, results[rank].payload, want)
	}

	rootStats := channels[0].Accounting().Total
	if rootStats.WireBytesSent != 208 || rootStats.WireBytesRecv != 208 ||
		rootStats.PayloadBytesSent != 128 || rootStats.PayloadBytesRecv != 128 {
		t.Fatalf("unexpected root byte accounting: %+v", rootStats)
	}
	for rank := 1; rank < worldSize; rank++ {
		workerStats := channels[rank].Accounting().Total
		if workerStats.WireBytesSent != 104 || workerStats.WireBytesRecv != 104 ||
			workerStats.PayloadBytesSent != 64 || workerStats.PayloadBytesRecv != 64 {
			t.Fatalf("unexpected worker %d byte accounting: %+v", rank, workerStats)
		}
	}
}

func TestMPIPayloadCanonicalDecoding(t *testing.T) {
	payload := MPIPayload{
		Fields: []fr.Element{fr.NewElement(19)},
		G1:     []bn254.G1Affine{mpiTestG1(23)},
	}
	encoded, err := encodeMPIPayload(payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := decodeMPIPayload(encoded, payload.Shape())
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	assertMPIPayloadEqual(t, decoded, payload)

	nonCanonicalField := fr.Modulus().Bytes()
	bad := make([]byte, fr.Bytes)
	copy(bad[fr.Bytes-len(nonCanonicalField):], nonCanonicalField)
	if _, err := decodeMPIPayload(bad, MPIPayloadShape{Fields: 1}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("non-canonical field: got %v, want ErrMPIPayload", err)
	}

	badPoint := append([]byte(nil), encoded[fr.Bytes:]...)
	badPoint[0] = 0x40 // compressed-infinity tag with nonzero trailing bytes.
	if _, err := decodeMPIPayload(badPoint, MPIPayloadShape{G1: 1}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("non-canonical G1: got %v, want ErrMPIPayload", err)
	}
}

func TestNewMPIChannelRejectsUninitializedWorld(t *testing.T) {
	oldRank, oldSize := mpi.SelfRank, mpi.WorldSize
	defer func() {
		mpi.SelfRank, mpi.WorldSize = oldRank, oldSize
	}()
	mpi.SelfRank, mpi.WorldSize = 0, 0
	if _, err := NewMPIChannel(); !errors.Is(err, ErrMPIConfiguration) {
		t.Fatalf("uninitialized simpleMPI: got %v, want ErrMPIConfiguration", err)
	}
}

func TestNewMPIChannelUsesWorldSizeOneWithoutNetworkIO(t *testing.T) {
	oldRank, oldSize := mpi.SelfRank, mpi.WorldSize
	defer func() {
		mpi.SelfRank, mpi.WorldSize = oldRank, oldSize
	}()
	mpi.SelfRank, mpi.WorldSize = 0, 1

	channel, err := NewMPIChannel()
	if err != nil {
		t.Fatalf("new WorldSize=1 channel: %v", err)
	}
	if err := channel.Barrier(MPIPhaseU0); err != nil {
		t.Fatalf("WorldSize=1 barrier: %v", err)
	}
	local := MPIPayload{Fields: []fr.Element{fr.NewElement(31)}}
	aggregate, err := channel.RootAggregate(MPIPhaseU0, local)
	if err != nil {
		t.Fatalf("WorldSize=1 aggregate: %v", err)
	}
	assertMPIPayloadEqual(t, aggregate, local)
	if channel.Accounting().Total.WireBytesSent != 0 || channel.Accounting().Total.WireBytesRecv != 0 {
		t.Fatalf("WorldSize=1 simpleMPI channel performed network I/O: %+v", channel.Accounting().Total)
	}
}

func mpiTestG1(scalar int64) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var result bn254.G1Affine
	result.ScalarMultiplication(&generator, big.NewInt(scalar))
	return result
}

func assertMPIPayloadEqual(t *testing.T, got, want MPIPayload) {
	t.Helper()
	if len(got.Fields) != len(want.Fields) || len(got.G1) != len(want.G1) {
		t.Fatalf("payload shape %+v, want %+v", got.Shape(), want.Shape())
	}
	for i := range want.Fields {
		if !got.Fields[i].Equal(&want.Fields[i]) {
			t.Fatalf("field %d differs", i)
		}
	}
	for i := range want.G1 {
		if !got.G1[i].Equal(&want.G1[i]) {
			t.Fatalf("G1 %d differs", i)
		}
	}
}

type fakeMPIMessage struct {
	bytes []byte
	from  uint64
}

type fakeMPINetwork struct {
	size  uint64
	links map[[2]uint64]chan fakeMPIMessage
}

func newFakeMPINetwork(size uint64) *fakeMPINetwork {
	network := &fakeMPINetwork{
		size:  size,
		links: make(map[[2]uint64]chan fakeMPIMessage),
	}
	for rank := uint64(1); rank < size; rank++ {
		network.links[[2]uint64{0, rank}] = make(chan fakeMPIMessage, 8)
		network.links[[2]uint64{rank, 0}] = make(chan fakeMPIMessage, 8)
	}
	return network
}

func (n *fakeMPINetwork) transport(rank uint64) *fakeMPITransport {
	return &fakeMPITransport{network: n, rank: rank}
}

type fakeMPITransport struct {
	network *fakeMPINetwork
	rank    uint64
}

func (t *fakeMPITransport) Rank() uint64 { return t.rank }
func (t *fakeMPITransport) Size() uint64 { return t.network.size }

func (t *fakeMPITransport) SendBytes(buf []byte, rank uint64) error {
	destination := rank
	if t.rank != 0 {
		destination = 0
	}
	link, ok := t.network.links[[2]uint64{t.rank, destination}]
	if !ok {
		return fmt.Errorf("no fake MPI link %d -> %d", t.rank, destination)
	}
	link <- fakeMPIMessage{bytes: append([]byte(nil), buf...), from: t.rank}
	return nil
}

func (t *fakeMPITransport) ReceiveBytes(size uint64, rank uint64) ([]byte, error) {
	source := rank
	if t.rank != 0 {
		source = 0
	}
	link, ok := t.network.links[[2]uint64{source, t.rank}]
	if !ok {
		return nil, fmt.Errorf("no fake MPI link %d -> %d", source, t.rank)
	}
	message := <-link
	if uint64(len(message.bytes)) != size {
		return nil, fmt.Errorf("fake MPI message from %d has %d bytes, want %d", message.from, len(message.bytes), size)
	}
	return message.bytes, nil
}
