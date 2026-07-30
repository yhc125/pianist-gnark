package dlinkzg

import (
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestProtocolMPIChannelTopology(t *testing.T) {
	for _, partitions := range []uint64{2, 4} {
		worldSize := partitions
		network := newFakeMPINetwork(worldSize)
		for rank := uint64(0); rank < worldSize; rank++ {
			channel, err := newProtocolMPIChannel(network.transport(rank))
			if err != nil {
				t.Fatalf("M=%d rank=%d: %v", partitions, rank, err)
			}
			if channel.Rank() != rank || channel.WorldSize() != worldSize ||
				channel.PartitionCount() != partitions {
				t.Fatalf("M=%d rank=%d: wrong topology", partitions, rank)
			}
			slot, party := channel.PartySlot()
			if !party || slot != rank {
				t.Fatalf("rank %d maps to slot %d, party=%t", rank, slot, party)
			}
			if channel.IsCoordinator() != (rank == 0) {
				t.Fatalf("rank %d coordinator=%t", rank, channel.IsCoordinator())
			}
		}
	}
}

func TestProtocolMPIChannelRejectsInvalidTopology(t *testing.T) {
	tests := []struct {
		rank uint64
		size uint64
	}{
		{rank: 0, size: 0},
		{rank: 0, size: 1},
		{rank: 0, size: 3},
		{rank: 0, size: 5},
		{rank: 0, size: 6},
		{rank: 4, size: 4},
	}
	for _, test := range tests {
		transport := &protocolStaticTransport{rank: test.rank, size: test.size}
		if _, err := newProtocolMPIChannel(transport); !errors.Is(err, ErrProtocolMPIConfiguration) {
			t.Fatalf("rank=%d size=%d: got %v, want configuration error", test.rank, test.size, err)
		}
	}
	if _, err := newProtocolMPIChannel(nil); !errors.Is(err, ErrProtocolMPIConfiguration) {
		t.Fatalf("nil transport: got %v", err)
	}
}

func TestProtocolMPIShapeLedger(t *testing.T) {
	base := []struct {
		phase     ProtocolMPIPhase
		operation protocolMPIOperation
		shape     MPIPayloadShape
	}{
		{ProtocolMPIPhaseW0, protocolMPIOpAggregate, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseW0, protocolMPIOpBroadcast, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseW1, protocolMPIOpAggregate, MPIPayloadShape{G1: 1}},
		{ProtocolMPIPhaseW1, protocolMPIOpBroadcast, MPIPayloadShape{G1: 1}},
		{ProtocolMPIPhaseW2, protocolMPIOpAggregate, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseW2, protocolMPIOpBroadcast, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseW3, protocolMPIOpGather, MPIPayloadShape{Fields: 21}},
		{ProtocolMPIPhaseW3, protocolMPIOpScatter, MPIPayloadShape{Fields: 2}},
		{ProtocolMPIPhaseU0, protocolMPIOpAggregate, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseU0, protocolMPIOpBroadcast, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseU1, protocolMPIOpGather, MPIPayloadShape{Fields: 3, G1: 1}},
		{ProtocolMPIPhaseU1, protocolMPIOpBroadcast, MPIPayloadShape{Fields: 3, G1: 2}},
		{ProtocolMPIPhaseU2, protocolMPIOpAggregate, MPIPayloadShape{Fields: 6}},
		{ProtocolMPIPhaseU2, protocolMPIOpBroadcast, MPIPayloadShape{Fields: 14}},
		{ProtocolMPIPhaseU3, protocolMPIOpAggregate, MPIPayloadShape{G1: 3}},
		{ProtocolMPIPhaseU3, protocolMPIOpBroadcast, MPIPayloadShape{G1: 4}},
	}
	for _, partitions := range []uint64{2, 4} {
		for _, entry := range base {
			got, err := protocolMPIExpectedShape(entry.phase, entry.operation, partitions)
			if err != nil {
				t.Fatalf("M=%d %s %s: %v", partitions, entry.phase, entry.operation, err)
			}
			if got != entry.shape {
				t.Fatalf("M=%d %s %s: got %+v, want %+v", partitions, entry.phase, entry.operation, got, entry.shape)
			}
		}
		w3, err := protocolMPIExpectedShape(ProtocolMPIPhaseW3, protocolMPIOpBroadcast, partitions)
		if err != nil {
			t.Fatal(err)
		}
		wantFields := 32
		if partitions == 4 {
			wantFields = 38
		}
		if w3 != (MPIPayloadShape{Fields: wantFields, G1: 2}) {
			t.Fatalf("M=%d W3 broadcast: got %+v", partitions, w3)
		}
	}
}

func TestProtocolMPIChannelFullSchedule(t *testing.T) {
	for _, partitions := range []uint64{2, 4} {
		partitions := partitions
		t.Run(fmt.Sprintf("M=%d", partitions), func(t *testing.T) {
			t.Parallel()
			runProtocolMPIFullSchedule(t, partitions)
		})
	}
}

type protocolFullResult struct {
	err         error
	aggregates  [8]MPIPayload
	broadcasts  [8]MPIPayload
	w3Gather    []MPIPayload
	u1Gather    []MPIPayload
	w3Scatter   MPIPayload
	accounting  ProtocolMPIAccounting
	partySlot   uint64
	isParty     bool
	coordinator bool
}

func runProtocolMPIFullSchedule(t *testing.T, partitions uint64) {
	t.Helper()
	worldSize := partitions
	network := newFakeMPINetwork(worldSize)
	channels := make([]*ProtocolMPIChannel, worldSize)
	for rank := uint64(0); rank < worldSize; rank++ {
		channel, err := newProtocolMPIChannel(network.transport(rank))
		if err != nil {
			t.Fatalf("new channel rank %d: %v", rank, err)
		}
		channels[rank] = channel
	}

	results := make([]protocolFullResult, worldSize)
	var wait sync.WaitGroup
	for rank := uint64(0); rank < worldSize; rank++ {
		rank := rank
		wait.Add(1)
		go func() {
			defer wait.Done()
			channel := channels[rank]
			results[rank].coordinator = channel.IsCoordinator()
			results[rank].partySlot, results[rank].isParty = channel.PartySlot()

			for _, phase := range []ProtocolMPIPhase{
				ProtocolMPIPhaseW0, ProtocolMPIPhaseW1, ProtocolMPIPhaseW2,
			} {
				if err := runProtocolAggregateBroadcast(
					channel, phase, rank, &results[rank],
				); err != nil {
					results[rank].err = err
					return
				}
			}

			w3GatherShape, _ := protocolMPIExpectedShape(
				ProtocolMPIPhaseW3, protocolMPIOpGather, partitions,
			)
			w3Local := protocolPayloadWithScalar(w3GatherShape, 1000+rank)
			gathered, err := channel.GatherWorkers(ProtocolMPIPhaseW3, w3GatherShape, w3Local)
			if err != nil {
				results[rank].err = fmt.Errorf("W3 gather: %w", err)
				return
			}
			results[rank].w3Gather = gathered

			scatterShape, _ := protocolMPIExpectedShape(
				ProtocolMPIPhaseW3, protocolMPIOpScatter, partitions,
			)
			var scatterPayloads []MPIPayload
			if rank == 0 {
				scatterPayloads = make([]MPIPayload, partitions)
				for slot := range scatterPayloads {
					scatterPayloads[slot] = protocolPayloadWithScalar(scatterShape, 2000+uint64(slot))
				}
			}
			scattered, err := channel.RootScatter(
				ProtocolMPIPhaseW3, scatterShape, scatterPayloads,
			)
			if err != nil {
				results[rank].err = fmt.Errorf("W3 scatter: %w", err)
				return
			}
			results[rank].w3Scatter = scattered

			w3BroadcastShape, _ := protocolMPIExpectedShape(
				ProtocolMPIPhaseW3, protocolMPIOpBroadcast, partitions,
			)
			w3Public := MPIPayload{}
			if rank == 0 {
				w3Public = protocolPayloadWithScalar(w3BroadcastShape, 3000)
			}
			broadcast, err := channel.RootBroadcast(
				ProtocolMPIPhaseW3, w3BroadcastShape, w3Public,
			)
			if err != nil {
				results[rank].err = fmt.Errorf("W3 broadcast: %w", err)
				return
			}
			results[rank].broadcasts[ProtocolMPIPhaseW3] = broadcast

			if err := runProtocolAggregateBroadcast(
				channel, ProtocolMPIPhaseU0, rank, &results[rank],
			); err != nil {
				results[rank].err = err
				return
			}

			u1GatherShape, _ := protocolMPIExpectedShape(
				ProtocolMPIPhaseU1, protocolMPIOpGather, partitions,
			)
			u1Local := protocolPayloadWithScalar(u1GatherShape, 4000+rank)
			u1Gather, err := channel.GatherWorkers(ProtocolMPIPhaseU1, u1GatherShape, u1Local)
			if err != nil {
				results[rank].err = fmt.Errorf("U1 gather: %w", err)
				return
			}
			results[rank].u1Gather = u1Gather

			u1BroadcastShape, _ := protocolMPIExpectedShape(
				ProtocolMPIPhaseU1, protocolMPIOpBroadcast, partitions,
			)
			u1Public := MPIPayload{}
			if rank == 0 {
				u1Public = protocolPayloadWithScalar(u1BroadcastShape, 5000)
			}
			broadcast, err = channel.RootBroadcast(
				ProtocolMPIPhaseU1, u1BroadcastShape, u1Public,
			)
			if err != nil {
				results[rank].err = fmt.Errorf("U1 broadcast: %w", err)
				return
			}
			results[rank].broadcasts[ProtocolMPIPhaseU1] = broadcast

			for _, phase := range []ProtocolMPIPhase{ProtocolMPIPhaseU2, ProtocolMPIPhaseU3} {
				if err := runProtocolAggregateBroadcast(
					channel, phase, rank, &results[rank],
				); err != nil {
					results[rank].err = err
					return
				}
			}
			results[rank].accounting = channel.Accounting()
		}()
	}
	wait.Wait()

	for rank := uint64(0); rank < worldSize; rank++ {
		if results[rank].err != nil {
			t.Fatalf("rank %d: %v", rank, results[rank].err)
		}
		if !results[rank].isParty || results[rank].partySlot != rank ||
			results[rank].coordinator != (rank == 0) {
			t.Fatalf("rank %d slot mismatch", rank)
		}

		for _, phase := range []ProtocolMPIPhase{
			ProtocolMPIPhaseW0, ProtocolMPIPhaseW1, ProtocolMPIPhaseW2,
			ProtocolMPIPhaseU0, ProtocolMPIPhaseU2, ProtocolMPIPhaseU3,
		} {
			wantAggregate := protocolAggregateScalar(phase, partitions)
			if rank == 0 {
				assertProtocolPayloadScalar(t, results[rank].aggregates[phase], wantAggregate)
			} else if !emptyMPIPayload(results[rank].aggregates[phase]) {
				t.Fatalf("rank %d phase %s received aggregate", rank, phase)
			}
			assertProtocolPayloadScalar(t, results[rank].broadcasts[phase], protocolBroadcastScalar(phase))
		}
		assertProtocolPayloadScalar(t, results[rank].broadcasts[ProtocolMPIPhaseW3], 3000)
		assertProtocolPayloadScalar(t, results[rank].broadcasts[ProtocolMPIPhaseU1], 5000)

		if rank == 0 {
			if len(results[rank].w3Gather) != int(partitions) ||
				len(results[rank].u1Gather) != int(partitions) {
				t.Fatalf("root gather lengths W3=%d U1=%d", len(results[rank].w3Gather), len(results[rank].u1Gather))
			}
			for slot := uint64(0); slot < partitions; slot++ {
				assertProtocolPayloadScalar(t, results[rank].w3Gather[slot], 1000+slot)
				assertProtocolPayloadScalar(t, results[rank].u1Gather[slot], 4000+slot)
			}
		} else {
			if results[rank].w3Gather != nil || results[rank].u1Gather != nil {
				t.Fatalf("party rank %d received gathered records", rank)
			}
		}
		assertProtocolPayloadScalar(t, results[rank].w3Scatter, 2000+rank)

		wantAccounting := expectedProtocolMPIAccounting(rank, partitions)
		if !reflect.DeepEqual(results[rank].accounting, wantAccounting) {
			t.Fatalf(
				"rank %d accounting\n got: %+v\nwant: %+v",
				rank, results[rank].accounting, wantAccounting,
			)
		}
	}

	// Completion is terminal; replaying the last operation is rejected before
	// any transport action.
	if _, err := channels[0].RootBroadcast(
		ProtocolMPIPhaseU3, MPIPayloadShape{G1: 3}, protocolPayloadWithScalar(MPIPayloadShape{G1: 3}, 1),
	); !errors.Is(err, ErrProtocolMPISchedule) {
		t.Fatalf("post-completion replay: got %v", err)
	}
}

func runProtocolAggregateBroadcast(
	channel *ProtocolMPIChannel,
	phase ProtocolMPIPhase,
	rank uint64,
	result *protocolFullResult,
) error {
	aggregateShape, err := protocolMPIExpectedShape(phase, protocolMPIOpAggregate, channel.partitions)
	if err != nil {
		return err
	}
	local := protocolPayloadWithScalar(aggregateShape, protocolWorkerScalar(phase, rank))
	aggregate, err := channel.AggregateWorkers(phase, aggregateShape, local)
	if err != nil {
		return fmt.Errorf("%s aggregate: %w", phase, err)
	}
	result.aggregates[phase] = aggregate

	broadcastShape, err := protocolMPIExpectedShape(phase, protocolMPIOpBroadcast, channel.partitions)
	if err != nil {
		return err
	}
	public := MPIPayload{}
	if rank == 0 {
		public = protocolPayloadWithScalar(broadcastShape, protocolBroadcastScalar(phase))
	}
	broadcast, err := channel.RootBroadcast(phase, broadcastShape, public)
	if err != nil {
		return fmt.Errorf("%s broadcast: %w", phase, err)
	}
	result.broadcasts[phase] = broadcast
	return nil
}

func TestProtocolMPIChannelStrictScheduleAndRoles(t *testing.T) {
	root, err := newProtocolMPIChannel(newFakeMPINetwork(2).transport(0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := root.RootBroadcast(ProtocolMPIPhaseW0, MPIPayloadShape{G1: 3}, MPIPayload{}); !errors.Is(err, ErrProtocolMPISchedule) {
		t.Fatalf("broadcast before aggregate: got %v", err)
	}
	if _, err := root.AggregateWorkers(ProtocolMPIPhaseW1, MPIPayloadShape{G1: 1}, MPIPayload{}); !errors.Is(err, ErrProtocolMPISchedule) {
		t.Fatalf("phase jump: got %v", err)
	}
	if _, err := root.AggregateWorkers(ProtocolMPIPhaseW0, MPIPayloadShape{G1: 2}, MPIPayload{}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("wrong phase shape: got %v", err)
	}
	if _, err := root.AggregateWorkers(
		ProtocolMPIPhaseW0,
		MPIPayloadShape{G1: 3},
		MPIPayload{},
	); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("missing root worker share: got %v", err)
	}

	root.phase = ProtocolMPIPhaseW3
	root.phaseStep = 1
	root.sequence = 7
	if _, err := root.RootScatter(
		ProtocolMPIPhaseW3, MPIPayloadShape{Fields: 2}, []MPIPayload{{Fields: make([]fr.Element, 2)}},
	); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("short root scatter: got %v", err)
	}

	party, err := newProtocolMPIChannel(newFakeMPINetwork(2).transport(1))
	if err != nil {
		t.Fatal(err)
	}
	party.phase = ProtocolMPIPhaseW3
	party.phaseStep = 1
	party.sequence = 7
	if _, err := party.RootScatter(
		ProtocolMPIPhaseW3,
		MPIPayloadShape{Fields: 2},
		[]MPIPayload{protocolPayloadWithScalar(MPIPayloadShape{Fields: 2}, 1)},
	); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("party supplied scatter vector: got %v", err)
	}

	party.phase = ProtocolMPIPhaseW0
	party.phaseStep = 1
	party.sequence = 1
	if _, err := party.RootBroadcast(
		ProtocolMPIPhaseW0,
		MPIPayloadShape{G1: 3},
		protocolPayloadWithScalar(MPIPayloadShape{G1: 3}, 1),
	); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("party supplied root broadcast: got %v", err)
	}
}

func TestProtocolMPIChannelRejectsCorruptHeaders(t *testing.T) {
	shape := MPIPayloadShape{G1: 3}
	wantHeader := protocolMPIRecordHeader{
		Operation: protocolMPIOpAggregate,
		Phase:     ProtocolMPIPhaseW0,
		Sequence:  0,
		Shape:     shape,
	}
	canonical, err := encodeProtocolMPIHeader(wantHeader)
	if err != nil {
		t.Fatal(err)
	}

	mutations := map[string]func([]byte){
		"magic":     func(encoded []byte) { encoded[0] ^= 0xff },
		"reserved":  func(encoded []byte) { encoded[7] = 1 },
		"operation": func(encoded []byte) { encoded[5] = byte(protocolMPIOpGather) },
		"phase":     func(encoded []byte) { encoded[6] = byte(ProtocolMPIPhaseW1) },
		"sequence":  func(encoded []byte) { binary.BigEndian.PutUint32(encoded[8:12], 1) },
		"shape":     func(encoded []byte) { binary.BigEndian.PutUint32(encoded[16:20], 2) },
	}
	for name, mutate := range mutations {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			encoded := append([]byte(nil), canonical...)
			mutate(encoded)
			transport := &protocolScriptTransport{
				rank: 0,
				size: 2,
				receives: []protocolScriptReceive{{
					from:  1,
					bytes: encoded,
				}},
			}
			channel, channelErr := newProtocolMPIChannel(transport)
			if channelErr != nil {
				t.Fatal(channelErr)
			}
			if _, aggregateErr := channel.AggregateWorkers(
				ProtocolMPIPhaseW0, shape, protocolPayloadWithScalar(shape, 1),
			); !errors.Is(aggregateErr, ErrProtocolMPIRecord) {
				t.Fatalf("got %v, want protocol record error", aggregateErr)
			}
		})
	}
}

func TestProtocolMPIHeaderCanonicalRoundTrip(t *testing.T) {
	header := protocolMPIRecordHeader{
		Operation: protocolMPIOpScatter,
		Phase:     ProtocolMPIPhaseW3,
		Sequence:  7,
		Shape:     MPIPayloadShape{Fields: 2},
	}
	encoded, err := encodeProtocolMPIHeader(header)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeProtocolMPIHeader(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != header {
		t.Fatalf("decoded header %+v, want %+v", decoded, header)
	}
	if _, err := decodeProtocolMPIHeader(encoded[:len(encoded)-1]); !errors.Is(err, ErrProtocolMPIRecord) {
		t.Fatalf("short header: got %v", err)
	}
}

func protocolPayloadWithScalar(shape MPIPayloadShape, scalar uint64) MPIPayload {
	payload := MPIPayload{
		Fields: make([]fr.Element, shape.Fields),
		G1:     make([]bn254.G1Affine, shape.G1),
	}
	for index := range payload.Fields {
		payload.Fields[index] = fr.NewElement(scalar)
	}
	for index := range payload.G1 {
		payload.G1[index] = mpiTestG1(int64(scalar))
	}
	return payload
}

func assertProtocolPayloadScalar(t *testing.T, payload MPIPayload, scalar uint64) {
	t.Helper()
	wantField := fr.NewElement(scalar)
	wantPoint := mpiTestG1(int64(scalar))
	for index := range payload.Fields {
		if !payload.Fields[index].Equal(&wantField) {
			t.Fatalf("field %d differs from scalar %d", index, scalar)
		}
	}
	for index := range payload.G1 {
		if !payload.G1[index].Equal(&wantPoint) {
			t.Fatalf("G1 %d differs from scalar %d", index, scalar)
		}
	}
}

func protocolWorkerScalar(phase ProtocolMPIPhase, rank uint64) uint64 {
	return uint64(phase+1)*10 + rank
}

func protocolAggregateScalar(phase ProtocolMPIPhase, partitions uint64) uint64 {
	return partitions*uint64(phase+1)*10 + partitions*(partitions-1)/2
}

func protocolBroadcastScalar(phase ProtocolMPIPhase) uint64 {
	return 6000 + uint64(phase)
}

func expectedProtocolMPIAccounting(rank, partitions uint64) ProtocolMPIAccounting {
	result := ProtocolMPIAccounting{
		Rank:           rank,
		WorldSize:      partitions,
		PartitionCount: partitions,
	}
	for phase := ProtocolMPIPhaseW0; phase <= ProtocolMPIPhaseU3; phase++ {
		for _, operation := range protocolMPISchedule[int(phase)] {
			shape, err := protocolMPIExpectedShape(phase, operation, partitions)
			if err != nil {
				panic(err)
			}
			payloadBytes, err := mpiPayloadByteSize(shape)
			if err != nil {
				panic(err)
			}
			phaseAccounting := &result.Phases[int(phase)]
			totalAccounting := &result.Total
			for _, accounting := range []*ProtocolMPIPhaseAccounting{phaseAccounting, totalAccounting} {
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
				multiplier := uint64(1)
				if rank == 0 {
					multiplier = partitions - 1
				}
				wireBytes := multiplier * (uint64(protocolMPIHeaderSize) + payloadBytes)
				protocolBytes := multiplier * payloadBytes
				if (rank == 0 && (operation == protocolMPIOpAggregate || operation == protocolMPIOpGather)) ||
					(rank != 0 && (operation == protocolMPIOpScatter || operation == protocolMPIOpBroadcast)) {
					accounting.WireBytesRecv += wireBytes
					accounting.PayloadBytesRecv += protocolBytes
				} else {
					accounting.WireBytesSent += wireBytes
					accounting.PayloadBytesSent += protocolBytes
				}
			}
			result.Operations++
		}
	}
	return result
}

type protocolStaticTransport struct {
	rank uint64
	size uint64
}

func (transport *protocolStaticTransport) Rank() uint64 { return transport.rank }
func (transport *protocolStaticTransport) Size() uint64 { return transport.size }
func (*protocolStaticTransport) SendBytes([]byte, uint64) error {
	return errors.New("unexpected send")
}
func (*protocolStaticTransport) ReceiveBytes(uint64, uint64) ([]byte, error) {
	return nil, errors.New("unexpected receive")
}

type protocolScriptReceive struct {
	from  uint64
	bytes []byte
}

type protocolScriptTransport struct {
	rank     uint64
	size     uint64
	receives []protocolScriptReceive
	next     int
}

func (transport *protocolScriptTransport) Rank() uint64 { return transport.rank }
func (transport *protocolScriptTransport) Size() uint64 { return transport.size }
func (*protocolScriptTransport) SendBytes([]byte, uint64) error {
	return errors.New("unexpected send")
}
func (transport *protocolScriptTransport) ReceiveBytes(size uint64, rank uint64) ([]byte, error) {
	if transport.next >= len(transport.receives) {
		return nil, errors.New("unexpected receive")
	}
	receive := transport.receives[transport.next]
	transport.next++
	if receive.from != rank {
		return nil, fmt.Errorf("receive from rank %d, want %d", rank, receive.from)
	}
	if uint64(len(receive.bytes)) != size {
		return nil, fmt.Errorf("receive size %d, want %d", size, len(receive.bytes))
	}
	return append([]byte(nil), receive.bytes...), nil
}
