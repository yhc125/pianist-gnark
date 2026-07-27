package dlinkzg

import (
	"errors"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestProtocolPayloadFixedRecordsRoundTripInSemanticOrder(t *testing.T) {
	t.Run("W0", func(t *testing.T) {
		message := OuterW0Message{WitnessCommitments: protocolPayloadTestPoints3(10)}
		payload, err := PackProtocolW0(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestShape(t, payload, MPIPayloadShape{G1: 3})
		protocolPayloadTestPointOrder(t, payload.G1, message.WitnessCommitments[:])
		decoded, err := UnpackProtocolW0(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, decoded.WitnessCommitments[:], message.WitnessCommitments[:])
	})

	t.Run("W1", func(t *testing.T) {
		message := OuterW1Message{AccumulatorCommitment: protocolPayloadTestPoint(20)}
		payload, err := PackProtocolW1(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestShape(t, payload, MPIPayloadShape{G1: 1})
		protocolPayloadTestPointOrder(t, payload.G1, []bn254.G1Affine{message.AccumulatorCommitment})
		decoded, err := UnpackProtocolW1(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, []bn254.G1Affine{decoded.AccumulatorCommitment}, payload.G1)
	})

	t.Run("W2", func(t *testing.T) {
		message := OuterW2Message{QuotientCommitments: protocolPayloadTestPoints3(30)}
		payload, err := PackProtocolW2(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestShape(t, payload, MPIPayloadShape{G1: 3})
		protocolPayloadTestPointOrder(t, payload.G1, message.QuotientCommitments[:])
		decoded, err := UnpackProtocolW2(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, decoded.QuotientCommitments[:], message.QuotientCommitments[:])
	})

	t.Run("W3 gather 13/1/7", func(t *testing.T) {
		var terminals LocalTerminalEvaluations
		protocolPayloadTestFillFields(terminals.Alpha[:], 100)
		protocolPayloadTestFillFields(terminals.OmegaAlpha[:], 200)
		protocolPayloadTestFillFields(terminals.XStar[:], 300)
		record := ProtocolW3GatherRecord{Terminals: terminals}
		payload, err := PackProtocolW3Gather(record)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestShape(t, payload, MPIPayloadShape{Fields: 21})
		protocolPayloadTestFieldOrder(t, payload.Fields, terminals.Flatten())
		decoded, err := UnpackProtocolW3Gather(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, decoded.Terminals.Flatten(), terminals.Flatten())
	})

	t.Run("W3 scatter", func(t *testing.T) {
		record := ProtocolW3ScatterRecord{T0: fr.NewElement(401), T1: fr.NewElement(402)}
		payload, err := PackProtocolW3Scatter(record)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, payload.Fields, []fr.Element{record.T0, record.T1})
		decoded, err := UnpackProtocolW3Scatter(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, []fr.Element{decoded.T0, decoded.T1}, payload.Fields)
	})

	t.Run("U0", func(t *testing.T) {
		message := U0Message{PartialCommitments: protocolPayloadTestPoints3(50)}
		payload, err := PackProtocolU0(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, payload.G1, message.PartialCommitments[:])
		decoded, err := UnpackProtocolU0(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, decoded.PartialCommitments[:], payload.G1)
	})

	t.Run("U1 gather", func(t *testing.T) {
		record := ProtocolU1GatherRecord{
			D:                 [3]fr.Element{fr.NewElement(501), fr.NewElement(502), fr.NewElement(503)},
			LaurentCommitment: protocolPayloadTestPoint(60),
		}
		payload, err := PackProtocolU1Gather(record)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestShape(t, payload, MPIPayloadShape{Fields: 3, G1: 1})
		protocolPayloadTestFieldOrder(t, payload.Fields, record.D[:])
		protocolPayloadTestPointOrder(t, payload.G1, []bn254.G1Affine{record.LaurentCommitment})
		decoded, err := UnpackProtocolU1Gather(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, decoded.D[:], record.D[:])
		protocolPayloadTestPointOrder(t, []bn254.G1Affine{decoded.LaurentCommitment}, payload.G1)
	})

	t.Run("U1 broadcast", func(t *testing.T) {
		message := U1Message{
			LinkEvaluations:   [3]fr.Element{fr.NewElement(601), fr.NewElement(602), fr.NewElement(603)},
			LinkCommitment:    protocolPayloadTestPoint(70),
			LaurentCommitment: protocolPayloadTestPoint(71),
		}
		payload, err := PackProtocolU1Broadcast(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, payload.Fields, message.LinkEvaluations[:])
		protocolPayloadTestPointOrder(t, payload.G1, []bn254.G1Affine{message.LinkCommitment, message.LaurentCommitment})
		decoded, err := UnpackProtocolU1Broadcast(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, decoded.LinkEvaluations[:], message.LinkEvaluations[:])
		protocolPayloadTestPointOrder(t, []bn254.G1Affine{decoded.LinkCommitment, decoded.LaurentCommitment}, payload.G1)
	})

	t.Run("U2 aggregate interleaving", func(t *testing.T) {
		record := ProtocolU2AggregateRecord{
			PartialAtBeta:        [3]fr.Element{fr.NewElement(701), fr.NewElement(702), fr.NewElement(703)},
			PartialAtBetaInverse: [3]fr.Element{fr.NewElement(711), fr.NewElement(712), fr.NewElement(713)},
			SAtBeta:              fr.NewElement(720),
		}
		want := []fr.Element{
			record.PartialAtBeta[0], record.PartialAtBetaInverse[0],
			record.PartialAtBeta[1], record.PartialAtBetaInverse[1],
			record.PartialAtBeta[2], record.PartialAtBetaInverse[2],
			record.SAtBeta,
		}
		payload, err := PackProtocolU2Aggregate(record)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, payload.Fields, want)
		decoded, err := UnpackProtocolU2Aggregate(payload)
		protocolPayloadTestNoError(t, err)
		repacked, err := PackProtocolU2Aggregate(decoded)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, repacked.Fields, want)
	})

	t.Run("U2 broadcast proof-codec interleaving", func(t *testing.T) {
		message := U2Message{
			PartialAtBeta:        [3]fr.Element{fr.NewElement(801), fr.NewElement(802), fr.NewElement(803)},
			PartialAtBetaInverse: [3]fr.Element{fr.NewElement(811), fr.NewElement(812), fr.NewElement(813)},
			BatchAtBeta:          [4]fr.Element{fr.NewElement(821), fr.NewElement(822), fr.NewElement(823), fr.NewElement(824)},
			BatchAtBetaInverse:   [4]fr.Element{fr.NewElement(831), fr.NewElement(832), fr.NewElement(833), fr.NewElement(834)},
		}
		want := make([]fr.Element, 0, 14)
		for i := range message.PartialAtBeta {
			want = append(want, message.PartialAtBeta[i], message.PartialAtBetaInverse[i])
		}
		for i := range message.BatchAtBeta {
			want = append(want, message.BatchAtBeta[i], message.BatchAtBetaInverse[i])
		}
		payload, err := PackProtocolU2Broadcast(message)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, payload.Fields, want)
		decoded, err := UnpackProtocolU2Broadcast(payload)
		protocolPayloadTestNoError(t, err)
		repacked, err := PackProtocolU2Broadcast(decoded)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestFieldOrder(t, repacked.Fields, want)
	})

	t.Run("U3 aggregate and broadcast", func(t *testing.T) {
		aggregate := ProtocolU3AggregateRecord{
			WN: protocolPayloadTestPoint(90), PiZ: protocolPayloadTestPoint(92),
		}
		payload, err := PackProtocolU3Aggregate(aggregate)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, payload.G1, []bn254.G1Affine{aggregate.WN, aggregate.PiZ})
		decodedAggregate, err := UnpackProtocolU3Aggregate(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t,
			[]bn254.G1Affine{decodedAggregate.WN, decodedAggregate.PiZ}, payload.G1,
		)

		broadcast := U3Message{WN: aggregate.WN, PiZ: aggregate.PiZ, PiY: protocolPayloadTestPoint(93)}
		payload, err = PackProtocolU3Broadcast(broadcast)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t, payload.G1,
			[]bn254.G1Affine{broadcast.WN, broadcast.PiZ, broadcast.PiY},
		)
		decodedBroadcast, err := UnpackProtocolU3Broadcast(payload)
		protocolPayloadTestNoError(t, err)
		protocolPayloadTestPointOrder(t,
			[]bn254.G1Affine{decodedBroadcast.WN, decodedBroadcast.PiZ, decodedBroadcast.PiY},
			payload.G1,
		)
	})
}

func TestProtocolPayloadPackShapesMatchAllSeventeenOperations(t *testing.T) {
	const partitions = uint64(4)
	w0, err := PackProtocolW0(OuterW0Message{WitnessCommitments: protocolPayloadTestPoints3(1)})
	protocolPayloadTestNoError(t, err)
	w1, err := PackProtocolW1(OuterW1Message{AccumulatorCommitment: protocolPayloadTestPoint(4)})
	protocolPayloadTestNoError(t, err)
	w2, err := PackProtocolW2(OuterW2Message{QuotientCommitments: protocolPayloadTestPoints3(5)})
	protocolPayloadTestNoError(t, err)
	w3Gather, err := PackProtocolW3Gather(ProtocolW3GatherRecord{})
	protocolPayloadTestNoError(t, err)
	w3Scatter, err := PackProtocolW3Scatter(ProtocolW3ScatterRecord{})
	protocolPayloadTestNoError(t, err)
	w3Broadcast, err := PackProtocolW3Broadcast(ProtocolW3BroadcastRecord{
		ProductCheck: OuterProductCheckCommitmentsMessage{
			Commitments: [2]bn254.G1Affine{protocolPayloadTestPoint(8), protocolPayloadTestPoint(9)},
		},
		SumCheckRounds: make([]OuterSumCheckRoundMessage, 2),
	}, partitions)
	protocolPayloadTestNoError(t, err)
	u0, err := PackProtocolU0(U0Message{PartialCommitments: protocolPayloadTestPoints3(10)})
	protocolPayloadTestNoError(t, err)
	u1Gather, err := PackProtocolU1Gather(ProtocolU1GatherRecord{LaurentCommitment: protocolPayloadTestPoint(13)})
	protocolPayloadTestNoError(t, err)
	u1Broadcast, err := PackProtocolU1Broadcast(U1Message{
		LinkCommitment: protocolPayloadTestPoint(14), LaurentCommitment: protocolPayloadTestPoint(15),
	})
	protocolPayloadTestNoError(t, err)
	u2Aggregate, err := PackProtocolU2Aggregate(ProtocolU2AggregateRecord{})
	protocolPayloadTestNoError(t, err)
	u2Broadcast, err := PackProtocolU2Broadcast(U2Message{})
	protocolPayloadTestNoError(t, err)
	u3Aggregate, err := PackProtocolU3Aggregate(ProtocolU3AggregateRecord{
		WN: protocolPayloadTestPoint(16), PiZ: protocolPayloadTestPoint(18),
	})
	protocolPayloadTestNoError(t, err)
	u3Broadcast, err := PackProtocolU3Broadcast(U3Message{
		WN: protocolPayloadTestPoint(19), PiZ: protocolPayloadTestPoint(21), PiY: protocolPayloadTestPoint(22),
	})
	protocolPayloadTestNoError(t, err)

	tests := []struct {
		phase     ProtocolMPIPhase
		operation protocolMPIOperation
		payload   MPIPayload
	}{
		{ProtocolMPIPhaseW0, protocolMPIOpAggregate, w0},
		{ProtocolMPIPhaseW0, protocolMPIOpBroadcast, w0},
		{ProtocolMPIPhaseW1, protocolMPIOpAggregate, w1},
		{ProtocolMPIPhaseW1, protocolMPIOpBroadcast, w1},
		{ProtocolMPIPhaseW2, protocolMPIOpAggregate, w2},
		{ProtocolMPIPhaseW2, protocolMPIOpBroadcast, w2},
		{ProtocolMPIPhaseW3, protocolMPIOpGather, w3Gather},
		{ProtocolMPIPhaseW3, protocolMPIOpScatter, w3Scatter},
		{ProtocolMPIPhaseW3, protocolMPIOpBroadcast, w3Broadcast},
		{ProtocolMPIPhaseU0, protocolMPIOpAggregate, u0},
		{ProtocolMPIPhaseU0, protocolMPIOpBroadcast, u0},
		{ProtocolMPIPhaseU1, protocolMPIOpGather, u1Gather},
		{ProtocolMPIPhaseU1, protocolMPIOpBroadcast, u1Broadcast},
		{ProtocolMPIPhaseU2, protocolMPIOpAggregate, u2Aggregate},
		{ProtocolMPIPhaseU2, protocolMPIOpBroadcast, u2Broadcast},
		{ProtocolMPIPhaseU3, protocolMPIOpAggregate, u3Aggregate},
		{ProtocolMPIPhaseU3, protocolMPIOpBroadcast, u3Broadcast},
	}
	if len(tests) != 17 {
		t.Fatalf("tested operation count %d, want 17", len(tests))
	}
	for _, test := range tests {
		want, err := protocolMPIExpectedShape(test.phase, test.operation, partitions)
		protocolPayloadTestNoError(t, err)
		if got := test.payload.Shape(); got != want {
			t.Fatalf("%s %s shape %+v, want %+v", test.phase, test.operation, got, want)
		}
	}
}

func TestProtocolW3BroadcastPayloadM2M4ShapeOrderAndIsolation(t *testing.T) {
	for _, partitions := range []uint64{2, 4} {
		partitions := partitions
		t.Run("M="+new(big.Int).SetUint64(partitions).String(), func(t *testing.T) {
			rounds, err := protocolPayloadRoundCount(partitions)
			protocolPayloadTestNoError(t, err)
			record := ProtocolW3BroadcastRecord{
				ProductCheck: OuterProductCheckCommitmentsMessage{
					Commitments: [2]bn254.G1Affine{protocolPayloadTestPoint(101), protocolPayloadTestPoint(102)},
				},
				SumCheckRounds: make([]OuterSumCheckRoundMessage, rounds),
			}
			var wantFields []fr.Element
			for round := range record.SumCheckRounds {
				for coefficient := range record.SumCheckRounds[round].Coefficients {
					value := fr.NewElement(uint64(1000 + 100*round + coefficient))
					record.SumCheckRounds[round].Coefficients[coefficient] = value
					wantFields = append(wantFields, value)
				}
			}
			protocolPayloadTestFillFields(record.FinalEvaluations.Terminal[:], 2000)
			protocolPayloadTestFillFields(record.FinalEvaluations.ProductCheck[:], 3000)
			wantFields = append(wantFields, record.FinalEvaluations.Terminal[:]...)
			wantFields = append(wantFields, record.FinalEvaluations.ProductCheck[:]...)

			payload, err := PackProtocolW3Broadcast(record, partitions)
			protocolPayloadTestNoError(t, err)
			protocolPayloadTestShape(t, payload, MPIPayloadShape{Fields: 6*rounds + 26, G1: 2})
			protocolPayloadTestFieldOrder(t, payload.Fields, wantFields)
			protocolPayloadTestPointOrder(t, payload.G1, record.ProductCheck.Commitments[:])

			// Pack must not retain the dynamic SumCheck backing array.
			packedFirst := payload.Fields[0]
			record.SumCheckRounds[0].Coefficients[0].Add(&record.SumCheckRounds[0].Coefficients[0], protocolPayloadTestFieldPointer(1))
			if !payload.Fields[0].Equal(&packedFirst) {
				t.Fatal("packed W3 aliases source SumCheck rounds")
			}

			decoded, err := UnpackProtocolW3Broadcast(payload, partitions)
			protocolPayloadTestNoError(t, err)
			protocolPayloadTestFieldOrder(t, decoded.SumCheckRounds[0].Coefficients[:], wantFields[:6])
			protocolPayloadTestPointOrder(t, decoded.ProductCheck.Commitments[:], payload.G1)

			// Unpack must not retain either MPIPayload backing slice.
			decodedFirst := decoded.SumCheckRounds[0].Coefficients[0]
			payload.Fields[0].Add(&payload.Fields[0], protocolPayloadTestFieldPointer(9))
			payload.G1[0] = protocolPayloadTestPoint(999)
			if !decoded.SumCheckRounds[0].Coefficients[0].Equal(&decodedFirst) {
				t.Fatal("decoded W3 aliases source field payload")
			}
			wantFirstCommitment := protocolPayloadTestPoint(101)
			if !decoded.ProductCheck.Commitments[0].Equal(&wantFirstCommitment) {
				t.Fatal("decoded W3 aliases source G1 payload")
			}
		})
	}
}

func TestProtocolPayloadRejectsWrongShapesAndInvalidValues(t *testing.T) {
	validPoint := protocolPayloadTestPoint(1)
	validField := fr.NewElement(1)
	tests := []struct {
		name   string
		unpack func(MPIPayload) error
		valid  MPIPayload
	}{
		{"W0", func(p MPIPayload) error { _, err := UnpackProtocolW0(p); return err }, MPIPayload{G1: make([]bn254.G1Affine, 3)}},
		{"W1", func(p MPIPayload) error { _, err := UnpackProtocolW1(p); return err }, MPIPayload{G1: []bn254.G1Affine{validPoint}}},
		{"W2", func(p MPIPayload) error { _, err := UnpackProtocolW2(p); return err }, MPIPayload{G1: make([]bn254.G1Affine, 3)}},
		{"W3 gather", func(p MPIPayload) error { _, err := UnpackProtocolW3Gather(p); return err }, MPIPayload{Fields: make([]fr.Element, 21)}},
		{"W3 scatter", func(p MPIPayload) error { _, err := UnpackProtocolW3Scatter(p); return err }, MPIPayload{Fields: make([]fr.Element, 2)}},
		{"W3 broadcast", func(p MPIPayload) error { _, err := UnpackProtocolW3Broadcast(p, 2); return err }, MPIPayload{Fields: make([]fr.Element, 32), G1: make([]bn254.G1Affine, 2)}},
		{"U0", func(p MPIPayload) error { _, err := UnpackProtocolU0(p); return err }, MPIPayload{G1: make([]bn254.G1Affine, 3)}},
		{"U1 gather", func(p MPIPayload) error { _, err := UnpackProtocolU1Gather(p); return err }, MPIPayload{Fields: make([]fr.Element, 3), G1: []bn254.G1Affine{validPoint}}},
		{"U1 broadcast", func(p MPIPayload) error { _, err := UnpackProtocolU1Broadcast(p); return err }, MPIPayload{Fields: make([]fr.Element, 3), G1: make([]bn254.G1Affine, 2)}},
		{"U2 aggregate", func(p MPIPayload) error { _, err := UnpackProtocolU2Aggregate(p); return err }, MPIPayload{Fields: make([]fr.Element, 7)}},
		{"U2 broadcast", func(p MPIPayload) error { _, err := UnpackProtocolU2Broadcast(p); return err }, MPIPayload{Fields: make([]fr.Element, 14)}},
		{"U3 aggregate", func(p MPIPayload) error { _, err := UnpackProtocolU3Aggregate(p); return err }, MPIPayload{G1: make([]bn254.G1Affine, 2)}},
		{"U3 broadcast", func(p MPIPayload) error { _, err := UnpackProtocolU3Broadcast(p); return err }, MPIPayload{G1: make([]bn254.G1Affine, 3)}},
	}
	for _, test := range tests {
		t.Run(test.name+" fields", func(t *testing.T) {
			bad := test.valid.clone()
			bad.Fields = append(bad.Fields, validField)
			if err := test.unpack(bad); !errors.Is(err, ErrMPIPayload) {
				t.Fatalf("wrong field shape: got %v", err)
			}
		})
		t.Run(test.name+" G1", func(t *testing.T) {
			bad := test.valid.clone()
			bad.G1 = append(bad.G1, validPoint)
			if err := test.unpack(bad); !errors.Is(err, ErrMPIPayload) {
				t.Fatalf("wrong G1 shape: got %v", err)
			}
		})
	}

	nonCanonical := fr.Element{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}
	if _, err := PackProtocolW3Scatter(ProtocolW3ScatterRecord{T0: nonCanonical}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("pack non-canonical field: got %v", err)
	}
	if _, err := UnpackProtocolW3Scatter(MPIPayload{Fields: []fr.Element{validField, nonCanonical}}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("unpack non-canonical field: got %v", err)
	}

	invalidPoint := validPoint
	invalidPoint.X.SetUint64(1)
	invalidPoint.Y.SetUint64(1)
	if _, err := PackProtocolW1(OuterW1Message{AccumulatorCommitment: invalidPoint}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("pack invalid G1: got %v", err)
	}
	if _, err := UnpackProtocolW1(MPIPayload{G1: []bn254.G1Affine{invalidPoint}}); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("unpack invalid G1: got %v", err)
	}
}

func TestProtocolW3BroadcastPayloadRejectsPartitionAndRoundTampering(t *testing.T) {
	for _, partitions := range []uint64{0, 1, 3, 6} {
		if _, err := PackProtocolW3Broadcast(ProtocolW3BroadcastRecord{}, partitions); !errors.Is(err, ErrMPIPayload) {
			t.Fatalf("pack M=%d: got %v", partitions, err)
		}
		if _, err := UnpackProtocolW3Broadcast(MPIPayload{}, partitions); !errors.Is(err, ErrMPIPayload) {
			t.Fatalf("unpack M=%d: got %v", partitions, err)
		}
	}

	wrongRounds := ProtocolW3BroadcastRecord{
		ProductCheck: OuterProductCheckCommitmentsMessage{
			Commitments: [2]bn254.G1Affine{protocolPayloadTestPoint(1), protocolPayloadTestPoint(2)},
		},
		SumCheckRounds: make([]OuterSumCheckRoundMessage, 1),
	}
	if _, err := PackProtocolW3Broadcast(wrongRounds, 4); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("wrong round count: got %v", err)
	}

	rightRounds := wrongRounds
	rightRounds.SumCheckRounds = make([]OuterSumCheckRoundMessage, 2)
	payload, err := PackProtocolW3Broadcast(rightRounds, 4)
	protocolPayloadTestNoError(t, err)
	if _, err := UnpackProtocolW3Broadcast(payload, 2); !errors.Is(err, ErrMPIPayload) {
		t.Fatalf("partition reinterpretation changed shape without rejection: got %v", err)
	}

	// A scalar mutation remains in the exact corresponding semantic slot; the
	// payload layer must never silently permute a tampered round coefficient.
	payload.Fields[7] = fr.NewElement(0xdead)
	decoded, err := UnpackProtocolW3Broadcast(payload, 4)
	protocolPayloadTestNoError(t, err)
	if !decoded.SumCheckRounds[1].Coefficients[1].Equal(&payload.Fields[7]) {
		t.Fatal("tampered W3 round coefficient was not decoded at its canonical slot")
	}
}

func protocolPayloadTestPoints3(first int64) [3]bn254.G1Affine {
	return [3]bn254.G1Affine{
		protocolPayloadTestPoint(first), protocolPayloadTestPoint(first + 1), protocolPayloadTestPoint(first + 2),
	}
}

func protocolPayloadTestPoint(scalar int64) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var result bn254.G1Affine
	result.ScalarMultiplication(&generator, big.NewInt(scalar))
	return result
}

func protocolPayloadTestFillFields(destination []fr.Element, first uint64) {
	for i := range destination {
		destination[i] = fr.NewElement(first + uint64(i))
	}
}

func protocolPayloadTestFieldPointer(value uint64) *fr.Element {
	result := fr.NewElement(value)
	return &result
}

func protocolPayloadTestNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func protocolPayloadTestShape(t *testing.T, payload MPIPayload, want MPIPayloadShape) {
	t.Helper()
	if got := payload.Shape(); got != want {
		t.Fatalf("payload shape %+v, want %+v", got, want)
	}
}

func protocolPayloadTestFieldOrder(t *testing.T, got, want []fr.Element) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("field count %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !got[i].Equal(&want[i]) {
			t.Fatalf("field %d differs", i)
		}
	}
}

func protocolPayloadTestPointOrder(t *testing.T, got, want []bn254.G1Affine) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("G1 count %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !got[i].Equal(&want[i]) {
			t.Fatalf("G1 element %d differs", i)
		}
	}
}
