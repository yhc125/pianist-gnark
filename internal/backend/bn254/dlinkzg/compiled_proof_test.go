package dlinkzg

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math/big"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestCompiledProofInventoryAndRoundTrip(t *testing.T) {
	for _, partitions := range []uint64{2, 4} {
		t.Run(compiledProofTestPartitionName(partitions), func(t *testing.T) {
			proof := compiledProofTestFixture(partitions)
			encoded, err := proof.MarshalBinary(partitions)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}

			g1Elements, fieldElements, err := CompiledProofElementCounts(partitions)
			if err != nil {
				t.Fatalf("element counts: %v", err)
			}
			wantFields := 6*(bitsForCompiledProofTest(partitions)-1) + 43
			if g1Elements != 18 || fieldElements != wantFields {
				t.Fatalf("inventory = (%d G1, %d Fr), want (18 G1, %d Fr)", g1Elements, fieldElements, wantFields)
			}
			wantBytes := compiledProofHeaderSize + 18*bn254.SizeOfG1AffineCompressed + wantFields*fr.Bytes
			if len(encoded) != wantBytes {
				t.Fatalf("encoded length = %d, want %d", len(encoded), wantBytes)
			}
			reportedBytes, err := CompiledProofEncodedSize(partitions)
			if err != nil || reportedBytes != wantBytes {
				t.Fatalf("reported size = %d, %v; want %d", reportedBytes, err, wantBytes)
			}

			decoded, err := DecodeCompiledProof(encoded, partitions)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !reflect.DeepEqual(*decoded, proof) {
				t.Fatal("decoded proof differs")
			}
			reencoded, err := decoded.MarshalBinary(partitions)
			if err != nil {
				t.Fatalf("re-marshal: %v", err)
			}
			if !bytes.Equal(reencoded, encoded) {
				t.Fatal("round trip is not byte canonical")
			}

			valueDecoded, err := UnmarshalCompiledProof(encoded, partitions)
			if err != nil || !reflect.DeepEqual(valueDecoded, proof) {
				t.Fatalf("value unmarshal differs: %v", err)
			}
			var methodDecoded CompiledProof
			if err := methodDecoded.UnmarshalBinary(encoded, partitions); err != nil {
				t.Fatalf("method unmarshal: %v", err)
			}
			if !reflect.DeepEqual(methodDecoded, proof) {
				t.Fatal("method unmarshal differs")
			}
		})
	}
}

func TestCompiledProofRejectsInvalidPartitionShape(t *testing.T) {
	proof := compiledProofTestFixture(2)
	for _, partitions := range []uint64{0, 1, 3, 6} {
		if _, err := proof.MarshalBinary(partitions); !errors.Is(err, ErrInvalidCompiledProof) {
			t.Fatalf("marshal M=%d: got %v, want ErrInvalidCompiledProof", partitions, err)
		}
		if _, err := DecodeCompiledProof(nil, partitions); !errors.Is(err, ErrInvalidCompiledProof) {
			t.Fatalf("decode M=%d: got %v, want ErrInvalidCompiledProof", partitions, err)
		}
	}

	proof.SumCheckRounds = nil
	if _, err := proof.MarshalBinary(2); !errors.Is(err, ErrInvalidCompiledProof) {
		t.Fatalf("missing round: got %v, want ErrInvalidCompiledProof", err)
	}
	proof.SumCheckRounds = make([]OuterSumCheckRoundMessage, 2)
	if _, err := proof.MarshalBinary(2); !errors.Is(err, ErrInvalidCompiledProof) {
		t.Fatalf("extra round: got %v, want ErrInvalidCompiledProof", err)
	}

	var nilProof *CompiledProof
	if _, err := nilProof.MarshalBinary(2); !errors.Is(err, ErrInvalidCompiledProof) {
		t.Fatalf("nil marshal: got %v, want ErrInvalidCompiledProof", err)
	}
	if err := nilProof.UnmarshalBinary(nil, 2); !errors.Is(err, ErrInvalidCompiledProof) {
		t.Fatalf("nil unmarshal: got %v, want ErrInvalidCompiledProof", err)
	}
}

func TestCompiledProofRejectsMalformedHeaderAndLength(t *testing.T) {
	proof := compiledProofTestFixture(4)
	encoded, err := proof.MarshalBinary(4)
	if err != nil {
		t.Fatal(err)
	}

	for cut := 0; cut < len(encoded); cut++ {
		if _, err := DecodeCompiledProof(encoded[:cut], 4); !errors.Is(err, ErrCompiledProofEncoding) {
			t.Fatalf("truncation at %d: got %v, want ErrCompiledProofEncoding", cut, err)
		}
	}
	trailing := append(append([]byte(nil), encoded...), 0)
	if _, err := DecodeCompiledProof(trailing, 4); !errors.Is(err, ErrCompiledProofEncoding) {
		t.Fatalf("trailing byte: got %v, want ErrCompiledProofEncoding", err)
	}

	cases := []struct {
		name   string
		mutate func([]byte)
	}{
		{"magic", func(value []byte) { value[0] ^= 1 }},
		{"version", func(value []byte) { binary.BigEndian.PutUint16(value[8:10], compiledProofVersion+1) }},
		{"reserved", func(value []byte) { value[11] = 1 }},
		{"partitions", func(value []byte) { binary.BigEndian.PutUint64(value[12:20], 2) }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			malformed := append([]byte(nil), encoded...)
			testCase.mutate(malformed)
			if _, err := DecodeCompiledProof(malformed, 4); !errors.Is(err, ErrCompiledProofEncoding) {
				t.Fatalf("got %v, want ErrCompiledProofEncoding", err)
			}
		})
	}
}

func TestCompiledProofRejectsNonCanonicalElements(t *testing.T) {
	proof := compiledProofTestFixture(2)
	encoded, err := proof.MarshalBinary(2)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("field modulus", func(t *testing.T) {
		malformed := append([]byte(nil), encoded...)
		firstField := compiledProofHeaderSize + 9*bn254.SizeOfG1AffineCompressed
		fr.Modulus().FillBytes(malformed[firstField : firstField+fr.Bytes])
		if _, err := DecodeCompiledProof(malformed, 2); !errors.Is(err, ErrCompiledProofEncoding) {
			t.Fatalf("got %v, want ErrCompiledProofEncoding", err)
		}
	})

	t.Run("noncanonical infinity", func(t *testing.T) {
		malformed := append([]byte(nil), encoded...)
		point := malformed[compiledProofHeaderSize : compiledProofHeaderSize+bn254.SizeOfG1AffineCompressed]
		for i := range point {
			point[i] = 0
		}
		point[0] = 0x40
		point[len(point)-1] = 1
		if _, err := DecodeCompiledProof(malformed, 2); !errors.Is(err, ErrCompiledProofEncoding) {
			t.Fatalf("got %v, want ErrCompiledProofEncoding", err)
		}
	})

	t.Run("uncompressed point", func(t *testing.T) {
		malformed := append([]byte(nil), encoded...)
		malformed[compiledProofHeaderSize] = 0
		if _, err := DecodeCompiledProof(malformed, 2); !errors.Is(err, ErrCompiledProofEncoding) {
			t.Fatalf("got %v, want ErrCompiledProofEncoding", err)
		}
	})
}

func TestCompiledProofUsesTranscriptMessageValidation(t *testing.T) {
	proof := compiledProofTestFixture(2)
	proof.W0.WitnessCommitments[0].Y.SetZero()
	if err := validateTranscriptG1(&proof.W0.WitnessCommitments[0]); err == nil {
		t.Fatal("malformed fixture unexpectedly passes transcript validation")
	}
	if _, err := proof.MarshalBinary(2); !errors.Is(err, ErrInvalidCompiledProof) {
		t.Fatalf("got %v, want ErrInvalidCompiledProof", err)
	}
}

func TestCompiledProofCanonicalTamperRemainsTranscriptSensitive(t *testing.T) {
	proof := compiledProofTestFixture(4)
	encoded, err := proof.MarshalBinary(4)
	if err != nil {
		t.Fatal(err)
	}

	tampered := append([]byte(nil), encoded...)
	replacementPoint := compiledProofTestG1(9999)
	replacement := replacementPoint.Bytes()
	copy(tampered[compiledProofHeaderSize:compiledProofHeaderSize+bn254.SizeOfG1AffineCompressed], replacement[:])
	decoded, err := DecodeCompiledProof(tampered, 4)
	if err != nil {
		t.Fatalf("canonical algebraic tamper should decode: %v", err)
	}
	if decoded.W0.WitnessCommitments[0].Equal(&proof.W0.WitnessCommitments[0]) {
		t.Fatal("canonical tamper did not alter W0")
	}

	left, err := NewOuterTranscript(testOuterTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	right, err := NewOuterTranscript(testOuterTranscriptContext())
	if err != nil {
		t.Fatal(err)
	}
	if err := left.AppendW0(proof.W0); err != nil {
		t.Fatal(err)
	}
	if err := right.AppendW0(decoded.W0); err != nil {
		t.Fatal(err)
	}
	leftChallenges, err := left.DeriveInitialChallenges()
	if err != nil {
		t.Fatal(err)
	}
	rightChallenges, err := right.DeriveInitialChallenges()
	if err != nil {
		t.Fatal(err)
	}
	if compiledProofTestChallengeBatchEqual(leftChallenges, rightChallenges) {
		t.Fatal("canonical W0 tamper did not change the following transcript challenges")
	}
}

func TestCompiledProofDecodeDoesNotAlias(t *testing.T) {
	proof := compiledProofTestFixture(4)
	encoded, err := proof.MarshalBinary(4)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeCompiledProof(encoded, 4)
	if err != nil {
		t.Fatal(err)
	}

	firstRound := decoded.SumCheckRounds[0].Coefficients[0]
	for i := range encoded {
		encoded[i] = 0
	}
	if !decoded.SumCheckRounds[0].Coefficients[0].Equal(&firstRound) {
		t.Fatal("decoded proof aliases encoded input")
	}
	decoded.SumCheckRounds[0].Coefficients[0] = fr.NewElement(123456)
	if proof.SumCheckRounds[0].Coefficients[0].Equal(&decoded.SumCheckRounds[0].Coefficients[0]) {
		t.Fatal("decoded SumCheck rounds alias the source proof")
	}

	destination := compiledProofTestFixture(2)
	want := destination
	if err := destination.UnmarshalBinary([]byte("malformed"), 2); !errors.Is(err, ErrCompiledProofEncoding) {
		t.Fatalf("got %v, want ErrCompiledProofEncoding", err)
	}
	if !reflect.DeepEqual(destination, want) {
		t.Fatal("failed unmarshal partially mutated its destination")
	}
}

func compiledProofTestFixture(partitions uint64) CompiledProof {
	rounds, err := compiledProofRoundCount(partitions)
	if err != nil {
		panic(err)
	}
	proof := CompiledProof{SumCheckRounds: make([]OuterSumCheckRoundMessage, rounds)}
	nextPoint := int64(1)
	point := func() bn254.G1Affine {
		result := compiledProofTestG1(nextPoint)
		nextPoint++
		return result
	}
	nextField := uint64(1001)
	field := func() fr.Element {
		result := fr.NewElement(nextField)
		nextField++
		return result
	}

	for i := range proof.W0.WitnessCommitments {
		proof.W0.WitnessCommitments[i] = point()
	}
	proof.W1.AccumulatorCommitment = point()
	for i := range proof.W2.QuotientCommitments {
		proof.W2.QuotientCommitments[i] = point()
	}
	for i := range proof.ProductCheck.Commitments {
		proof.ProductCheck.Commitments[i] = point()
	}
	for round := range proof.SumCheckRounds {
		for coefficient := range proof.SumCheckRounds[round].Coefficients {
			proof.SumCheckRounds[round].Coefficients[coefficient] = field()
		}
	}
	for i := range proof.FinalEvaluations.Terminal {
		proof.FinalEvaluations.Terminal[i] = field()
	}
	for i := range proof.FinalEvaluations.ProductCheck {
		proof.FinalEvaluations.ProductCheck[i] = field()
	}
	for i := range proof.U0.PartialCommitments {
		proof.U0.PartialCommitments[i] = point()
	}
	for i := range proof.U1.PartialAtChallenge {
		proof.U1.PartialAtChallenge[i] = field()
	}
	proof.U1.FunctionalCommitment = point()
	proof.U1.LaurentCommitment = point()
	for claim := range proof.U2.CircuitCrossValues {
		for cross := range proof.U2.CircuitCrossValues[claim] {
			proof.U2.CircuitCrossValues[claim][cross] = field()
		}
	}
	for i := range proof.U2.LaurentAtBeta {
		proof.U2.LaurentAtBeta[i] = field()
		proof.U2.LaurentAtBetaInverse[i] = field()
	}
	proof.U3.WCirc = point()
	proof.U3.WLaur = point()
	proof.U3.PiU = point()
	proof.U3.PiV = point()
	return proof
}

func compiledProofTestG1(scalar int64) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var result bn254.G1Affine
	result.ScalarMultiplication(&generator, big.NewInt(scalar))
	return result
}

func compiledProofTestPartitionName(partitions uint64) string {
	return "M=" + new(big.Int).SetUint64(partitions).String()
}

func bitsForCompiledProofTest(value uint64) int {
	count := 0
	for value != 0 {
		count++
		value >>= 1
	}
	return count
}

func compiledProofTestChallengeBatchEqual(left, right OuterInitialChallenges) bool {
	return left.EtaPart.Counter == right.EtaPart.Counter && left.EtaPart.Value.Equal(&right.EtaPart.Value) &&
		left.EtaX.Counter == right.EtaX.Counter && left.EtaX.Value.Equal(&right.EtaX.Value) &&
		left.Gamma.Counter == right.Gamma.Counter && left.Gamma.Value.Equal(&right.Gamma.Value)
}
