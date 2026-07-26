package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestDegreeFiveSumCheckHonestTranscript(t *testing.T) {
	theta := []fr.Element{fieldElement(2), fieldElement(4), fieldElement(6)}
	equalityTable, err := EqualityEvaluationTable(theta)
	if err != nil {
		t.Fatalf("build equality table: %v", err)
	}

	tables := [][]fr.Element{equalityTable}
	for oracle := 0; oracle < 4; oracle++ {
		table := make([]fr.Element, 8)
		for i := range table {
			table[i].SetUint64(uint64((oracle+2)*(i+1) + oracle + 1))
		}
		tables = append(tables, table)
	}
	productComposition := func(values []fr.Element) fr.Element {
		result := fieldElement(1)
		for i := range values {
			result.Mul(&result, &values[i])
		}
		return result
	}
	rawSum := compositionHypercubeSum(tables, productComposition)
	var offset fr.Element
	offset.Div(&rawSum, fieldElementPointer(uint64(len(tables[0]))))
	composition := func(values []fr.Element) fr.Element {
		result := productComposition(values)
		result.Sub(&result, &offset)
		return result
	}
	var claim fr.Element // Protocol 1's declared SumCheck claim is zero.
	challenges := []fr.Element{fieldElement(7), fieldElement(11), fieldElement(13)}

	transcript, finalEvaluations, err := ProveDegreeFiveSumCheck(claim, tables, composition, challenges)
	if err != nil {
		t.Fatalf("prove degree-five SumCheck: %v", err)
	}
	if len(transcript.Rounds) != len(challenges) {
		t.Fatalf("round count: got %d, want %d", len(transcript.Rounds), len(challenges))
	}
	for round := range transcript.Rounds {
		if len(transcript.Rounds[round]) != SumCheckRoundCoefficients {
			t.Fatalf("round %d coefficient count: got %d, want 6", round+1, len(transcript.Rounds[round]))
		}
	}

	for oracle := range tables {
		want, err := EvaluateMultilinear(tables[oracle], challenges)
		if err != nil {
			t.Fatalf("evaluate final oracle %d: %v", oracle, err)
		}
		if !finalEvaluations[oracle].Equal(&want) {
			t.Fatalf("final oracle %d does not match direct MLE evaluation", oracle)
		}
	}
	endpoint := composition(finalEvaluations)
	if err := VerifyDegreeFiveSumCheck(claim, challenges, transcript, endpoint); err != nil {
		t.Fatalf("verify honest degree-five SumCheck: %v", err)
	}
}

func TestDegreeFiveSumCheckRejectsTampering(t *testing.T) {
	tables := deterministicOracleTables(3, 5)
	composition := func(values []fr.Element) fr.Element {
		result := fieldElement(1)
		for i := range values {
			result.Mul(&result, &values[i])
		}
		return result
	}
	claim := compositionHypercubeSum(tables, composition)
	challenges := []fr.Element{fieldElement(7), fieldElement(11), fieldElement(13)}
	transcript, finalEvaluations, err := ProveDegreeFiveSumCheck(claim, tables, composition, challenges)
	if err != nil {
		t.Fatalf("prove degree-five SumCheck: %v", err)
	}
	endpoint := composition(finalEvaluations)

	tampered := cloneSumCheckTranscript(transcript)
	tampered.Rounds[1][3].Add(&tampered.Rounds[1][3], fieldElementPointer(1))
	if err := VerifyDegreeFiveSumCheck(claim, challenges, tampered, endpoint); !errors.Is(err, ErrInvalidSumCheckTranscript) {
		t.Fatalf("coefficient tampering: got %v, want ErrInvalidSumCheckTranscript", err)
	}

	malformed := cloneSumCheckTranscript(transcript)
	malformed.Rounds[0] = malformed.Rounds[0][:SumCheckRoundDegree]
	if err := VerifyDegreeFiveSumCheck(claim, challenges, malformed, endpoint); !errors.Is(err, ErrInvalidSumCheckTranscript) {
		t.Fatalf("malformed coefficient encoding: got %v, want ErrInvalidSumCheckTranscript", err)
	}

	tamperedEndpoint := endpoint
	tamperedEndpoint.Add(&tamperedEndpoint, fieldElementPointer(1))
	if err := VerifyDegreeFiveSumCheck(claim, challenges, transcript, tamperedEndpoint); !errors.Is(err, ErrInvalidSumCheckTranscript) {
		t.Fatalf("endpoint tampering: got %v, want ErrInvalidSumCheckTranscript", err)
	}

	wrongClaim := claim
	wrongClaim.Add(&wrongClaim, fieldElementPointer(1))
	if _, _, err := ProveDegreeFiveSumCheck(wrongClaim, tables, composition, challenges); !errors.Is(err, ErrInvalidSumCheckClaim) {
		t.Fatalf("wrong prover claim: got %v, want ErrInvalidSumCheckClaim", err)
	}
}

func TestSumCheckCanonicalLeadingZeroPadding(t *testing.T) {
	tables := deterministicOracleTables(3, 2)
	composition := func(values []fr.Element) fr.Element {
		var product, result fr.Element
		product.Mul(&values[0], &values[1])
		result.Add(&product, &values[0])
		result.Add(&result, fieldElementPointer(9))
		return result
	}
	claim := compositionHypercubeSum(tables, composition)
	challenges := []fr.Element{fieldElement(7), fieldElement(11), fieldElement(13)}

	transcript, finalEvaluations, err := ProveDegreeFiveSumCheck(claim, tables, composition, challenges)
	if err != nil {
		t.Fatalf("prove padded SumCheck: %v", err)
	}
	for round := range transcript.Rounds {
		for degree := 3; degree < SumCheckRoundCoefficients; degree++ {
			if !transcript.Rounds[round][degree].IsZero() {
				t.Fatalf("round %d coefficient X^%d is not canonical zero padding", round+1, degree)
			}
		}
	}
	endpoint := composition(finalEvaluations)
	if err := VerifyDegreeFiveSumCheck(claim, challenges, transcript, endpoint); err != nil {
		t.Fatalf("verify padded SumCheck: %v", err)
	}
}

func TestEqualityEvaluationTableUsesLittleEndianBits(t *testing.T) {
	theta := []fr.Element{fieldElement(2), fieldElement(3)}
	table, err := EqualityEvaluationTable(theta)
	if err != nil {
		t.Fatalf("build equality table: %v", err)
	}

	var oneMinusTheta0, oneMinusTheta1 fr.Element
	oneMinusTheta0.SetOne().Sub(&oneMinusTheta0, &theta[0])
	oneMinusTheta1.SetOne().Sub(&oneMinusTheta1, &theta[1])
	want := make([]fr.Element, 4)
	want[0].Mul(&oneMinusTheta0, &oneMinusTheta1) // bits (0,0)
	want[1].Mul(&theta[0], &oneMinusTheta1)       // bits (1,0)
	want[2].Mul(&oneMinusTheta0, &theta[1])       // bits (0,1)
	want[3].Mul(&theta[0], &theta[1])             // bits (1,1)
	assertFieldSliceEqual(t, table, want)
}

func deterministicOracleTables(variables, oracles int) [][]fr.Element {
	tables := make([][]fr.Element, oracles)
	for oracle := range tables {
		tables[oracle] = make([]fr.Element, 1<<uint(variables))
		for index := range tables[oracle] {
			tables[oracle][index].SetUint64(uint64((oracle+2)*(index+1) + 3))
		}
	}
	return tables
}

func compositionHypercubeSum(tables [][]fr.Element, composition SumCheckComposition) fr.Element {
	values := make([]fr.Element, len(tables))
	var sum fr.Element
	for index := range tables[0] {
		for oracle := range tables {
			values[oracle] = tables[oracle][index]
		}
		term := composition(values)
		sum.Add(&sum, &term)
	}
	return sum
}

func cloneSumCheckTranscript(transcript SumCheckTranscript) SumCheckTranscript {
	clone := SumCheckTranscript{Rounds: make([][]fr.Element, len(transcript.Rounds))}
	for round := range transcript.Rounds {
		clone.Rounds[round] = append([]fr.Element(nil), transcript.Rounds[round]...)
	}
	return clone
}
