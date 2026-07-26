package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestStatefulDegreeFiveSumCheckMatchesOneShot(t *testing.T) {
	tables := deterministicOracleTables(3, 5)
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(tables, composition)
	challenges := []fr.Element{fieldElement(7), fieldElement(11), fieldElement(13)}

	wantTranscript, wantFinal, err := ProveDegreeFiveSumCheck(
		claim,
		tables,
		composition,
		challenges,
	)
	if err != nil {
		t.Fatalf("one-shot prover: %v", err)
	}

	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new stateful prover: %v", err)
	}
	gotTranscript := SumCheckTranscript{Rounds: make([][]fr.Element, 0, len(challenges))}
	for round, challenge := range challenges {
		coefficients, nextErr := prover.NextRound()
		if nextErr != nil {
			t.Fatalf("next round %d: %v", round, nextErr)
		}
		assertFieldSliceEqual(t, coefficients[:], wantTranscript.Rounds[round])
		gotTranscript.Rounds = append(gotTranscript.Rounds, append([]fr.Element(nil), coefficients[:]...))
		if bindErr := prover.BindChallenge(challenge); bindErr != nil {
			t.Fatalf("bind round %d: %v", round, bindErr)
		}
	}
	gotFinal, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("final evaluations: %v", err)
	}
	assertFieldSliceEqual(t, gotFinal, wantFinal)

	endpoint := composition(gotFinal)
	if err := VerifyDegreeFiveSumCheck(claim, challenges, gotTranscript, endpoint); err != nil {
		t.Fatalf("verify stateful transcript: %v", err)
	}
}

func TestStatefulDegreeFiveSumCheckCallOrder(t *testing.T) {
	tables := deterministicOracleTables(2, 3)
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(tables, composition)
	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new stateful prover: %v", err)
	}

	if _, err := prover.FinalEvaluations(); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("early final evaluations: got %v, want ErrSumCheckProverState", err)
	}
	if err := prover.BindChallenge(fieldElement(3)); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("bind before NextRound: got %v, want ErrSumCheckProverState", err)
	}
	first, err := prover.NextRound()
	if err != nil {
		t.Fatalf("first NextRound: %v", err)
	}
	if len(first) != SumCheckRoundCoefficients {
		t.Fatalf("coefficient count: got %d, want %d", len(first), SumCheckRoundCoefficients)
	}
	if _, err := prover.NextRound(); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("duplicate NextRound: got %v, want ErrSumCheckProverState", err)
	}

	// NextRound returns an array by value. Mutating it must not alter the round
	// polynomial retained for BindChallenge.
	first[0].Add(&first[0], fieldElementPointer(99))
	if err := prover.BindChallenge(fieldElement(5)); err != nil {
		t.Fatalf("bind first round: %v", err)
	}
	if err := prover.BindChallenge(fieldElement(7)); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("duplicate BindChallenge: got %v, want ErrSumCheckProverState", err)
	}

	if _, err := prover.NextRound(); err != nil {
		t.Fatalf("second NextRound: %v", err)
	}
	if err := prover.BindChallenge(fieldElement(7)); err != nil {
		t.Fatalf("bind second round: %v", err)
	}
	if _, err := prover.NextRound(); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("NextRound after completion: got %v, want ErrSumCheckProverState", err)
	}
	if err := prover.BindChallenge(fieldElement(9)); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("BindChallenge after completion: got %v, want ErrSumCheckProverState", err)
	}

	final, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("final evaluations: %v", err)
	}
	originalFirst := final[0]
	final[0].Add(&final[0], fieldElementPointer(1))
	again, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("repeat final evaluations: %v", err)
	}
	if !again[0].Equal(&originalFirst) {
		t.Fatal("FinalEvaluations returned an alias of prover state")
	}

	var nilProver *DegreeFiveSumCheckProver
	if _, err := nilProver.NextRound(); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("nil NextRound: got %v, want ErrSumCheckProverState", err)
	}
	if err := nilProver.BindChallenge(fr.Element{}); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("nil BindChallenge: got %v, want ErrSumCheckProverState", err)
	}
	if _, err := nilProver.FinalEvaluations(); !errors.Is(err, ErrSumCheckProverState) {
		t.Fatalf("nil FinalEvaluations: got %v, want ErrSumCheckProverState", err)
	}
}

func TestStatefulDegreeFiveSumCheckAcceptsBooleanChallenges(t *testing.T) {
	tables := deterministicOracleTables(2, 2)
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(tables, composition)
	challenges := []fr.Element{{}, fr.One()}

	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new stateful prover: %v", err)
	}
	transcript := SumCheckTranscript{Rounds: make([][]fr.Element, 0, len(challenges))}
	for round, challenge := range challenges {
		coefficients, nextErr := prover.NextRound()
		if nextErr != nil {
			t.Fatalf("NextRound %d: %v", round, nextErr)
		}
		transcript.Rounds = append(transcript.Rounds, append([]fr.Element(nil), coefficients[:]...))
		if bindErr := prover.BindChallenge(challenge); bindErr != nil {
			t.Fatalf("BindChallenge(%s): %v", challenge.String(), bindErr)
		}
	}
	final, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("final evaluations: %v", err)
	}
	if err := VerifyDegreeFiveSumCheck(claim, challenges, transcript, composition(final)); err != nil {
		t.Fatalf("verify Boolean-challenge transcript: %v", err)
	}
}

func TestStatefulDegreeFiveSumCheckCopiesInputTables(t *testing.T) {
	tables := deterministicOracleTables(3, 4)
	snapshot := cloneStatefulOracleTables(tables)
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(snapshot, composition)
	challenges := []fr.Element{fieldElement(7), fieldElement(11), fieldElement(13)}

	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new stateful prover: %v", err)
	}
	for oracle := range tables {
		for index := range tables[oracle] {
			tables[oracle][index].SetUint64(uint64(1000 + 17*oracle + index))
		}
	}

	wantTranscript, wantFinal, err := ProveDegreeFiveSumCheck(
		claim,
		snapshot,
		composition,
		challenges,
	)
	if err != nil {
		t.Fatalf("one-shot snapshot prover: %v", err)
	}
	for round, challenge := range challenges {
		coefficients, nextErr := prover.NextRound()
		if nextErr != nil {
			t.Fatalf("NextRound %d: %v", round, nextErr)
		}
		assertFieldSliceEqual(t, coefficients[:], wantTranscript.Rounds[round])
		if bindErr := prover.BindChallenge(challenge); bindErr != nil {
			t.Fatalf("BindChallenge %d: %v", round, bindErr)
		}
	}
	gotFinal, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("final evaluations: %v", err)
	}
	assertFieldSliceEqual(t, gotFinal, wantFinal)
}

func TestStatefulDegreeFiveSumCheckRejectsInvalidConstruction(t *testing.T) {
	composition := productSumCheckComposition
	one := fr.One()

	tests := []struct {
		name   string
		tables [][]fr.Element
		claim  fr.Element
		comp   SumCheckComposition
		want   error
	}{
		{name: "nil composition", tables: [][]fr.Element{{{}, one}}, comp: nil, want: ErrInvalidSumCheckInput},
		{name: "no tables", comp: composition, want: ErrInvalidSumCheckInput},
		{name: "empty table", tables: [][]fr.Element{{}}, comp: composition, want: ErrInvalidSumCheckInput},
		{name: "zero rounds", tables: [][]fr.Element{{one}}, claim: one, comp: composition, want: ErrInvalidSumCheckInput},
		{name: "non power of two", tables: [][]fr.Element{{{}, one, one}}, comp: composition, want: ErrInvalidSumCheckInput},
		{name: "mismatched lengths", tables: [][]fr.Element{{{}, one}, {{}, one, one, one}}, comp: composition, want: ErrInvalidSumCheckInput},
		{name: "wrong claim", tables: [][]fr.Element{{{}, one}}, comp: composition, want: ErrInvalidSumCheckClaim},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewDegreeFiveSumCheckProver(test.claim, test.tables, test.comp); !errors.Is(err, test.want) {
				t.Fatalf("got %v, want %v", err, test.want)
			}
		})
	}
}

func TestStatefulDegreeFiveSumCheckRejectsDegreeSixEndpoint(t *testing.T) {
	// Every oracle is the MLE of [0,1], so the single round's true polynomial
	// is X^6. Six samples determine only its degree-five interpolation; binding
	// at 7 therefore exposes the degree violation at the final endpoint.
	tables := make([][]fr.Element, 6)
	for oracle := range tables {
		tables[oracle] = []fr.Element{{}, fr.One()}
	}
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(tables, composition)
	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new degree-six prover: %v", err)
	}
	if _, err := prover.NextRound(); err != nil {
		t.Fatalf("degree-six NextRound: %v", err)
	}
	if err := prover.BindChallenge(fieldElement(7)); err != nil {
		t.Fatalf("degree-six BindChallenge: %v", err)
	}
	if _, err := prover.FinalEvaluations(); !errors.Is(err, ErrSumCheckDegree) {
		t.Fatalf("degree-six endpoint: got %v, want ErrSumCheckDegree", err)
	}
}

func TestStatefulDegreeFiveSumCheckTranscriptTamperingRejected(t *testing.T) {
	tables := deterministicOracleTables(2, 3)
	composition := productSumCheckComposition
	claim := compositionHypercubeSum(tables, composition)
	challenges := []fr.Element{fieldElement(5), fieldElement(7)}
	prover, err := NewDegreeFiveSumCheckProver(claim, tables, composition)
	if err != nil {
		t.Fatalf("new stateful prover: %v", err)
	}
	transcript := SumCheckTranscript{Rounds: make([][]fr.Element, 0, len(challenges))}
	for round, challenge := range challenges {
		coefficients, nextErr := prover.NextRound()
		if nextErr != nil {
			t.Fatalf("NextRound %d: %v", round, nextErr)
		}
		transcript.Rounds = append(transcript.Rounds, append([]fr.Element(nil), coefficients[:]...))
		if bindErr := prover.BindChallenge(challenge); bindErr != nil {
			t.Fatalf("BindChallenge %d: %v", round, bindErr)
		}
	}
	final, err := prover.FinalEvaluations()
	if err != nil {
		t.Fatalf("final evaluations: %v", err)
	}
	endpoint := composition(final)

	tamperedRound := cloneSumCheckTranscript(transcript)
	tamperedRound.Rounds[0][2].Add(&tamperedRound.Rounds[0][2], fieldElementPointer(1))
	if err := VerifyDegreeFiveSumCheck(claim, challenges, tamperedRound, endpoint); !errors.Is(err, ErrInvalidSumCheckTranscript) {
		t.Fatalf("round tampering: got %v, want ErrInvalidSumCheckTranscript", err)
	}
	tamperedEndpoint := endpoint
	tamperedEndpoint.Add(&tamperedEndpoint, fieldElementPointer(1))
	if err := VerifyDegreeFiveSumCheck(claim, challenges, transcript, tamperedEndpoint); !errors.Is(err, ErrInvalidSumCheckTranscript) {
		t.Fatalf("endpoint tampering: got %v, want ErrInvalidSumCheckTranscript", err)
	}
}

func productSumCheckComposition(values []fr.Element) fr.Element {
	result := fr.One()
	for i := range values {
		result.Mul(&result, &values[i])
	}
	return result
}

func cloneStatefulOracleTables(tables [][]fr.Element) [][]fr.Element {
	clone := make([][]fr.Element, len(tables))
	for oracle := range tables {
		clone[oracle] = append([]fr.Element(nil), tables[oracle]...)
	}
	return clone
}
