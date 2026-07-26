package gpiano

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
	bn254witness "github.com/consensys/gnark/internal/backend/bn254/witness"
)

type splitSolveCircuit struct {
	X frontend.Variable
	Y frontend.Variable `gnark:",public"`
}

func (c *splitSolveCircuit) Define(api frontend.API) error {
	api.AssertIsEqual(api.Mul(c.X, c.X), c.Y)
	return nil
}

func compileSplitSolveCircuit(t *testing.T) *cs.SparseR1CS {
	t.Helper()
	ccs, err := frontend.Compile(ecc.BN254, scs.NewBuilder, &splitSolveCircuit{})
	if err != nil {
		t.Fatalf("compile circuit: %v", err)
	}
	return ccs.(*cs.SparseR1CS)
}

func splitSolveWitness(t *testing.T) bn254witness.Witness {
	t.Helper()
	w, err := frontend.NewWitness(&splitSolveCircuit{X: 3, Y: 9}, ecc.BN254)
	if err != nil {
		t.Fatalf("build witness: %v", err)
	}
	concrete, ok := w.Vector.(*bn254witness.Witness)
	if !ok {
		t.Fatalf("unexpected witness type %T", w.Vector)
	}
	return *concrete
}

func TestSolveProducesCircuitBoundSolution(t *testing.T) {
	spr := compileSplitSolveCircuit(t)
	solution, err := Solve(spr, splitSolveWitness(t), backend.ProverConfig{})
	if err != nil {
		t.Fatalf("solve circuit: %v", err)
	}
	if solution.system != spr {
		t.Fatal("solution is not bound to its constraint system")
	}
	wantWires := spr.NbPublicVariables + spr.NbSecretVariables + spr.NbInternalVariables
	if got := len(solution.wireValues); got != wantWires {
		t.Fatalf("solution wires: got %d, want %d", got, wantWires)
	}

	other := compileSplitSolveCircuit(t)
	if _, err := ProveWithSolution(other, nil, solution); !errors.Is(err, ErrInvalidSolution) {
		t.Fatalf("mismatched solution: got %v, want ErrInvalidSolution", err)
	}
	if _, err := ProveWithSolution(spr, nil, nil); !errors.Is(err, ErrInvalidSolution) {
		t.Fatalf("nil solution: got %v, want ErrInvalidSolution", err)
	}
}
