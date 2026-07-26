package plonk

import (
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
)

func TestRunSyntheticMultiplication(t *testing.T) {
	for _, constraints := range []int{2, 8} {
		result, err := Run(Config{Constraints: constraints, Seed: 7})
		if err != nil {
			t.Fatalf("run PLONK adapter with %d constraints: %v", constraints, err)
		}
		if !result.Accepted {
			t.Fatal("valid proof was not accepted")
		}
		if result.RequestedConstraints != constraints {
			t.Fatalf("requested constraints: got %d, want %d", result.RequestedConstraints, constraints)
		}
		if result.RealizedConstraints != constraints {
			t.Fatalf("realized constraints: got %d, want %d", result.RealizedConstraints, constraints)
		}
		if result.ProofBytes <= 0 {
			t.Fatalf("proof bytes: got %d, want positive", result.ProofBytes)
		}
		if result.PublicVariables != 1 {
			t.Fatalf("public variables: got %d, want 1", result.PublicVariables)
		}
		wantDomain := ecc.NextPowerOfTwo(uint64(constraints + 1))
		if result.DomainSize != wantDomain {
			t.Fatalf("PLONK domain: got %d, want %d", result.DomainSize, wantDomain)
		}
		t.Logf("constraints=%d proof_bytes=%d", constraints, result.ProofBytes)
		assertPositiveDuration(t, "compile", result.Timings.Compile)
		assertPositiveDuration(t, "SRS", result.Timings.SRS)
		assertPositiveDuration(t, "setup", result.Timings.Setup)
		assertPositiveDuration(t, "witness solve", result.Timings.WitnessSolve)
		assertPositiveDuration(t, "cryptographic prove", result.Timings.ProveCrypto)
		assertPositiveDuration(t, "end-to-end prove", result.Timings.ProveEndToEnd)
		if result.Timings.ProveEndToEnd != result.Timings.WitnessSolve+result.Timings.ProveCrypto {
			t.Fatal("end-to-end prove timing is not the sum of solve and cryptographic prove")
		}
		assertPositiveDuration(t, "verify", result.Timings.Verify)
	}
}

func TestRunRejectsTooFewConstraints(t *testing.T) {
	if _, err := Run(Config{Constraints: 1}); err == nil {
		t.Fatal("expected invalid constraint count to fail")
	}
}

func assertPositiveDuration(t *testing.T, name string, got time.Duration) {
	t.Helper()
	if got <= 0 {
		t.Fatalf("%s duration: got %s, want positive", name, got)
	}
}
