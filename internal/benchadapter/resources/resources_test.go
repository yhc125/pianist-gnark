package resources

import (
	"runtime"
	"testing"
)

func TestNormalizePeakRSS(t *testing.T) {
	tests := []struct {
		name         string
		value        int64
		bytesPerUnit uint64
		want         uint64
	}{
		{name: "zero bytes", value: 0, bytesPerUnit: 1, want: 0},
		{name: "Darwin bytes", value: 1234, bytesPerUnit: 1, want: 1234},
		{name: "Linux KiB", value: 1234, bytesPerUnit: 1024, want: 1263616},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizePeakRSS(test.value, test.bytesPerUnit)
			if err != nil {
				t.Fatalf("normalize peak RSS: %v", err)
			}
			if got != test.want {
				t.Fatalf("normalized RSS: got %d, want %d", got, test.want)
			}
		})
	}
}

func TestNormalizePeakRSSRejectsInvalidValues(t *testing.T) {
	if _, err := normalizePeakRSS(-1, 1); err == nil {
		t.Fatal("negative RSS value was accepted")
	}
	if _, err := normalizePeakRSS(1, 0); err == nil {
		t.Fatal("zero-byte RSS unit was accepted")
	}
	if _, err := normalizePeakRSS(1<<62, 1024); err == nil {
		t.Fatal("overflowing RSS value was accepted")
	}
}

func TestRead(t *testing.T) {
	sample, err := Read()
	if err != nil {
		t.Fatalf("read process resources: %v", err)
	}

	switch runtime.GOOS {
	case "linux", "darwin":
		if !sample.PeakRSSSupported {
			t.Fatalf("peak RSS unexpectedly unsupported on %s", runtime.GOOS)
		}
		if sample.PeakRSSBytes == 0 {
			t.Fatal("peak RSS is zero on a supported platform")
		}
	default:
		if sample.PeakRSSSupported {
			t.Fatalf("peak RSS unexpectedly supported on %s", runtime.GOOS)
		}
		if sample.PeakRSSBytes != 0 {
			t.Fatalf("unsupported peak RSS: got %d bytes, want zero", sample.PeakRSSBytes)
		}
	}
}
