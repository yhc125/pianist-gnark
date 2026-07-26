// Package resources reads process-level resource measurements for benchmark
// adapters.
//
// Measurements are local to the calling process. In a distributed benchmark,
// every rank must read and report its own sample; choosing an aggregation (for
// example, maximum per-rank peak RSS) remains the caller's responsibility.
package resources

import "fmt"

// Sample is a process-level resource snapshot. PeakRSSBytes is the maximum
// resident-set size reached over the lifetime of the calling process, rather
// than its resident-set size at the instant Read is called.
//
// PeakRSSSupported is false on platforms for which this package has no native
// peak-RSS implementation. In that case PeakRSSBytes is zero.
type Sample struct {
	PeakRSSBytes     uint64
	PeakRSSSupported bool
}

// Read returns resource measurements for the calling process.
//
// Peak RSS is obtained from getrusage(RUSAGE_SELF) on Linux and Darwin. The
// platform-specific units are normalized to bytes. A distributed caller must
// invoke Read separately on every rank and perform any rank aggregation itself.
func Read() (Sample, error) {
	peakRSSBytes, supported, err := readPeakRSSBytes()
	if err != nil {
		return Sample{}, fmt.Errorf("read peak RSS: %w", err)
	}
	return Sample{
		PeakRSSBytes:     peakRSSBytes,
		PeakRSSSupported: supported,
	}, nil
}

func normalizePeakRSS(value int64, bytesPerUnit uint64) (uint64, error) {
	if value < 0 {
		return 0, fmt.Errorf("negative getrusage value %d", value)
	}
	if bytesPerUnit == 0 {
		return 0, fmt.Errorf("invalid zero-byte RSS unit")
	}

	unsigned := uint64(value)
	if unsigned > ^uint64(0)/bytesPerUnit {
		return 0, fmt.Errorf("getrusage value %d overflows bytes", value)
	}
	return unsigned * bytesPerUnit, nil
}
