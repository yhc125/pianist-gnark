//go:build darwin
// +build darwin

package resources

import "syscall"

func readPeakRSSBytes() (uint64, bool, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, true, err
	}

	// Darwin reports ru_maxrss in bytes.
	peakRSS, err := normalizePeakRSS(usage.Maxrss, 1)
	return peakRSS, true, err
}
