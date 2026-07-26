//go:build linux
// +build linux

package resources

import "syscall"

func readPeakRSSBytes() (uint64, bool, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, true, err
	}

	// Linux reports ru_maxrss in KiB.
	peakRSS, err := normalizePeakRSS(usage.Maxrss, 1024)
	return peakRSS, true, err
}
