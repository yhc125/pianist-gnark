//go:build !linux && !darwin
// +build !linux,!darwin

package resources

func readPeakRSSBytes() (uint64, bool, error) {
	return 0, false, nil
}
