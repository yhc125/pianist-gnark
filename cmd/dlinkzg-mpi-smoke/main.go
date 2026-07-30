package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"strings"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	backend "github.com/consensys/gnark/internal/backend/bn254/dlinkzg"
	"github.com/sunblaze-ucb/simpleMPI/mpi"
)

type result struct {
	WorldSize  uint64                `json:"world_size"`
	Accepted   bool                  `json:"accepted"`
	Accounting backend.MPIAccounting `json:"accounting"`
}

func initializeWorld() {
	if len(os.Args) > 1 && strings.EqualFold(os.Args[len(os.Args)-1], "slave") {
		mpi.WorldInit("", "", "")
		return
	}
	ipFile := os.Getenv("PIANIST_MPI_IP_FILE")
	sshKey := os.Getenv("PIANIST_MPI_SSH_KEY")
	sshUser := os.Getenv("PIANIST_MPI_SSH_USER")
	localLauncher := strings.EqualFold(
		strings.TrimSpace(os.Getenv("SIMPLEMPI_LAUNCH_MODE")),
		"local",
	)
	if ipFile == "" || (!localLauncher && (sshKey == "" || sshUser == "")) {
		fmt.Fprintln(
			os.Stderr,
			"dlinkzg MPI smoke needs PIANIST_MPI_IP_FILE and SSH credentials unless using the local launcher",
		)
		os.Exit(2)
	}
	mpi.WorldInit(ipFile, sshKey, sshUser)
}

func pointForScalar(scalar fr.Element) bn254.G1Affine {
	_, _, generator, _ := bn254.Generators()
	var regular big.Int
	scalar.ToBigIntRegular(&regular)
	var point bn254.G1Affine
	point.ScalarMultiplication(&generator, &regular)
	return point
}

func expectedScalar(phase backend.MPIPhase) fr.Element {
	phaseOffset := uint64(phase) + 1
	sum := mpi.WorldSize*phaseOffset + mpi.WorldSize*(mpi.WorldSize-1)/2
	return fr.NewElement(sum)
}

func run() (backend.MPIAccounting, error) {
	channel, err := backend.NewMPIChannel()
	if err != nil {
		return backend.MPIAccounting{}, err
	}
	for phase := backend.MPIPhaseU0; phase <= backend.MPIPhaseU3; phase++ {
		if err := channel.Barrier(phase); err != nil {
			return backend.MPIAccounting{}, err
		}
		localScalar := fr.NewElement(mpi.SelfRank + uint64(phase) + 1)
		local := backend.MPIPayload{
			Fields: []fr.Element{localScalar},
			G1:     []bn254.G1Affine{pointForScalar(localScalar)},
		}
		aggregate, err := channel.RootAggregate(phase, local)
		if err != nil {
			return backend.MPIAccounting{}, err
		}
		shape := backend.MPIPayloadShape{Fields: 1, G1: 1}
		broadcast, err := channel.RootBroadcast(phase, shape, aggregate)
		if err != nil {
			return backend.MPIAccounting{}, err
		}
		want := expectedScalar(phase)
		if len(broadcast.Fields) != 1 || !broadcast.Fields[0].Equal(&want) {
			return backend.MPIAccounting{}, fmt.Errorf("phase %s field aggregate mismatch", phase)
		}
		wantPoint := pointForScalar(want)
		if len(broadcast.G1) != 1 || !broadcast.G1[0].Equal(&wantPoint) {
			return backend.MPIAccounting{}, fmt.Errorf("phase %s G1 aggregate mismatch", phase)
		}
	}
	return channel.Accounting(), nil
}

func main() {
	initializeWorld()
	workers := flag.Uint64("workers", mpi.WorldSize, "expected MPI world size")
	flag.Parse()
	if *workers != mpi.WorldSize {
		fmt.Fprintf(os.Stderr, "workers=%d does not match MPI world size %d\n", *workers, mpi.WorldSize)
		os.Exit(2)
	}
	accounting, err := run()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if mpi.SelfRank != 0 {
		return
	}
	if err := json.NewEncoder(os.Stdout).Encode(result{
		WorldSize:  mpi.WorldSize,
		Accepted:   true,
		Accounting: accounting,
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
