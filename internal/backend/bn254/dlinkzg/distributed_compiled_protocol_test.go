package dlinkzg

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type compiledMPITestResult struct {
	proof      *CompiledProof
	trace      compiledMPITrace
	accounting ProtocolMPIAccounting
	err        error
}

func TestCompiledProveMPIEqualsCentralProofM2M4(t *testing.T) {
	for _, partitions := range []int{2, 4} {
		partitions := partitions
		t.Run(adapterWorldName(partitions), func(t *testing.T) {
			fixture := newCompiledProtocolTestFixture(t, partitions, byte(90+partitions))
			centralEncoded, err := fixture.proof.MarshalBinary(uint64(partitions))
			if err != nil {
				t.Fatalf("marshal central proof: %v", err)
			}

			worldSize := uint64(partitions)
			network := newFakeMPINetwork(worldSize)
			channels := make([]*ProtocolMPIChannel, worldSize)
			roles := make([]CompiledMPIRole, worldSize)
			solutions := make([][]fr.Element, worldSize)
			coordinator := fixture.setup.Coordinator
			for rank := uint64(0); rank < worldSize; rank++ {
				party := fixture.setup.Parties[rank]
				roles[rank] = CompiledMPIRole{Party: &party}
				if rank == 0 {
					roles[rank].Coordinator = &coordinator
				}
				solutions[rank] = append([]fr.Element(nil), fixture.solution...)
			}
			// The goroutines receive only their extracted role. Removing the
			// bundle's role collections makes accidental full-setup access fail.
			fixture.setup.Parties = nil
			fixture.setup.Coordinator = CompiledCoordinatorSetup{}
			for rank := uint64(0); rank < worldSize; rank++ {
				channels[rank], err = newProtocolMPIChannel(network.transport(rank))
				if err != nil {
					t.Fatalf("new channel rank %d: %v", rank, err)
				}
			}

			results := make([]compiledMPITestResult, worldSize)
			var wait sync.WaitGroup
			for rank := uint64(0); rank < worldSize; rank++ {
				rank := rank
				wait.Add(1)
				go func() {
					defer wait.Done()
					results[rank].proof, results[rank].trace, results[rank].err = compiledProveMPIWithTrace(
						channels[rank], fixture.vk, fixture.statement, roles[rank], solutions[rank],
					)
					results[rank].accounting = channels[rank].Accounting()
				}()
			}
			done := make(chan struct{})
			go func() {
				wait.Wait()
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(30 * time.Second):
				t.Fatal("distributed compiled protocol deadlocked")
			}

			for rank := uint64(0); rank < worldSize; rank++ {
				if results[rank].err != nil {
					t.Fatalf("rank %d: %v", rank, results[rank].err)
				}
				if rank == 0 {
					if results[rank].proof == nil {
						t.Fatal("coordinator returned nil proof")
					}
				} else if results[rank].proof != nil {
					t.Fatalf("party rank %d returned a public proof", rank)
				}
				if results[rank].trace != results[0].trace {
					t.Fatalf("rank %d transcript trace differs from coordinator", rank)
				}
				wantAccounting := expectedProtocolMPIAccounting(rank, uint64(partitions))
				if !reflect.DeepEqual(results[rank].accounting, wantAccounting) {
					t.Fatalf("rank %d accounting\n got: %+v\nwant: %+v", rank, results[rank].accounting, wantAccounting)
				}
				if results[rank].accounting.Operations != 17 {
					t.Fatalf("rank %d operations = %d, want 17", rank, results[rank].accounting.Operations)
				}
			}
			distributedEncoded, err := results[0].proof.MarshalBinary(uint64(partitions))
			if err != nil {
				t.Fatalf("marshal distributed proof: %v", err)
			}
			if !bytes.Equal(distributedEncoded, centralEncoded) {
				t.Fatal("distributed proof differs byte-for-byte from centralized proof")
			}
			if err := CompiledVerify(fixture.vk, fixture.statement, *results[0].proof); err != nil {
				t.Fatalf("verify distributed proof: %v", err)
			}

			logM := bitsForCompiledProtocol(partitions)
			root := results[0].accounting.Total
			wantRootReceive := uint64((partitions - 1) * 1408)
			wantRootSend := uint64((partitions - 1) * (1984 + 192*logM))
			if root.PayloadBytesRecv != wantRootReceive || root.PayloadBytesSent != wantRootSend {
				t.Fatalf(
					"root payload bytes recv/send = %d/%d, want %d/%d",
					root.PayloadBytesRecv, root.PayloadBytesSent, wantRootReceive, wantRootSend,
				)
			}
		})
	}
}

func TestCompiledMPIRoleValidationIsCompositeAndRankLocal(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 2, 111)
	network := newFakeMPINetwork(2)
	root, err := newProtocolMPIChannel(network.transport(0))
	if err != nil {
		t.Fatal(err)
	}
	partyOne, err := newProtocolMPIChannel(network.transport(1))
	if err != nil {
		t.Fatal(err)
	}
	coordinator := fixture.setup.Coordinator
	party0 := fixture.setup.Parties[0]
	party1 := fixture.setup.Parties[1]

	tests := []struct {
		name     string
		channel  *ProtocolMPIChannel
		role     CompiledMPIRole
		solution []fr.Element
	}{
		{"root missing coordinator", root, CompiledMPIRole{Party: &party0}, fixture.solution},
		{"root missing party", root, CompiledMPIRole{Coordinator: &coordinator}, nil},
		{"root missing solution", root, CompiledMPIRole{Coordinator: &coordinator, Party: &party0}, nil},
		{"root wrong party slot", root, CompiledMPIRole{Coordinator: &coordinator, Party: &party1}, fixture.solution},
		{"party missing role", partyOne, CompiledMPIRole{}, fixture.solution},
		{"party also owns coordinator", partyOne, CompiledMPIRole{Party: &party1, Coordinator: &coordinator}, fixture.solution},
		{"wrong party slot", partyOne, CompiledMPIRole{Party: &party0}, fixture.solution},
		{"party missing solution", partyOne, CompiledMPIRole{Party: &party1}, nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateCompiledMPIRole(
				test.channel, fixture.vk, fixture.statement, test.role, test.solution,
			); !errors.Is(err, ErrInvalidCompiledMPIRole) {
				t.Fatalf("error = %v, want ErrInvalidCompiledMPIRole", err)
			}
		})
	}

	if err := validateCompiledMPIRole(
		root, fixture.vk, fixture.statement,
		CompiledMPIRole{Coordinator: &coordinator, Party: &party0}, fixture.solution,
	); err != nil {
		t.Fatalf("valid composite root role: %v", err)
	}
	if err := validateCompiledMPIRole(
		partyOne, fixture.vk, fixture.statement,
		CompiledMPIRole{Party: &party1}, fixture.solution,
	); err != nil {
		t.Fatalf("valid party role: %v", err)
	}
}

func TestCompiledMPIRoleTypeContainsNoFullSetup(t *testing.T) {
	typeOfRole := reflect.TypeOf(CompiledMPIRole{})
	for field := 0; field < typeOfRole.NumField(); field++ {
		fieldType := typeOfRole.Field(field).Type
		if fieldType == reflect.TypeOf((*CompiledSetup)(nil)) ||
			fieldType == reflect.TypeOf([]CompiledPartySetup{}) {
			t.Fatalf("role field %s exposes a full setup/party collection", typeOfRole.Field(field).Name)
		}
	}
	if typeOfRole.NumField() != 2 {
		t.Fatalf("CompiledMPIRole has %d fields, want exactly Party/Coordinator", typeOfRole.NumField())
	}
}

func TestProvisionCompiledMPIRoleExtractsOnlyAssignedRank(t *testing.T) {
	fixture := newCompiledProtocolTestFixture(t, 4, 121)
	for rank := uint64(0); rank < 4; rank++ {
		vk, role, err := ProvisionCompiledMPIRole(fixture.setup, rank)
		if err != nil {
			t.Fatalf("rank %d provision: %v", rank, err)
		}
		if vk.Metadata != fixture.vk.Metadata {
			t.Fatalf("rank %d received different VK metadata", rank)
		}
		if role.Party == nil || role.Party.Rank != int(rank) ||
			role.Party == &fixture.setup.Parties[rank] {
			t.Fatalf("rank %d party role was not the isolated slot copy", rank)
		}
		if rank == 0 {
			if role.Coordinator == nil || role.Coordinator == &fixture.setup.Coordinator {
				t.Fatal("coordinator role was not an isolated value copy")
			}
		} else {
			if role.Coordinator != nil {
				t.Fatalf("rank %d party role was not the isolated slot copy", rank)
			}
		}
	}
	if _, _, err := ProvisionCompiledMPIRole(fixture.setup, 4); !errors.Is(err, ErrInvalidCompiledMPIRole) {
		t.Fatalf("out-of-range rank error = %v", err)
	}
	if _, _, err := ProvisionCompiledMPIRole(nil, 0); err == nil {
		t.Fatal("nil setup provision was accepted")
	}
}

func ExampleCompiledMPIRole() {
	role := CompiledMPIRole{Party: &CompiledPartySetup{Rank: 0}}
	fmt.Println(role.Party != nil && role.Coordinator == nil)
	// Output: true
}
