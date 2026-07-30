package dlinkzg

// This file assembles the witness-independent objects used by the centralized
// compiled-protocol benchmark. The only constructor is deliberately named as
// a deterministic benchmark/test helper: a production deployment must replace
// it with authenticated role views from an MPC ceremony.

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
)

const (
	// DeterministicCompiledSetupNotice fixes the security boundary of
	// NewDeterministicCompiledSetup. TauY and TauZ are accepted only by that
	// benchmark/test factory and are never retained in any returned online key.
	DeterministicCompiledSetupNotice = "deterministic compiled setup: benchmark/test only; production must use authenticated MPC-derived role views"

	compiledIndexPrefixDomain = "DLinKZG/compiled-setup/index-prefix/v1"
	compiledIndexDomain       = "DLinKZG/compiled-setup/index/v1"
	compiledSRSDomain         = "DLinKZG/compiled-setup/srs/v1"
	compiledManifestDomain    = "DLinKZG/compiled-setup/party-manifest/v1"
	compiledSetupDomain       = "DLinKZG/compiled-setup/metadata/v1"
)

var (
	// ErrInvalidCompiledSetup reports a malformed circuit, protocol shape, or
	// role view while constructing or validating the benchmark setup.
	ErrInvalidCompiledSetup = errors.New("dlinkzg: invalid compiled setup")
	// ErrCompiledSetupMetadataMismatch reports that a transcript context or an
	// online role is not bound to this setup's authenticated metadata.
	ErrCompiledSetupMetadataMismatch = errors.New("dlinkzg: compiled setup metadata mismatch")
)

// CompiledSetupMetadata is the public, verifiable identity shared by every
// role. CircuitDigest binds the SparseR1CS structure. IndexPrefixDigest binds
// the manifest-ordered adapter preprocessing before the canonical index
// shift; IndexDigest additionally binds (counter,sigma). SRSDigest and
// PartyManifestDigest are recomputed from the split role views. SetupDigest
// binds the complete record and the aggregate fixed-index commitments.
//
// The metadata contains no KZG trapdoor. Shift is public protocol metadata.
type CompiledSetupMetadata struct {
	CircuitDigest       TranscriptDigest
	IndexPrefixDigest   TranscriptDigest
	IndexDigest         TranscriptDigest
	SRSDigest           TranscriptDigest
	PartyManifestDigest TranscriptDigest
	SetupDigest         TranscriptDigest

	IndexShift           IndexShift
	PartitionCount       int
	LocalDomainSize      int
	PublicVariableCount  int
	LocalDomainGenerator fr.Element
}

// CompiledPartySetup is one manifest slot's online proving key. RowSRS owns
// exactly one semantic/native mixed row and one shared-Z row; it is not a
// monolithic rectangular SRS. Preprocessing and FastFixed are reusable across
// witnesses.
type CompiledPartySetup struct {
	Rank             int
	Metadata         CompiledSetupMetadata
	ConstraintSystem *cs.SparseR1CS
	Preprocessing    *LocalPIOPPreprocessing
	FastFixed        *FastLocalPIOPFixed
	FastTaylorShift  *cryptodlinkzg.FastTaylorShiftPrecomputation
	RowSRS           *cryptodlinkzg.PartyRowSRS
	FixedCommitments LocalPIOPFixedCommitmentSlot
}

// CompiledCoordinatorSetup is the root-only online role. It has only the O(M)
// coordinator SRS view, the aggregate fixed index, and the public-input map.
type CompiledCoordinatorSetup struct {
	Metadata             CompiledSetupMetadata
	SRS                  *cryptodlinkzg.CoordinatorSRS
	FixedCommitments     AggregateLocalPIOPFixedCommitments
	PublicInputPlacement *StatementPublicInputPlacement
}

// CompiledVerifierSetup is the constant-SRS verifier role plus the public
// fixed-index and statement-placement data used by the outer verifier.
type CompiledVerifierSetup struct {
	Metadata             CompiledSetupMetadata
	SRS                  *cryptodlinkzg.VerifierSRS
	FixedCommitments     AggregateLocalPIOPFixedCommitments
	PublicInputPlacement *StatementPublicInputPlacement
}

// CompiledSetup is the centralized benchmark bundle. It keeps the party keys
// as distinct rank roles so the same objects can later be handed to M worker
// processes; there is no M-by-T rectangular SRS field.
type CompiledSetup struct {
	Metadata    CompiledSetupMetadata
	Parties     []CompiledPartySetup
	Coordinator CompiledCoordinatorSetup
	Verifier    CompiledVerifierSetup
}

// NewDeterministicCompiledSetup constructs all reusable setup state for a real
// BN254 SparseR1CS. tauY and tauZ are transient benchmark/test inputs. The
// returned bundle stores only group elements and public metadata.
func NewDeterministicCompiledSetup(
	spr *cs.SparseR1CS,
	partitions int,
	tauY, tauZ fr.Element,
) (*CompiledSetup, error) {
	if spr == nil {
		return nil, fmt.Errorf("%w: nil SparseR1CS", ErrInvalidCompiledSetup)
	}
	if partitions < 2 || !isPowerOfTwo(partitions) {
		return nil, fmt.Errorf("%w: M=%d is not a power of two at least two", ErrInvalidCompiledSetup, partitions)
	}
	if err := validateTranscriptField(&tauY); err != nil || tauY.IsZero() {
		return nil, fmt.Errorf("%w: invalid benchmark tauY", ErrInvalidCompiledSetup)
	}
	if err := validateTranscriptField(&tauZ); err != nil || tauZ.IsZero() {
		return nil, fmt.Errorf("%w: invalid benchmark tauZ", ErrInvalidCompiledSetup)
	}
	preprocessings, err := PreprocessAllLocalPIOP(spr, partitions)
	if err != nil {
		return nil, fmt.Errorf("%w: batch adapter preprocessing: %v", ErrInvalidCompiledSetup, err)
	}
	t := preprocessings[0].LocalRows()
	if t < 4 || !isPowerOfTwo(t) || partitions > t || t > FastLocalPIOPMaxDomainSize {
		return nil, fmt.Errorf(
			"%w: require power-of-two 2 <= M <= T, 4 <= T <= %d; got M=%d,T=%d",
			ErrInvalidCompiledSetup, FastLocalPIOPMaxDomainSize, partitions, t,
		)
	}

	fastFixed := make([]*FastLocalPIOPFixed, partitions)
	placement, err := NewStatementPublicInputPlacement(preprocessings)
	if err != nil {
		return nil, fmt.Errorf("%w: public-input placement: %v", ErrInvalidCompiledSetup, err)
	}
	omega, err := sparseR1CSAdapterRoot(t)
	if err != nil {
		return nil, fmt.Errorf("%w: local domain: %v", ErrInvalidCompiledSetup, err)
	}
	indexPrefixDigest := compiledSetupIndexPrefixDigest(preprocessings)
	shift, err := DeriveIndexShift(indexPrefixDigest, uint64(t), omega)
	if err != nil {
		return nil, fmt.Errorf("%w: derive index shift: %v", ErrInvalidCompiledSetup, err)
	}

	result := &CompiledSetup{
		Parties: make([]CompiledPartySetup, partitions),
		Metadata: CompiledSetupMetadata{
			CircuitDigest:        TranscriptDigest(preprocessings[0].systemDigest),
			IndexPrefixDigest:    indexPrefixDigest,
			IndexDigest:          compiledSetupIndexDigest(indexPrefixDigest, shift),
			IndexShift:           shift,
			PartitionCount:       partitions,
			LocalDomainSize:      t,
			PublicVariableCount:  placement.PublicVariables(),
			LocalDomainGenerator: omega,
		},
	}

	fixedSlots := make([]LocalPIOPFixedCommitmentSlot, partitions)
	var fastTaylorShift *cryptodlinkzg.FastTaylorShiftPrecomputation
	for rank := 0; rank < partitions; rank++ {
		if rank == 0 {
			fastFixed[rank], err = PreprocessFastLocalPIOPFixed(
				preprocessings[rank].FixedTable(),
				preprocessings[rank].Index(),
			)
		} else {
			fastFixed[rank], err = preprocessFastLocalPIOPFixedWithDomains(
				preprocessings[rank].FixedTable(),
				preprocessings[rank].Index(),
				fastFixed[0].domain,
				fastFixed[0].extendedDomain,
				fastFixed[0].convolutionDomain,
			)
		}
		if err != nil {
			return nil, fmt.Errorf("%w: fast fixed rank %d: %v", ErrInvalidCompiledSetup, rank, err)
		}
		if rank == 0 {
			fastTaylorShift = cryptodlinkzg.NewFastTaylorShiftPrecomputation(
				t,
				shift.Sigma,
				fastFixed[0].convolutionDomain,
			)
		}
		rowSRS, srsErr := cryptodlinkzg.NewDeterministicPartyRowSRSWithShift(
			partitions, t, rank, tauY, tauZ, shift.Sigma,
		)
		if srsErr != nil {
			return nil, fmt.Errorf("%w: party SRS rank %d: %v", ErrInvalidCompiledSetup, rank, srsErr)
		}
		fixedSlots[rank], err = SetupLocalPIOPFixedCommitmentSlot(preprocessings[rank], rowSRS)
		if err != nil {
			return nil, fmt.Errorf("%w: fixed commitments rank %d: %v", ErrInvalidCompiledSetup, rank, err)
		}
		result.Parties[rank] = CompiledPartySetup{
			Rank:             rank,
			ConstraintSystem: spr,
			Preprocessing:    preprocessings[rank],
			FastFixed:        fastFixed[rank],
			FastTaylorShift:  fastTaylorShift,
			RowSRS:           rowSRS,
			FixedCommitments: fixedSlots[rank],
		}
	}

	aggregateFixed, err := AggregateLocalPIOPFixedCommitmentSlots(fixedSlots)
	if err != nil {
		return nil, fmt.Errorf("%w: aggregate fixed commitments: %v", ErrInvalidCompiledSetup, err)
	}
	coordinatorSRS, err := cryptodlinkzg.NewDeterministicCoordinatorSRSWithShift(
		partitions, tauY, tauZ, shift.Sigma,
	)
	if err != nil {
		return nil, fmt.Errorf("%w: coordinator SRS: %v", ErrInvalidCompiledSetup, err)
	}
	verifierSRS := cryptodlinkzg.NewDeterministicVerifierSRSWithShift(
		tauY, tauZ, shift.Sigma,
	)
	result.Coordinator = CompiledCoordinatorSetup{
		SRS:                  coordinatorSRS,
		FixedCommitments:     aggregateFixed,
		PublicInputPlacement: placement,
	}
	result.Verifier = CompiledVerifierSetup{
		SRS:                  verifierSRS,
		FixedCommitments:     aggregateFixed,
		PublicInputPlacement: placement,
	}

	result.Metadata.SRSDigest = compiledSetupSRSDigest(result)
	result.Metadata.PartyManifestDigest = compiledSetupManifestDigest(result)
	result.Metadata.SetupDigest = compiledSetupMetadataDigest(result.Metadata, aggregateFixed)
	for rank := range result.Parties {
		result.Parties[rank].Metadata = result.Metadata
	}
	result.Coordinator.Metadata = result.Metadata
	result.Verifier.Metadata = result.Metadata

	// Construction already produced every role from the same validated inputs.
	// Keep the factory to one SparseR1CS validation/hash pass; a transported or
	// loaded bundle must run the O(MT) Validate audit before entering the timed
	// path, whose per-proof guard is ValidateOnlineRoles.
	if err := result.ValidateOnlineRoles(); err != nil {
		return nil, err
	}
	return result, nil
}

// Validate recomputes every public binding and checks all split role shapes.
// It is an O(MT) offline/setup-time audit, suitable after key generation or
// role-key transport. Per-proof code should call ValidateOnlineRoles instead.
func (setup *CompiledSetup) Validate() error {
	if setup == nil {
		return fmt.Errorf("%w: nil bundle", ErrInvalidCompiledSetup)
	}
	m := setup.Metadata.PartitionCount
	t := setup.Metadata.LocalDomainSize
	if m < 2 || !isPowerOfTwo(m) || t < 4 || !isPowerOfTwo(t) || m > t ||
		t > FastLocalPIOPMaxDomainSize || len(setup.Parties) != m ||
		setup.Metadata.PublicVariableCount < 0 || setup.Metadata.PublicVariableCount > t {
		return fmt.Errorf("%w: malformed M/T or party count", ErrInvalidCompiledSetup)
	}
	if setup.Metadata.CircuitDigest == (TranscriptDigest{}) ||
		setup.Metadata.IndexPrefixDigest == (TranscriptDigest{}) ||
		setup.Metadata.IndexDigest == (TranscriptDigest{}) ||
		setup.Metadata.SRSDigest == (TranscriptDigest{}) ||
		setup.Metadata.PartyManifestDigest == (TranscriptDigest{}) ||
		setup.Metadata.SetupDigest == (TranscriptDigest{}) {
		return fmt.Errorf("%w: zero metadata digest", ErrInvalidCompiledSetup)
	}
	omega, err := sparseR1CSAdapterRoot(t)
	if err != nil || !omega.Equal(&setup.Metadata.LocalDomainGenerator) {
		return fmt.Errorf("%w: local-domain generator mismatch", ErrInvalidCompiledSetup)
	}
	if err := VerifyIndexShift(
		setup.Metadata.IndexPrefixDigest,
		uint64(t),
		omega,
		setup.Metadata.IndexShift.Counter,
		setup.Metadata.IndexShift.Sigma,
	); err != nil {
		return fmt.Errorf("%w: index shift: %v", ErrInvalidCompiledSetup, err)
	}
	if compiledSetupIndexDigest(setup.Metadata.IndexPrefixDigest, setup.Metadata.IndexShift) != setup.Metadata.IndexDigest {
		return fmt.Errorf("%w: full index digest", ErrCompiledSetupMetadataMismatch)
	}

	preprocessings := make([]*LocalPIOPPreprocessing, m)
	fixedSlots := make([]LocalPIOPFixedCommitmentSlot, m)
	constraintSystem := setup.Parties[0].ConstraintSystem
	if constraintSystem == nil ||
		TranscriptDigest(sparseR1CSAdapterDigest(constraintSystem)) != setup.Metadata.CircuitDigest {
		return fmt.Errorf("%w: retained constraint system", ErrCompiledSetupMetadataMismatch)
	}
	for rank := 0; rank < m; rank++ {
		party := &setup.Parties[rank]
		if party.Rank != rank || party.ConstraintSystem == nil || party.Preprocessing == nil || party.FastFixed == nil ||
			party.FastTaylorShift == nil || party.FastTaylorShift.PolynomialLength() != t || party.RowSRS == nil {
			return fmt.Errorf("%w: malformed party role at manifest rank %d", ErrInvalidCompiledSetup, rank)
		}
		if party.Metadata != setup.Metadata {
			return fmt.Errorf("%w: party rank %d", ErrCompiledSetupMetadataMismatch, rank)
		}
		if party.ConstraintSystem != constraintSystem ||
			party.Preprocessing.rank != rank || party.Preprocessing.world != m ||
			party.Preprocessing.layout != setup.Parties[0].Preprocessing.layout ||
			party.Preprocessing.layout.localRows != t ||
			party.Preprocessing.layout.publicVariables != setup.Metadata.PublicVariableCount ||
			TranscriptDigest(party.Preprocessing.systemDigest) != setup.Metadata.CircuitDigest {
			return fmt.Errorf("%w: preprocessing role at rank %d", ErrCompiledSetupMetadataMismatch, rank)
		}
		if err := validateStatementPublicInputIndex(party.Preprocessing, rank); err != nil {
			return fmt.Errorf("%w: preprocessing index rank %d: %v", ErrCompiledSetupMetadataMismatch, rank, err)
		}
		if err := validateFastLocalPIOPFixed(party.FastFixed); err != nil {
			return fmt.Errorf("%w: fast fixed rank %d: %v", ErrInvalidCompiledSetup, rank, err)
		}
		if err := party.RowSRS.Validate(); err != nil || party.RowSRS.Rank != rank ||
			party.RowSRS.Parties != m || party.RowSRS.DegreeBound != t ||
			!compiledPartySRSPointsValid(party.RowSRS) {
			return fmt.Errorf("%w: party SRS role at rank %d", ErrInvalidCompiledSetup, rank)
		}
		if err := validateLocalPIOPFixedCommitmentSlot(party.FixedCommitments); err != nil ||
			party.FixedCommitments.Rank != rank || party.FixedCommitments.Parties != m ||
			party.FixedCommitments.DomainSize != t {
			return fmt.Errorf("%w: fixed slot at rank %d", ErrInvalidCompiledSetup, rank)
		}
		preprocessings[rank] = party.Preprocessing
		fixedSlots[rank] = party.FixedCommitments
	}
	if compiledSetupIndexPrefixDigest(preprocessings) != setup.Metadata.IndexPrefixDigest {
		return fmt.Errorf("%w: adapter preprocessing identity", ErrCompiledSetupMetadataMismatch)
	}
	placement, err := NewStatementPublicInputPlacement(preprocessings)
	if err != nil {
		return fmt.Errorf("%w: public-input placement: %v", ErrInvalidCompiledSetup, err)
	}
	if setup.Coordinator.PublicInputPlacement == nil || setup.Verifier.PublicInputPlacement == nil ||
		!compiledPlacementEqual(setup.Coordinator.PublicInputPlacement, placement) ||
		!compiledPlacementEqual(setup.Verifier.PublicInputPlacement, placement) {
		return fmt.Errorf("%w: public-input placement role", ErrCompiledSetupMetadataMismatch)
	}

	if setup.Coordinator.Metadata != setup.Metadata ||
		setup.Verifier.Metadata != setup.Metadata {
		return fmt.Errorf("%w: coordinator/verifier role", ErrCompiledSetupMetadataMismatch)
	}
	if setup.Coordinator.SRS == nil || setup.Coordinator.SRS.Validate() != nil ||
		setup.Coordinator.SRS.Parties != m || !compiledCoordinatorSRSPointsValid(setup.Coordinator.SRS) {
		return fmt.Errorf("%w: coordinator SRS role", ErrInvalidCompiledSetup)
	}
	if setup.Verifier.SRS == nil || setup.Verifier.SRS.Validate() != nil ||
		!compiledVerifierSRSPointsValid(setup.Verifier.SRS) {
		return fmt.Errorf("%w: verifier SRS role", ErrInvalidCompiledSetup)
	}
	aggregateFixed, err := AggregateLocalPIOPFixedCommitmentSlots(fixedSlots)
	if err != nil {
		return fmt.Errorf("%w: aggregate fixed commitments: %v", ErrInvalidCompiledSetup, err)
	}
	if !compiledAggregateFixedEqual(aggregateFixed, setup.Coordinator.FixedCommitments) ||
		!compiledAggregateFixedEqual(aggregateFixed, setup.Verifier.FixedCommitments) {
		return fmt.Errorf("%w: aggregate fixed commitments", ErrCompiledSetupMetadataMismatch)
	}
	if compiledSetupSRSDigest(setup) != setup.Metadata.SRSDigest {
		return fmt.Errorf("%w: SRS digest", ErrCompiledSetupMetadataMismatch)
	}
	if compiledSetupManifestDigest(setup) != setup.Metadata.PartyManifestDigest {
		return fmt.Errorf("%w: party manifest digest", ErrCompiledSetupMetadataMismatch)
	}
	if compiledSetupMetadataDigest(setup.Metadata, aggregateFixed) != setup.Metadata.SetupDigest {
		return fmt.Errorf("%w: setup digest", ErrCompiledSetupMetadataMismatch)
	}
	return nil
}

// ValidateOnlineRoles performs only O(M) metadata and shape checks. It assumes
// Validate has already authenticated the setup at the offline boundary and
// deliberately does not rehash the O(MT) SRS/fixed preprocessing or rederive
// the canonical shift. Role objects must remain immutable after that boundary:
// this is a timed-path shape/tag check, not a post-validation mutation detector.
// ValidateOuterContext performs the public canonical-shift check.
func (setup *CompiledSetup) ValidateOnlineRoles() error {
	if setup == nil {
		return fmt.Errorf("%w: nil bundle", ErrInvalidCompiledSetup)
	}
	m := setup.Metadata.PartitionCount
	t := setup.Metadata.LocalDomainSize
	if m < 2 || !isPowerOfTwo(m) || t < 4 || !isPowerOfTwo(t) || m > t ||
		t > FastLocalPIOPMaxDomainSize || len(setup.Parties) != m ||
		setup.Metadata.PublicVariableCount < 0 || setup.Metadata.PublicVariableCount > t ||
		setup.Metadata.CircuitDigest == (TranscriptDigest{}) ||
		setup.Metadata.IndexPrefixDigest == (TranscriptDigest{}) ||
		setup.Metadata.IndexDigest == (TranscriptDigest{}) ||
		setup.Metadata.SRSDigest == (TranscriptDigest{}) ||
		setup.Metadata.PartyManifestDigest == (TranscriptDigest{}) ||
		setup.Metadata.SetupDigest == (TranscriptDigest{}) {
		return fmt.Errorf("%w: malformed online metadata shape", ErrInvalidCompiledSetup)
	}
	constraintSystem := setup.Parties[0].ConstraintSystem
	for rank := range setup.Parties {
		party := &setup.Parties[rank]
		if party.Rank != rank || party.Metadata != setup.Metadata ||
			party.ConstraintSystem == nil || party.ConstraintSystem != constraintSystem ||
			party.Preprocessing == nil || party.FastFixed == nil ||
			party.RowSRS == nil || party.Preprocessing.rank != rank || party.Preprocessing.world != m ||
			party.Preprocessing.layout != setup.Parties[0].Preprocessing.layout ||
			party.Preprocessing.layout.localRows != t ||
			party.Preprocessing.layout.publicVariables != setup.Metadata.PublicVariableCount ||
			TranscriptDigest(party.Preprocessing.systemDigest) != setup.Metadata.CircuitDigest ||
			party.FastFixed.domainSize != t ||
			party.RowSRS.Rank != rank || party.RowSRS.Parties != m || party.RowSRS.DegreeBound != t {
			return fmt.Errorf("%w: online party role at rank %d", ErrCompiledSetupMetadataMismatch, rank)
		}
		if err := party.RowSRS.Validate(); err != nil {
			return fmt.Errorf("%w: online party SRS rank %d", ErrInvalidCompiledSetup, rank)
		}
		if !compiledFastFixedShapeMatches(
			party.FastFixed, party.Preprocessing, t, setup.Metadata.LocalDomainGenerator,
		) {
			return fmt.Errorf("%w: online fast fixed rank %d", ErrInvalidCompiledSetup, rank)
		}
		if err := validateStatementPublicInputIndex(party.Preprocessing, rank); err != nil {
			return fmt.Errorf("%w: online preprocessing index rank %d", ErrCompiledSetupMetadataMismatch, rank)
		}
		if party.FixedCommitments.Rank != rank || party.FixedCommitments.Parties != m ||
			party.FixedCommitments.DomainSize != t ||
			!party.FixedCommitments.SlotLabel.Equal(&party.Preprocessing.index.SlotLabel) {
			return fmt.Errorf("%w: online fixed slot rank %d", ErrCompiledSetupMetadataMismatch, rank)
		}
	}
	if setup.Coordinator.Metadata != setup.Metadata || setup.Coordinator.SRS == nil ||
		setup.Coordinator.SRS.Parties != m || setup.Coordinator.SRS.Validate() != nil ||
		setup.Coordinator.PublicInputPlacement == nil ||
		setup.Coordinator.PublicInputPlacement.partitions != m ||
		setup.Coordinator.PublicInputPlacement.localRows != t ||
		setup.Coordinator.PublicInputPlacement.publicVariables != setup.Metadata.PublicVariableCount ||
		!setup.Coordinator.PublicInputPlacement.omega.Equal(&setup.Metadata.LocalDomainGenerator) ||
		setup.Coordinator.FixedCommitments.Parties != m ||
		setup.Coordinator.FixedCommitments.DomainSize != t {
		return fmt.Errorf("%w: online coordinator role", ErrCompiledSetupMetadataMismatch)
	}
	if setup.Verifier.Metadata != setup.Metadata || setup.Verifier.SRS == nil ||
		setup.Verifier.SRS.Validate() != nil || setup.Verifier.PublicInputPlacement == nil ||
		setup.Verifier.PublicInputPlacement.partitions != m ||
		setup.Verifier.PublicInputPlacement.localRows != t ||
		setup.Verifier.PublicInputPlacement.publicVariables != setup.Metadata.PublicVariableCount ||
		!setup.Verifier.PublicInputPlacement.omega.Equal(&setup.Metadata.LocalDomainGenerator) ||
		setup.Verifier.PublicInputPlacement.publicVariables != setup.Coordinator.PublicInputPlacement.publicVariables ||
		setup.Verifier.FixedCommitments.Parties != m || setup.Verifier.FixedCommitments.DomainSize != t {
		return fmt.Errorf("%w: online verifier role", ErrCompiledSetupMetadataMismatch)
	}
	if !compiledAggregateFixedEqual(setup.Coordinator.FixedCommitments, setup.Verifier.FixedCommitments) ||
		compiledSetupMetadataDigest(setup.Metadata, setup.Coordinator.FixedCommitments) != setup.Metadata.SetupDigest {
		return fmt.Errorf("%w: online aggregate fixed metadata", ErrCompiledSetupMetadataMismatch)
	}
	return nil
}

// ValidateOuterContext checks that an outer transcript uses this exact setup.
// Statement/session fields remain proof-specific and are validated by the
// ordinary outer-transcript validator.
func (setup *CompiledSetup) ValidateOuterContext(context OuterTranscriptContext) error {
	if err := setup.ValidateOnlineRoles(); err != nil {
		return err
	}
	if err := validateOuterTranscriptContext(context); err != nil {
		return err
	}
	metadata := setup.Metadata
	if context.SRSDigest != metadata.SRSDigest || context.IndexDigest != metadata.IndexDigest ||
		context.IndexPrefixDigest != metadata.IndexPrefixDigest ||
		context.PartyManifestDigest != metadata.PartyManifestDigest ||
		context.ShiftCounter != metadata.IndexShift.Counter || !context.Shift.Equal(&metadata.IndexShift.Sigma) ||
		context.PartitionCount != uint64(metadata.PartitionCount) ||
		context.LocalDomainSize != uint64(metadata.LocalDomainSize) ||
		!context.LocalDomainGenerator.Equal(&metadata.LocalDomainGenerator) {
		return ErrCompiledSetupMetadataMismatch
	}
	return nil
}

// ValidateOpeningContext checks the setup-authenticated fields shared with
// the Protocol 2 opening suffix.
func (setup *CompiledSetup) ValidateOpeningContext(context TranscriptContext) error {
	if err := setup.ValidateOnlineRoles(); err != nil {
		return err
	}
	if err := validateTranscriptContext(context); err != nil {
		return err
	}
	if context.SRSDigest != setup.Metadata.SRSDigest ||
		context.IndexDigest != setup.Metadata.IndexDigest ||
		context.PartyManifestDigest != setup.Metadata.PartyManifestDigest {
		return ErrCompiledSetupMetadataMismatch
	}
	return nil
}

func compiledSetupIndexPrefixDigest(preprocessings []*LocalPIOPPreprocessing) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledIndexPrefixDomain)
	compiledHashUint64(hasher, uint64(len(preprocessings)))
	for rank, preprocessing := range preprocessings {
		compiledHashUint64(hasher, uint64(rank))
		if preprocessing == nil {
			compiledHashString(hasher, "nil")
			continue
		}
		_, _ = hasher.Write(preprocessing.systemDigest[:])
		compiledHashUint64(hasher, uint64(preprocessing.rank))
		compiledHashUint64(hasher, uint64(preprocessing.world))
		compiledHashUint64(hasher, uint64(preprocessing.layout.localRows))
		compiledHashField(hasher, preprocessing.index.SlotLabel)
		for wire := 0; wire < LocalWireCount; wire++ {
			compiledHashField(hasher, preprocessing.index.WireCosets[wire])
		}
		fixed := preprocessing.fixed
		for selector := 0; selector < LocalSelectorCount; selector++ {
			compiledHashFields(hasher, fixed.Selectors[selector])
		}
		for wire := 0; wire < LocalWireCount; wire++ {
			compiledHashFields(hasher, fixed.SigmaX[wire])
			compiledHashFields(hasher, fixed.SigmaPart[wire])
		}
	}
	return compiledDigest(hasher)
}

func compiledSetupIndexDigest(prefix TranscriptDigest, shift IndexShift) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledIndexDomain)
	_, _ = hasher.Write(prefix[:])
	compiledHashUint64(hasher, uint64(shift.Counter))
	compiledHashField(hasher, shift.Sigma)
	return compiledDigest(hasher)
}

func compiledSetupSRSDigest(setup *CompiledSetup) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledSRSDomain)
	compiledHashUint64(hasher, uint64(len(setup.Parties)))
	for rank := range setup.Parties {
		party := setup.Parties[rank]
		compiledHashUint64(hasher, uint64(party.Rank))
		if party.RowSRS == nil {
			compiledHashString(hasher, "nil")
			continue
		}
		compiledHashUint64(hasher, uint64(party.RowSRS.Parties))
		compiledHashUint64(hasher, uint64(party.RowSRS.DegreeBound))
		compiledHashG1s(hasher, party.RowSRS.G1SemanticRow)
		compiledHashG1s(hasher, party.RowSRS.G1Row)
		compiledHashG1s(hasher, party.RowSRS.G1ZShared)
	}
	if setup.Coordinator.SRS == nil {
		compiledHashString(hasher, "nil-coordinator")
	} else {
		compiledHashUint64(hasher, uint64(setup.Coordinator.SRS.Parties))
		compiledHashG1s(hasher, setup.Coordinator.SRS.G1Y)
		compiledHashG1s(hasher, setup.Coordinator.SRS.G1ZShared)
	}
	if setup.Verifier.SRS == nil {
		compiledHashString(hasher, "nil-verifier")
	} else {
		compiledHashG1s(hasher, setup.Verifier.SRS.G1ZVerifier)
		compiledHashG2s(hasher, setup.Verifier.SRS.G2Y)
		compiledHashG2s(hasher, setup.Verifier.SRS.G2Z)
	}
	return compiledDigest(hasher)
}

func compiledSetupManifestDigest(setup *CompiledSetup) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledManifestDomain)
	_, _ = hasher.Write(setup.Metadata.CircuitDigest[:])
	_, _ = hasher.Write(setup.Metadata.IndexDigest[:])
	_, _ = hasher.Write(setup.Metadata.SRSDigest[:])
	compiledHashUint64(hasher, uint64(len(setup.Parties)))
	for rank := range setup.Parties {
		party := setup.Parties[rank]
		compiledHashUint64(hasher, uint64(rank))
		compiledHashUint64(hasher, uint64(party.Rank))
		compiledHashFastFixed(hasher, party.FastFixed)
		compiledHashFixedSlot(hasher, party.FixedCommitments)
	}
	return compiledDigest(hasher)
}

func compiledSetupMetadataDigest(metadata CompiledSetupMetadata, fixed AggregateLocalPIOPFixedCommitments) TranscriptDigest {
	hasher := sha256.New()
	compiledHashString(hasher, compiledSetupDomain)
	_, _ = hasher.Write(metadata.CircuitDigest[:])
	_, _ = hasher.Write(metadata.IndexPrefixDigest[:])
	_, _ = hasher.Write(metadata.IndexDigest[:])
	_, _ = hasher.Write(metadata.SRSDigest[:])
	_, _ = hasher.Write(metadata.PartyManifestDigest[:])
	compiledHashUint64(hasher, uint64(metadata.IndexShift.Counter))
	compiledHashField(hasher, metadata.IndexShift.Sigma)
	compiledHashUint64(hasher, uint64(metadata.PartitionCount))
	compiledHashUint64(hasher, uint64(metadata.LocalDomainSize))
	compiledHashUint64(hasher, uint64(metadata.PublicVariableCount))
	compiledHashField(hasher, metadata.LocalDomainGenerator)
	compiledHashAggregateFixed(hasher, fixed)
	return compiledDigest(hasher)
}

func compiledHashFixedSlot(hasher hash.Hash, slot LocalPIOPFixedCommitmentSlot) {
	compiledHashUint64(hasher, uint64(slot.Rank))
	compiledHashUint64(hasher, uint64(slot.Parties))
	compiledHashUint64(hasher, uint64(slot.DomainSize))
	compiledHashField(hasher, slot.SlotLabel)
	for selector := range slot.Selectors {
		compiledHashG1(hasher, slot.Selectors[selector])
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		compiledHashG1(hasher, slot.SigmaX[wire])
		compiledHashG1(hasher, slot.SigmaPart[wire])
	}
	compiledHashG1(hasher, slot.One)
	compiledHashG1(hasher, slot.X)
}

func compiledHashFastFixed(hasher hash.Hash, fixed *FastLocalPIOPFixed) {
	if fixed == nil || fixed.domain == nil {
		compiledHashString(hasher, "nil-fast-fixed")
		return
	}
	compiledHashUint64(hasher, uint64(fixed.domainSize))
	compiledHashField(hasher, fixed.domain.Generator)
	compiledHashField(hasher, fixed.index.SlotLabel)
	for wire := 0; wire < LocalWireCount; wire++ {
		compiledHashField(hasher, fixed.index.WireCosets[wire])
		compiledHashFields(hasher, fixed.sigmaX[wire])
		compiledHashFields(hasher, fixed.sigmaPart[wire])
		compiledHashFields(hasher, fixed.sigmaXEvaluations[wire])
		compiledHashFields(hasher, fixed.sigmaPartEvaluations[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		compiledHashFields(hasher, fixed.selectors[selector])
	}
	compiledHashFields(hasher, fixed.lZero)
	compiledHashFields(hasher, fixed.lStar)
}

func compiledHashAggregateFixed(hasher hash.Hash, fixed AggregateLocalPIOPFixedCommitments) {
	compiledHashUint64(hasher, uint64(fixed.Parties))
	compiledHashUint64(hasher, uint64(fixed.DomainSize))
	for selector := range fixed.Selectors {
		compiledHashG1(hasher, fixed.Selectors[selector])
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		compiledHashG1(hasher, fixed.SigmaX[wire])
		compiledHashG1(hasher, fixed.SigmaPart[wire])
	}
	compiledHashG1(hasher, fixed.Public.One)
	compiledHashG1(hasher, fixed.Public.Upsilon)
	compiledHashG1(hasher, fixed.Public.X)
}

func compiledHashString(hasher hash.Hash, value string) {
	compiledHashUint64(hasher, uint64(len(value)))
	_, _ = hasher.Write([]byte(value))
}

func compiledHashUint64(hasher hash.Hash, value uint64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], value)
	_, _ = hasher.Write(encoded[:])
}

func compiledHashField(hasher hash.Hash, value fr.Element) {
	encoded := value.Bytes()
	_, _ = hasher.Write(encoded[:])
}

func compiledHashFields(hasher hash.Hash, values []fr.Element) {
	compiledHashUint64(hasher, uint64(len(values)))
	for i := range values {
		compiledHashField(hasher, values[i])
	}
}

func compiledHashG1(hasher hash.Hash, value bn254.G1Affine) {
	encoded := value.Bytes()
	_, _ = hasher.Write(encoded[:])
}

func compiledHashG1s(hasher hash.Hash, values []bn254.G1Affine) {
	compiledHashUint64(hasher, uint64(len(values)))
	for i := range values {
		compiledHashG1(hasher, values[i])
	}
}

func compiledHashG2s(hasher hash.Hash, values []bn254.G2Affine) {
	compiledHashUint64(hasher, uint64(len(values)))
	for i := range values {
		encoded := values[i].Bytes()
		_, _ = hasher.Write(encoded[:])
	}
}

func compiledDigest(hasher hash.Hash) TranscriptDigest {
	var result TranscriptDigest
	copy(result[:], hasher.Sum(nil))
	return result
}

func compiledPartySRSPointsValid(srs *cryptodlinkzg.PartyRowSRS) bool {
	return compiledG1sValid(srs.G1SemanticRow) && compiledG1sValid(srs.G1Row) && compiledG1sValid(srs.G1ZShared)
}

func compiledCoordinatorSRSPointsValid(srs *cryptodlinkzg.CoordinatorSRS) bool {
	return compiledG1sValid(srs.G1Y) && compiledG1sValid(srs.G1ZShared)
}

func compiledVerifierSRSPointsValid(srs *cryptodlinkzg.VerifierSRS) bool {
	if !compiledG1sValid(srs.G1ZVerifier) {
		return false
	}
	for i := range srs.G2Y {
		if !srs.G2Y[i].IsOnCurve() || !srs.G2Y[i].IsInSubGroup() {
			return false
		}
	}
	for i := range srs.G2Z {
		if !srs.G2Z[i].IsOnCurve() || !srs.G2Z[i].IsInSubGroup() {
			return false
		}
	}
	return true
}

func compiledG1sValid(values []bn254.G1Affine) bool {
	for i := range values {
		if !values[i].IsOnCurve() || !values[i].IsInSubGroup() {
			return false
		}
	}
	return true
}

func compiledFastFixedShapeMatches(
	fixed *FastLocalPIOPFixed,
	preprocessing *LocalPIOPPreprocessing,
	t int,
	omega fr.Element,
) bool {
	if fixed == nil || fixed.domain == nil || preprocessing == nil || fixed.domainSize != t ||
		!fixed.domain.Generator.Equal(&omega) ||
		!fixed.index.SlotLabel.Equal(&preprocessing.index.SlotLabel) ||
		len(fixed.lZero) != t || len(fixed.lStar) != t {
		return false
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !fixed.index.WireCosets[wire].Equal(&preprocessing.index.WireCosets[wire]) ||
			len(fixed.sigmaX[wire]) != t || len(fixed.sigmaPart[wire]) != t ||
			len(fixed.sigmaXEvaluations[wire]) != t || len(fixed.sigmaPartEvaluations[wire]) != t {
			return false
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if len(fixed.selectors[selector]) != t {
			return false
		}
	}
	return true
}

func compiledElementsEqual(left, right []fr.Element) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if !left[i].Equal(&right[i]) {
			return false
		}
	}
	return true
}

func compiledAggregateFixedEqual(left, right AggregateLocalPIOPFixedCommitments) bool {
	if left.Parties != right.Parties || left.DomainSize != right.DomainSize {
		return false
	}
	for selector := range left.Selectors {
		if !left.Selectors[selector].Equal(&right.Selectors[selector]) {
			return false
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !left.SigmaX[wire].Equal(&right.SigmaX[wire]) ||
			!left.SigmaPart[wire].Equal(&right.SigmaPart[wire]) {
			return false
		}
	}
	return left.Public.One.Equal(&right.Public.One) &&
		left.Public.Upsilon.Equal(&right.Public.Upsilon) &&
		left.Public.X.Equal(&right.Public.X)
}

func compiledPlacementEqual(left, right *StatementPublicInputPlacement) bool {
	return left != nil && right != nil && left.partitions == right.partitions &&
		left.publicVariables == right.publicVariables && left.localRows == right.localRows &&
		left.omega.Equal(&right.omega)
}
