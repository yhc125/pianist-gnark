package dlinkzg

// This file is the production-oriented extraction boundary between gnark's
// BN254 SparseR1CS and the explicit row tables consumed by piop.go. It mirrors
// GPiano's row partition, public-placeholder, and padding semantics, but takes
// rank/world explicitly and never reads global MPI state.

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/internal/backend/bn254/cs"
	bn254witness "github.com/consensys/gnark/internal/backend/bn254/witness"
)

const (
	// SparseR1CSAdapterNotice makes the adapter boundary explicit. The result
	// still needs BuildLocalPIOPRelation and the transcript/PCS layers.
	SparseR1CSAdapterNotice = "BN254 SparseR1CS row-table adapter only: explicit rank/world, no MPI state, transcript, SumCheck, or PCS"

	bn254AdapterMaxTwoAdicity = 28
	bn254AdapterRootOfUnity   = "19103219067921713944291392827692070036145651957329286315305642004821462161904"
)

var (
	ErrInvalidLocalPIOPAdapterConfig  = errors.New("dlinkzg: invalid local PIOP adapter configuration")
	ErrMalformedSparseR1CS            = errors.New("dlinkzg: malformed BN254 SparseR1CS")
	ErrLocalPublicInputsDoNotFit      = errors.New("dlinkzg: public inputs do not fit in rank zero")
	ErrLocalPIOPAdapterSolve          = errors.New("dlinkzg: SparseR1CS witness solve failed")
	ErrLocalPIOPAdapterDomain         = errors.New("dlinkzg: unsupported local PIOP adapter domain")
	ErrLocalPIOPPreprocessingMismatch = errors.New("dlinkzg: local PIOP preprocessing does not match constraint system")
)

// LocalPIOPAdapterConfig supplies all process-local state explicitly. World
// is the number of row partitions and Rank is in [0,World). ProverConfig is
// forwarded to SparseR1CS.Solve so callers can supply registered hints.
type LocalPIOPAdapterConfig struct {
	Rank         int
	World        int
	ProverConfig backend.ProverConfig
}

type localPIOPAdapterLayout struct {
	publicVariables int
	constraints     int
	variables       int
	logicalRows     int
	localRows       int
	totalRows       int
}

// LocalPIOPPreprocessing owns the witness-independent rank-local table: the
// layout, index, selectors, and two permutation-coordinate families. Its
// fields are private and accessors return values or deep copies, so caller
// mutation cannot alter later online table construction.
type LocalPIOPPreprocessing struct {
	rank         int
	world        int
	layout       localPIOPAdapterLayout
	index        LocalPIOPIndex
	fixed        LocalPIOPTable
	systemDigest [sha256.Size]byte
}

// Rank returns the row-partition rank bound into preprocessing.
func (preprocessing *LocalPIOPPreprocessing) Rank() int {
	if preprocessing == nil {
		return -1
	}
	return preprocessing.rank
}

// World returns the partition count bound into preprocessing.
func (preprocessing *LocalPIOPPreprocessing) World() int {
	if preprocessing == nil {
		return 0
	}
	return preprocessing.world
}

// LocalRows returns the padded local domain size.
func (preprocessing *LocalPIOPPreprocessing) LocalRows() int {
	if preprocessing == nil {
		return 0
	}
	return preprocessing.layout.localRows
}

// LogicalRows returns the unpadded global row count.
func (preprocessing *LocalPIOPPreprocessing) LogicalRows() int {
	if preprocessing == nil {
		return 0
	}
	return preprocessing.layout.logicalRows
}

// TotalRows returns World()*LocalRows().
func (preprocessing *LocalPIOPPreprocessing) TotalRows() int {
	if preprocessing == nil {
		return 0
	}
	return preprocessing.layout.totalRows
}

// Index returns the fixed local index by value.
func (preprocessing *LocalPIOPPreprocessing) Index() LocalPIOPIndex {
	if preprocessing == nil {
		return LocalPIOPIndex{}
	}
	return preprocessing.index
}

// FixedTable returns a deep copy containing Selectors, SigmaX, and SigmaPart.
// Its witness Wires and PublicInput slices are nil by construction.
func (preprocessing *LocalPIOPPreprocessing) FixedTable() LocalPIOPTable {
	if preprocessing == nil {
		return LocalPIOPTable{}
	}
	return cloneLocalPIOPAdapterFixedTable(preprocessing.fixed)
}

// ExtractLocalPIOPTable is the compatibility wrapper that solves fullWitness
// and then extracts the natural-order LocalPIOPTable and LocalPIOPIndex for
// config.Rank. Its total allocation includes everything owned by
// SparseR1CS.Solve; use ExtractLocalPIOPTableFromSolution when solve timing and
// row-table extraction timing must be measured separately.
func ExtractLocalPIOPTable(
	spr *cs.SparseR1CS,
	fullWitness bn254witness.Witness,
	config LocalPIOPAdapterConfig,
) (table LocalPIOPTable, index LocalPIOPIndex, err error) {
	// A malformed hand-built SparseR1CS must become an API error rather than
	// an indexing panic. Valid compiled systems do not exercise this path.
	defer func() {
		if recovered := recover(); recovered != nil {
			table = LocalPIOPTable{}
			index = LocalPIOPIndex{}
			err = fmt.Errorf("%w: panic while extracting table: %v", ErrMalformedSparseR1CS, recovered)
		}
	}()

	layout, err := validateSparseR1CSAdapterInput(spr, fullWitness, config)
	if err != nil {
		return LocalPIOPTable{}, LocalPIOPIndex{}, err
	}

	solution, err := spr.Solve(fullWitness, config.ProverConfig)
	if err != nil {
		return LocalPIOPTable{}, LocalPIOPIndex{}, fmt.Errorf("%w: %v", ErrLocalPIOPAdapterSolve, err)
	}
	if len(solution) != layout.variables {
		return LocalPIOPTable{}, LocalPIOPIndex{}, fmt.Errorf(
			"%w: solver returned %d variables, want %d",
			ErrMalformedSparseR1CS, len(solution), layout.variables,
		)
	}
	return ExtractLocalPIOPTableFromSolution(spr, solution, config.Rank, config.World)
}

// ExtractLocalPIOPTableFromSolution is the convenience composition of
// PreprocessLocalPIOP and BuildLocalPIOPTableFromSolution. Use those two APIs
// separately to keep witness-independent setup out of online proving time.
func ExtractLocalPIOPTableFromSolution(
	spr *cs.SparseR1CS,
	solution []fr.Element,
	rank int,
	world int,
) (table LocalPIOPTable, index LocalPIOPIndex, err error) {
	preprocessing, err := PreprocessLocalPIOP(spr, rank, world)
	if err != nil {
		return LocalPIOPTable{}, LocalPIOPIndex{}, err
	}
	return BuildLocalPIOPTableFromSolution(preprocessing, spr, solution)
}

// PreprocessLocalPIOP constructs the complete witness-independent rank-local
// table without solving a witness. Beyond its returned fixed table it uses
// O(number-of-variables) permutation-cycle state and O(localRows) destination
// state, and never allocates GPiano's length-3*globalRows lro array.
func PreprocessLocalPIOP(
	spr *cs.SparseR1CS,
	rank int,
	world int,
) (preprocessing *LocalPIOPPreprocessing, err error) {
	// A malformed hand-built SparseR1CS must become an API error rather than
	// an indexing panic. Valid compiled systems do not exercise this path.
	defer func() {
		if recovered := recover(); recovered != nil {
			preprocessing = nil
			err = fmt.Errorf("%w: panic while preprocessing table: %v", ErrMalformedSparseR1CS, recovered)
		}
	}()

	layout, err := validateSparseR1CSAdapterStructure(spr, rank, world)
	if err != nil {
		return nil, err
	}
	localOmega, err := sparseR1CSAdapterRoot(layout.localRows)
	if err != nil {
		return nil, err
	}

	preprocessing = &LocalPIOPPreprocessing{
		rank:         rank,
		world:        world,
		layout:       layout,
		systemDigest: sparseR1CSAdapterDigest(spr),
	}
	preprocessing.index.SlotLabel.SetUint64(uint64(rank))
	preprocessing.index.WireCosets[LocalWireA].SetOne()
	preprocessing.index.WireCosets[LocalWireB].SetUint64(5)
	preprocessing.index.WireCosets[LocalWireC].Square(&preprocessing.index.WireCosets[LocalWireB])
	preprocessing.fixed = allocateLocalPIOPAdapterFixedTable(layout.localRows)
	fillLocalPIOPAdapterFixedRows(&preprocessing.fixed, spr, rank, layout)

	destinations, err := streamLocalPermutationDestinations(spr, rank, layout)
	if err != nil {
		return nil, err
	}
	fillLocalPIOPAdapterDestinations(
		&preprocessing.fixed,
		destinations,
		preprocessing.index,
		localOmega,
		world,
		layout,
	)
	return preprocessing, nil
}

// PreprocessAllLocalPIOP constructs the complete manifest-ordered fixed state
// for all ranks in one batch. SparseR1CS structure validation and hashing are
// performed once, and the global L || R || O permutation stream is scanned
// once for all ranks. The total work and retained output are O(MT); no rank
// aliases another rank's fixed-table slices.
func PreprocessAllLocalPIOP(
	spr *cs.SparseR1CS,
	world int,
) (preprocessings []*LocalPIOPPreprocessing, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			preprocessings = nil
			err = fmt.Errorf("%w: panic while batch preprocessing tables: %v", ErrMalformedSparseR1CS, recovered)
		}
	}()

	layout, err := validateSparseR1CSAdapterStructure(spr, 0, world)
	if err != nil {
		return nil, err
	}
	localOmega, err := sparseR1CSAdapterRoot(layout.localRows)
	if err != nil {
		return nil, err
	}
	systemDigest := sparseR1CSAdapterDigest(spr)
	destinations, err := streamAllLocalPermutationDestinations(spr, world, layout)
	if err != nil {
		return nil, err
	}

	preprocessings = make([]*LocalPIOPPreprocessing, world)
	for rank := 0; rank < world; rank++ {
		preprocessing := &LocalPIOPPreprocessing{
			rank:         rank,
			world:        world,
			layout:       layout,
			systemDigest: systemDigest,
			fixed:        allocateLocalPIOPAdapterFixedTable(layout.localRows),
		}
		preprocessing.index.SlotLabel.SetUint64(uint64(rank))
		preprocessing.index.WireCosets[LocalWireA].SetOne()
		preprocessing.index.WireCosets[LocalWireB].SetUint64(5)
		preprocessing.index.WireCosets[LocalWireC].Square(&preprocessing.index.WireCosets[LocalWireB])
		fillLocalPIOPAdapterFixedRows(&preprocessing.fixed, spr, rank, layout)
		fillLocalPIOPAdapterDestinations(
			&preprocessing.fixed,
			destinations[rank],
			preprocessing.index,
			localOmega,
			world,
			layout,
		)
		preprocessings[rank] = preprocessing
	}
	return preprocessings, nil
}

// BuildLocalPIOPTableFromSolution performs only the online, solution-dependent
// assembly. It deep-copies the preprocessed fixed columns and fills Wires and
// PublicInput from the read-only full solution vector. The structural digest
// rejects a different or subsequently mutated SparseR1CS.
func BuildLocalPIOPTableFromSolution(
	preprocessing *LocalPIOPPreprocessing,
	spr *cs.SparseR1CS,
	solution []fr.Element,
) (table LocalPIOPTable, index LocalPIOPIndex, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			table = LocalPIOPTable{}
			index = LocalPIOPIndex{}
			err = fmt.Errorf("%w: panic while building table from solution: %v", ErrMalformedSparseR1CS, recovered)
		}
	}()
	if preprocessing == nil {
		return LocalPIOPTable{}, LocalPIOPIndex{}, fmt.Errorf("%w: nil preprocessing", ErrLocalPIOPPreprocessingMismatch)
	}
	layout, err := validateSparseR1CSAdapterStructure(spr, preprocessing.rank, preprocessing.world)
	if err != nil {
		return LocalPIOPTable{}, LocalPIOPIndex{}, err
	}
	if layout != preprocessing.layout || sparseR1CSAdapterDigest(spr) != preprocessing.systemDigest {
		return LocalPIOPTable{}, LocalPIOPIndex{}, ErrLocalPIOPPreprocessingMismatch
	}
	if len(solution) != layout.variables {
		return LocalPIOPTable{}, LocalPIOPIndex{}, fmt.Errorf(
			"%w: solution has %d variables, want %d",
			ErrMalformedSparseR1CS, len(solution), layout.variables,
		)
	}
	table = allocateLocalPIOPAdapterTable(layout.localRows)
	copyLocalPIOPAdapterFixedTable(&table, preprocessing.fixed)
	fillLocalPIOPAdapterWitnessRows(&table, spr, solution, preprocessing.rank, layout)
	return table, preprocessing.index, nil
}

func validateSparseR1CSAdapterInput(
	spr *cs.SparseR1CS,
	fullWitness bn254witness.Witness,
	config LocalPIOPAdapterConfig,
) (localPIOPAdapterLayout, error) {
	layout, err := validateSparseR1CSAdapterStructure(spr, config.Rank, config.World)
	if err != nil {
		return layout, err
	}
	if len(fullWitness) != layout.publicVariables+spr.NbSecretVariables {
		return layout, fmt.Errorf(
			"%w: witness has %d entries, want %d public+secret entries",
			ErrMalformedSparseR1CS, len(fullWitness), layout.publicVariables+spr.NbSecretVariables,
		)
	}
	return layout, nil
}

func validateSparseR1CSAdapterStructure(
	spr *cs.SparseR1CS,
	rank int,
	world int,
) (localPIOPAdapterLayout, error) {
	var layout localPIOPAdapterLayout
	if world <= 0 || !isPowerOfTwo(world) {
		return layout, fmt.Errorf("%w: world %d is not a positive power of two", ErrInvalidLocalPIOPAdapterConfig, world)
	}
	if rank < 0 || rank >= world {
		return layout, fmt.Errorf("%w: rank %d is outside [0,%d)", ErrInvalidLocalPIOPAdapterConfig, rank, world)
	}
	if spr == nil {
		return layout, fmt.Errorf("%w: nil constraint system", ErrMalformedSparseR1CS)
	}
	if spr.CurveID() != ecc.BN254 {
		return layout, fmt.Errorf("%w: curve is %s, want BN254", ErrMalformedSparseR1CS, spr.CurveID().String())
	}
	if spr.NbPublicVariables < 0 || spr.NbSecretVariables < 0 || spr.NbInternalVariables < 0 {
		return layout, fmt.Errorf("%w: negative variable count", ErrMalformedSparseR1CS)
	}

	maxInt := int(^uint(0) >> 1)
	if spr.NbPublicVariables > maxInt-spr.NbSecretVariables ||
		spr.NbPublicVariables+spr.NbSecretVariables > maxInt-spr.NbInternalVariables {
		return layout, fmt.Errorf("%w: variable-count overflow", ErrMalformedSparseR1CS)
	}
	layout.publicVariables = spr.NbPublicVariables
	layout.constraints = len(spr.Constraints)
	layout.variables = spr.NbPublicVariables + spr.NbSecretVariables + spr.NbInternalVariables
	if layout.variables == 0 {
		return layout, fmt.Errorf("%w: empty variable vector cannot supply GPiano padding wire zero", ErrMalformedSparseR1CS)
	}
	if layout.constraints > maxInt-layout.publicVariables {
		return layout, fmt.Errorf("%w: logical-row count overflow", ErrMalformedSparseR1CS)
	}
	layout.logicalRows = layout.publicVariables + layout.constraints
	if layout.logicalRows == 0 {
		return layout, fmt.Errorf("%w: no placeholder or constraint rows", ErrMalformedSparseR1CS)
	}

	requestedLocalRows := layout.logicalRows / world
	if layout.logicalRows%world != 0 {
		requestedLocalRows++
	}
	// GPiano checks this before rounding the local domain to a power of two.
	if requestedLocalRows < layout.publicVariables {
		return layout, fmt.Errorf(
			"%w: rank-zero capacity %d is smaller than %d public placeholders",
			ErrLocalPublicInputsDoNotFit, requestedLocalRows, layout.publicVariables,
		)
	}
	if requestedLocalRows < 2 {
		return layout, fmt.Errorf("%w: GPiano local domain has fewer than two rows", ErrLocalPIOPAdapterDomain)
	}
	if requestedLocalRows > 1<<bn254AdapterMaxTwoAdicity {
		return layout, fmt.Errorf("%w: local row request %d exceeds BN254 two-adicity", ErrLocalPIOPAdapterDomain, requestedLocalRows)
	}
	layout.localRows = 1 << bits.Len(uint(requestedLocalRows-1))
	if layout.localRows > maxInt/world {
		return layout, fmt.Errorf("%w: global padded-row count overflow", ErrLocalPIOPAdapterDomain)
	}
	layout.totalRows = layout.localRows * world

	if len(spr.Coefficients) == 0 {
		return layout, fmt.Errorf("%w: empty coefficient table", ErrMalformedSparseR1CS)
	}
	for constraintIndex := range spr.Constraints {
		constraint := &spr.Constraints[constraintIndex]
		terms := []struct {
			name    string
			wireID  int
			coeffID int
		}{
			{name: "L", wireID: constraint.L.WireID(), coeffID: constraint.L.CoeffID()},
			{name: "R", wireID: constraint.R.WireID(), coeffID: constraint.R.CoeffID()},
			{name: "O", wireID: constraint.O.WireID(), coeffID: constraint.O.CoeffID()},
			{name: "M0", wireID: constraint.M[0].WireID(), coeffID: constraint.M[0].CoeffID()},
			{name: "M1", wireID: constraint.M[1].WireID(), coeffID: constraint.M[1].CoeffID()},
		}
		for _, term := range terms {
			if term.wireID < 0 || term.wireID >= layout.variables {
				return layout, fmt.Errorf(
					"%w: constraint %d term %s has wire %d outside [0,%d)",
					ErrMalformedSparseR1CS, constraintIndex, term.name, term.wireID, layout.variables,
				)
			}
			if term.coeffID < 0 || term.coeffID >= len(spr.Coefficients) {
				return layout, fmt.Errorf(
					"%w: constraint %d term %s has coefficient %d outside [0,%d)",
					ErrMalformedSparseR1CS, constraintIndex, term.name, term.coeffID, len(spr.Coefficients),
				)
			}
		}
		if constraint.K < 0 || constraint.K >= len(spr.Coefficients) {
			return layout, fmt.Errorf(
				"%w: constraint %d constant coefficient %d outside [0,%d)",
				ErrMalformedSparseR1CS, constraintIndex, constraint.K, len(spr.Coefficients),
			)
		}
		if constraint.M[0].WireID() != constraint.L.WireID() || constraint.M[1].WireID() != constraint.R.WireID() {
			return layout, fmt.Errorf(
				"%w: constraint %d multiplicative wires do not match L/R wires",
				ErrMalformedSparseR1CS, constraintIndex,
			)
		}
	}
	return layout, nil
}

// sparseR1CSAdapterDigest binds every SparseR1CS field that affects fixed
// table extraction or solution indexing. Solver scheduling and hint metadata
// are intentionally absent because BuildLocalPIOPTableFromSolution never
// invokes the solver.
func sparseR1CSAdapterDigest(spr *cs.SparseR1CS) [sha256.Size]byte {
	hasher := sha256.New()
	var encoded [8]byte
	writeUint64 := func(value uint64) {
		binary.LittleEndian.PutUint64(encoded[:], value)
		_, _ = hasher.Write(encoded[:])
	}
	writeInt := func(value int) {
		writeUint64(uint64(int64(value)))
	}

	writeInt(int(spr.CurveID()))
	writeInt(spr.NbPublicVariables)
	writeInt(spr.NbSecretVariables)
	writeInt(spr.NbInternalVariables)
	writeInt(len(spr.Coefficients))
	for coefficient := range spr.Coefficients {
		canonical := spr.Coefficients[coefficient].Bytes()
		_, _ = hasher.Write(canonical[:])
	}
	writeInt(len(spr.Constraints))
	for constraintIndex := range spr.Constraints {
		constraint := &spr.Constraints[constraintIndex]
		writeUint64(uint64(constraint.L))
		writeUint64(uint64(constraint.R))
		writeUint64(uint64(constraint.O))
		writeUint64(uint64(constraint.M[0]))
		writeUint64(uint64(constraint.M[1]))
		writeInt(constraint.K)
	}

	var digest [sha256.Size]byte
	copy(digest[:], hasher.Sum(nil))
	return digest
}

func allocateLocalPIOPAdapterTable(localRows int) LocalPIOPTable {
	var table LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		table.Wires[wire] = make([]fr.Element, localRows)
		table.SigmaX[wire] = make([]fr.Element, localRows)
		table.SigmaPart[wire] = make([]fr.Element, localRows)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		table.Selectors[selector] = make([]fr.Element, localRows)
	}
	table.PublicInput = make([]fr.Element, localRows)
	return table
}

func allocateLocalPIOPAdapterFixedTable(localRows int) LocalPIOPTable {
	var table LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		table.SigmaX[wire] = make([]fr.Element, localRows)
		table.SigmaPart[wire] = make([]fr.Element, localRows)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		table.Selectors[selector] = make([]fr.Element, localRows)
	}
	return table
}

func fillLocalPIOPAdapterFixedRows(
	table *LocalPIOPTable,
	spr *cs.SparseR1CS,
	rank int,
	layout localPIOPAdapterLayout,
) {
	for localRow := 0; localRow < layout.localRows; localRow++ {
		globalRow := rank*layout.localRows + localRow
		switch {
		case globalRow < layout.publicVariables:
			table.Selectors[LocalSelectorL][localRow].SetOne().Neg(&table.Selectors[LocalSelectorL][localRow])

		case globalRow < layout.logicalRows:
			constraint := &spr.Constraints[globalRow-layout.publicVariables]
			table.Selectors[LocalSelectorL][localRow] = spr.Coefficients[constraint.L.CoeffID()]
			table.Selectors[LocalSelectorR][localRow] = spr.Coefficients[constraint.R.CoeffID()]
			table.Selectors[LocalSelectorM][localRow].Mul(
				&spr.Coefficients[constraint.M[0].CoeffID()],
				&spr.Coefficients[constraint.M[1].CoeffID()],
			)
			table.Selectors[LocalSelectorO][localRow] = spr.Coefficients[constraint.O.CoeffID()]
			table.Selectors[LocalSelectorC][localRow] = spr.Coefficients[constraint.K]
		}
	}
}

func fillLocalPIOPAdapterWitnessRows(
	table *LocalPIOPTable,
	spr *cs.SparseR1CS,
	solution []fr.Element,
	rank int,
	layout localPIOPAdapterLayout,
) {
	paddingValue := solution[0]
	for localRow := 0; localRow < layout.localRows; localRow++ {
		globalRow := rank*layout.localRows + localRow
		switch {
		case globalRow < layout.publicVariables:
			// Only rank zero can own these rows because the fit check above
			// requires all placeholders to precede its constraint rows.
			table.Wires[LocalWireA][localRow] = solution[globalRow]
			table.Wires[LocalWireB][localRow] = paddingValue
			table.Wires[LocalWireC][localRow] = paddingValue
			table.PublicInput[localRow] = solution[globalRow]

		case globalRow < layout.logicalRows:
			constraint := &spr.Constraints[globalRow-layout.publicVariables]
			table.Wires[LocalWireA][localRow] = solution[constraint.L.WireID()]
			table.Wires[LocalWireB][localRow] = solution[constraint.R.WireID()]
			table.Wires[LocalWireC][localRow] = solution[constraint.O.WireID()]

		default:
			// GPiano assigns solution[0] to all three wires in padded rows.
			table.Wires[LocalWireA][localRow] = paddingValue
			table.Wires[LocalWireB][localRow] = paddingValue
			table.Wires[LocalWireC][localRow] = paddingValue
		}
	}
}

func cloneLocalPIOPAdapterFixedTable(source LocalPIOPTable) LocalPIOPTable {
	var result LocalPIOPTable
	for wire := 0; wire < LocalWireCount; wire++ {
		result.SigmaX[wire] = cloneElements(source.SigmaX[wire])
		result.SigmaPart[wire] = cloneElements(source.SigmaPart[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		result.Selectors[selector] = cloneElements(source.Selectors[selector])
	}
	return result
}

func copyLocalPIOPAdapterFixedTable(destination *LocalPIOPTable, source LocalPIOPTable) {
	for wire := 0; wire < LocalWireCount; wire++ {
		copy(destination.SigmaX[wire], source.SigmaX[wire])
		copy(destination.SigmaPart[wire], source.SigmaPart[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		copy(destination.Selectors[selector], source.Selectors[selector])
	}
}

// streamLocalPermutationDestinations implements GPiano's scan order over
// L || R || O. For each variable it keeps only the last global position; for
// the requested rank it records the previous occurrence, and closes first
// occurrences to the final one after the scan.
func streamLocalPermutationDestinations(
	spr *cs.SparseR1CS,
	rank int,
	layout localPIOPAdapterLayout,
) ([LocalWireCount][]int64, error) {
	var destinations [LocalWireCount][]int64
	for wire := 0; wire < LocalWireCount; wire++ {
		destinations[wire] = make([]int64, layout.localRows)
		for row := range destinations[wire] {
			destinations[wire][row] = -1
		}
	}

	lastPosition := make([]int64, layout.variables)
	for variable := range lastPosition {
		lastPosition[variable] = -1
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		for globalRow := 0; globalRow < layout.totalRows; globalRow++ {
			variable := sparseR1CSAdapterVariableAt(spr, wire, globalRow, layout)
			position := int64(wire)*int64(layout.totalRows) + int64(globalRow)
			if globalRow/layout.localRows == rank && lastPosition[variable] != -1 {
				destinations[wire][globalRow%layout.localRows] = lastPosition[variable]
			}
			lastPosition[variable] = position
		}
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		for localRow := 0; localRow < layout.localRows; localRow++ {
			if destinations[wire][localRow] != -1 {
				continue
			}
			globalRow := rank*layout.localRows + localRow
			variable := sparseR1CSAdapterVariableAt(spr, wire, globalRow, layout)
			if lastPosition[variable] < 0 {
				return destinations, fmt.Errorf(
					"%w: variable %d has no permutation occurrence",
					ErrMalformedSparseR1CS, variable,
				)
			}
			destinations[wire][localRow] = lastPosition[variable]
		}
	}
	return destinations, nil
}

// streamAllLocalPermutationDestinations is the batched counterpart of
// streamLocalPermutationDestinations. It records each occurrence's predecessor
// during one global scan, then closes every first occurrence to its variable's
// final position in one linear pass over the manifest-ordered output.
func streamAllLocalPermutationDestinations(
	spr *cs.SparseR1CS,
	world int,
	layout localPIOPAdapterLayout,
) ([][LocalWireCount][]int64, error) {
	destinations := make([][LocalWireCount][]int64, world)
	for rank := 0; rank < world; rank++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			destinations[rank][wire] = make([]int64, layout.localRows)
			for row := range destinations[rank][wire] {
				destinations[rank][wire][row] = -1
			}
		}
	}

	lastPosition := make([]int64, layout.variables)
	for variable := range lastPosition {
		lastPosition[variable] = -1
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		for globalRow := 0; globalRow < layout.totalRows; globalRow++ {
			variable := sparseR1CSAdapterVariableAt(spr, wire, globalRow, layout)
			position := int64(wire)*int64(layout.totalRows) + int64(globalRow)
			if lastPosition[variable] != -1 {
				rank := globalRow / layout.localRows
				row := globalRow % layout.localRows
				destinations[rank][wire][row] = lastPosition[variable]
			}
			lastPosition[variable] = position
		}
	}

	for rank := 0; rank < world; rank++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			for row := 0; row < layout.localRows; row++ {
				if destinations[rank][wire][row] != -1 {
					continue
				}
				globalRow := rank*layout.localRows + row
				variable := sparseR1CSAdapterVariableAt(spr, wire, globalRow, layout)
				if lastPosition[variable] < 0 {
					return nil, fmt.Errorf(
						"%w: variable %d has no permutation occurrence",
						ErrMalformedSparseR1CS, variable,
					)
				}
				destinations[rank][wire][row] = lastPosition[variable]
			}
		}
	}
	return destinations, nil
}

func sparseR1CSAdapterVariableAt(
	spr *cs.SparseR1CS,
	wire int,
	globalRow int,
	layout localPIOPAdapterLayout,
) int {
	if wire == int(LocalWireA) && globalRow < layout.publicVariables {
		return globalRow
	}
	if globalRow >= layout.publicVariables && globalRow < layout.logicalRows {
		constraint := &spr.Constraints[globalRow-layout.publicVariables]
		switch LocalWire(wire) {
		case LocalWireA:
			return constraint.L.WireID()
		case LocalWireB:
			return constraint.R.WireID()
		case LocalWireC:
			return constraint.O.WireID()
		}
	}
	// GPiano's zero-initialized lro array assigns every unused placeholder
	// R/O entry and every padded entry to variable zero.
	return 0
}

func fillLocalPIOPAdapterDestinations(
	table *LocalPIOPTable,
	destinations [LocalWireCount][]int64,
	index LocalPIOPIndex,
	localOmega fr.Element,
	world int,
	layout localPIOPAdapterLayout,
) {
	localPoints := make([]fr.Element, layout.localRows)
	localPoints[0].SetOne()
	for row := 1; row < layout.localRows; row++ {
		localPoints[row].Mul(&localPoints[row-1], &localOmega)
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		for row := 0; row < layout.localRows; row++ {
			destination := destinations[wire][row]
			destinationWire := int(destination / int64(layout.totalRows))
			withinWire := int(destination % int64(layout.totalRows))
			destinationRank := withinWire / layout.localRows
			destinationRow := withinWire % layout.localRows
			if destinationWire < 0 || destinationWire >= LocalWireCount ||
				destinationRank < 0 || destinationRank >= world ||
				destinationRow < 0 || destinationRow >= layout.localRows {
				panic("validated permutation destination is out of range")
			}

			table.SigmaPart[wire][row].SetUint64(uint64(destinationRank))
			table.SigmaX[wire][row].Mul(&index.WireCosets[destinationWire], &localPoints[destinationRow])
		}
	}
}

func sparseR1CSAdapterRoot(size int) (fr.Element, error) {
	if size <= 0 || !isPowerOfTwo(size) || size > 1<<bn254AdapterMaxTwoAdicity {
		return fr.Element{}, fmt.Errorf("%w: size %d is outside the BN254 two-adic domain", ErrLocalPIOPAdapterDomain, size)
	}
	var largestRoot fr.Element
	if _, err := largestRoot.SetString(bn254AdapterRootOfUnity); err != nil {
		return fr.Element{}, fmt.Errorf("%w: parse BN254 root: %v", ErrLocalPIOPAdapterDomain, err)
	}
	logSize := bits.TrailingZeros(uint(size))
	exponent := uint64(1) << uint(bn254AdapterMaxTwoAdicity-logSize)
	var root fr.Element
	root.Exp(largestRoot, new(big.Int).SetUint64(exponent))
	return root, nil
}

func sparseR1CSAdapterPower(base fr.Element, exponent int) fr.Element {
	var result fr.Element
	result.Exp(base, new(big.Int).SetUint64(uint64(exponent)))
	return result
}
