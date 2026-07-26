package dlinkzg

// This file implements the pure outer PIOP reduction of Protocol 1. It is an
// honest-coordinator/prover intermediate, not a succinct verifier statement:
// M-length tables exist only inside the coordinator state. The verifier-facing
// API consumes the 21 circuit folds and five authenticated tree evaluations.

import (
	"errors"
	"fmt"
	"math/bits"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

const (
	// OuterPIOPReductionNotice fixes the scope of this slice. Full alpha
	// exclusions for translated PCS queries, Fiat--Shamir timing/rejection,
	// MPI, commitments, and claim authentication are deliberately external.
	OuterPIOPReductionNotice = "honest-coordinator outer PIOP intermediate: clear M-row tables are not a verifier statement; no full PCS alpha exclusion, Fiat-Shamir, MPI, commitments, or authentication"

	outerOracleEquality      = 0
	outerOracleTerminalStart = 1
	outerOracleOmegaZ        = outerOracleTerminalStart + LocalTerminalAlphaCount
	outerOracleXStarStart    = outerOracleOmegaZ + LocalTerminalOmegaAlphaCount
	outerOracleSlotLabel     = outerOracleXStarStart + LocalTerminalXStarCount
	outerOraclePublicInput   = outerOracleSlotLabel + 1
	outerOracleTreeLeaves    = outerOraclePublicInput + 1
	outerOracleTreeParents   = outerOracleTreeLeaves + 1
	outerOracleTreeEven      = outerOracleTreeParents + 1
	outerOracleTreeOdd       = outerOracleTreeEven + 1
	outerOracleCount         = outerOracleTreeOdd + 1
)

var (
	ErrInvalidOuterPIOPRecord    = errors.New("dlinkzg: invalid outer PIOP local record")
	ErrInvalidOuterPIOPShape     = errors.New("dlinkzg: invalid outer PIOP reduction shape")
	ErrOuterPIOPLocalResidual    = errors.New("dlinkzg: nonzero outer PIOP local residual")
	ErrOuterPIOPProductRoot      = errors.New("dlinkzg: outer PIOP ProductCheck root is not one")
	ErrOuterPIOPSumCheckInstance = errors.New("dlinkzg: invalid outer PIOP SumCheck instance")
	ErrOuterPIOPEndpoint         = errors.New("dlinkzg: invalid outer PIOP endpoint")
)

// OuterPIOPLocalRecord is a full-relation convenience record for tests and
// local composition. The production reduction receives only its canonical
// 13/1/7 Terminals through BuildOuterPIOPCoordinatorStateFromTerminals.
type OuterPIOPLocalRecord struct {
	Relation  *LocalPIOPRelation
	Terminals LocalTerminalEvaluations
}

// OuterPIOPReductionContext is the common public/index state for a W3 batch.
// The partition label is not supplied: rank i always uses the canonical field
// embedding i_F.
type OuterPIOPReductionContext struct {
	DomainSize int
	Omega      fr.Element
	XStar      fr.Element
	Alpha      fr.Element
	WireCosets [LocalWireCount]fr.Element
	Challenges LocalPIOPChallenges
}

// NewOuterPIOPLocalRecord constructs and checks one canonical record.
func NewOuterPIOPLocalRecord(relation *LocalPIOPRelation, alpha fr.Element) (OuterPIOPLocalRecord, error) {
	if relation == nil {
		return OuterPIOPLocalRecord{}, fmt.Errorf("%w: nil local relation", ErrInvalidOuterPIOPRecord)
	}
	if err := VerifyLocalPIOPRelation(relation); err != nil {
		return OuterPIOPLocalRecord{}, fmt.Errorf("%w: %v", ErrInvalidOuterPIOPRecord, err)
	}
	terminals, err := relation.TerminalEvaluations(alpha)
	if err != nil {
		return OuterPIOPLocalRecord{}, fmt.Errorf("%w: %v", ErrInvalidOuterPIOPRecord, err)
	}
	return OuterPIOPLocalRecord{Relation: relation, Terminals: terminals}, nil
}

// OuterPIOPCoordinatorState owns the clear M-row witness used to construct
// ProductCheck and SumCheck. Its fields are deliberately private so this type
// cannot be confused with the public verifier input.
type OuterPIOPCoordinatorState struct {
	m          int
	t          int
	alpha      fr.Element
	omega      fr.Element
	xStar      fr.Element
	challenges LocalPIOPChallenges
	wireCosets [LocalWireCount]fr.Element

	localResiduals  []fr.Element
	terminalTables  [LocalTerminalTotalCount][]fr.Element
	partitionLabels []fr.Element
	publicInputs    []fr.Element
	productTable    []fr.Element
	t0              []fr.Element
	t1              []fr.Element
}

// Notice identifies this value as an honest-coordinator intermediate.
func (state *OuterPIOPCoordinatorState) Notice() string {
	return OuterPIOPReductionNotice
}

// LocalResiduals returns a diagnostic copy of the Boolean R0 values. This is
// coordinator data and is not accepted by the verifier as a public claim.
func (state *OuterPIOPCoordinatorState) LocalResiduals() []fr.Element {
	if state == nil {
		return nil
	}
	return cloneElements(state.localResiduals)
}

// CoordinatorProductCheckWitness returns copies of t,t0,t1 for commitment by
// the honest coordinator. A verifier must authenticate the resulting five
// functionals rather than receive these clear arrays.
func (state *OuterPIOPCoordinatorState) CoordinatorProductCheckWitness() (table, t0, t1 []fr.Element) {
	if state == nil {
		return nil, nil, nil
	}
	return cloneElements(state.productTable), cloneElements(state.t0), cloneElements(state.t1)
}

// CoordinatorPublicInputAt folds the statement-derived PI(alpha) table. In a
// compiled verifier this scalar is recomputed directly from the public input
// placement map; it is not a prover-supplied terminal claim.
func (state *OuterPIOPCoordinatorState) CoordinatorPublicInputAt(point []fr.Element) (fr.Element, error) {
	if state == nil {
		return fr.Element{}, fmt.Errorf("%w: nil coordinator state", ErrInvalidOuterPIOPShape)
	}
	return EvaluateMultilinear(state.publicInputs, point)
}

// BuildOuterPIOPCoordinatorState is a convenience wrapper for tests and local
// composition. It validates full local relations, converts them to their W3
// messages, and delegates to BuildOuterPIOPCoordinatorStateFromTerminals. The
// distributed coordinator does not receive these full relations.
func BuildOuterPIOPCoordinatorState(records []OuterPIOPLocalRecord, alpha fr.Element) (*OuterPIOPCoordinatorState, error) {
	m := len(records)
	if m < 2 || !isPowerOfTwo(m) {
		return nil, fmt.Errorf("%w: record count M=%d is not a power of two at least two", ErrInvalidOuterPIOPShape, m)
	}
	if records[0].Relation == nil {
		return nil, fmt.Errorf("%w: rank 0 has a nil relation", ErrInvalidOuterPIOPRecord)
	}

	first := records[0].Relation
	context := OuterPIOPReductionContext{
		DomainSize: first.DomainSize,
		Omega:      first.Omega,
		XStar:      first.XStar,
		Alpha:      alpha,
		WireCosets: first.Index.WireCosets,
		Challenges: first.Challenges,
	}
	terminals := make([]LocalTerminalEvaluations, m)
	publicInputAtAlpha := make([]fr.Element, m)
	for rank := range records {
		relation := records[rank].Relation
		if relation == nil {
			return nil, fmt.Errorf("%w: rank %d has a nil relation", ErrInvalidOuterPIOPRecord, rank)
		}
		if err := VerifyLocalPIOPRelation(relation); err != nil {
			return nil, fmt.Errorf("%w: rank %d: %v", ErrInvalidOuterPIOPRecord, rank, err)
		}
		if relation.DomainSize != context.DomainSize ||
			!relation.Omega.Equal(&context.Omega) ||
			!relation.XStar.Equal(&context.XStar) ||
			!equalOuterPIOPChallenges(relation.Challenges, context.Challenges) ||
			!equalOuterPIOPWireCosets(relation.Index.WireCosets, context.WireCosets) {
			return nil, fmt.Errorf("%w: rank %d uses inconsistent domain, challenges, or wire cosets", ErrInvalidOuterPIOPShape, rank)
		}
		wantSlot := fr.NewElement(uint64(rank))
		if !relation.Index.SlotLabel.Equal(&wantSlot) {
			return nil, fmt.Errorf("%w: rank %d slot label is not i_F", ErrInvalidOuterPIOPShape, rank)
		}

		expectedTerminals, err := relation.TerminalEvaluations(alpha)
		if err != nil {
			return nil, fmt.Errorf("%w: rank %d: %v", ErrInvalidOuterPIOPRecord, rank, err)
		}
		if !equalOuterPIOPTerminals(expectedTerminals, records[rank].Terminals) {
			return nil, fmt.Errorf("%w: rank %d terminal record differs from its relation", ErrInvalidOuterPIOPRecord, rank)
		}
		terminals[rank] = records[rank].Terminals
		publicInputAtAlpha[rank] = evaluateLocalPolynomial(relation.PublicInput, alpha)
	}
	return BuildOuterPIOPCoordinatorStateFromTerminals(terminals, publicInputAtAlpha, context)
}

// BuildOuterPIOPCoordinatorStateFromTerminals is the production outer-PIOP
// boundary. terminals is exactly the rank-ordered 21-field W3 payload;
// publicInputAtAlpha is separate statement-derived data, never part of W3.
// The coordinator computes Delta_i(alpha), derives rho_i=u_i/v_i with an
// explicit v_i!=0 check, and constructs ProductCheck. Alpha is checked only
// against Omega_X union {0}; translated-PCS exclusions belong to the
// transcript/PCS layer.
func BuildOuterPIOPCoordinatorStateFromTerminals(
	terminals []LocalTerminalEvaluations,
	publicInputAtAlpha []fr.Element,
	context OuterPIOPReductionContext,
) (*OuterPIOPCoordinatorState, error) {
	m := len(terminals)
	if err := validateOuterPIOPReductionContext(m, context); err != nil {
		return nil, err
	}
	if len(publicInputAtAlpha) != m {
		return nil, fmt.Errorf(
			"%w: public-input vector has length %d, want M=%d",
			ErrInvalidOuterPIOPShape, len(publicInputAtAlpha), m,
		)
	}
	state := &OuterPIOPCoordinatorState{
		m:               m,
		t:               context.DomainSize,
		alpha:           context.Alpha,
		omega:           context.Omega,
		xStar:           context.XStar,
		challenges:      context.Challenges,
		wireCosets:      context.WireCosets,
		localResiduals:  make([]fr.Element, m),
		partitionLabels: make([]fr.Element, m),
		publicInputs:    make([]fr.Element, m),
	}
	for terminal := range state.terminalTables {
		state.terminalTables[terminal] = make([]fr.Element, m)
	}

	factors := make([]fr.Element, m)
	for rank := range terminals {
		state.partitionLabels[rank].SetUint64(uint64(rank))
		state.publicInputs[rank] = publicInputAtAlpha[rank]
		fXStar := multiplyThreeLocalScalars(
			terminals[rank].XStar[XStarPhiA],
			terminals[rank].XStar[XStarPhiB],
			terminals[rank].XStar[XStarPhiC],
		)
		fPrimeXStar := multiplyThreeLocalScalars(
			terminals[rank].XStar[XStarPhiPrimeA],
			terminals[rank].XStar[XStarPhiPrimeB],
			terminals[rank].XStar[XStarPhiPrimeC],
		)
		if fPrimeXStar.IsZero() {
			return nil, fmt.Errorf("%w: rank %d has v_i=0", ErrInvalidOuterPIOPRecord, rank)
		}
		var numerator fr.Element
		numerator.Mul(&terminals[rank].XStar[XStarZ], &fXStar)
		factors[rank].Div(&numerator, &fPrimeXStar)

		context := LocalQuotientEvaluationContext{
			DomainSize: state.t,
			Omega:      state.omega,
			XStar:      state.xStar,
			Index: LocalPIOPIndex{
				SlotLabel:  state.partitionLabels[rank],
				WireCosets: state.wireCosets,
			},
			Challenges:         state.challenges,
			PublicInputAtAlpha: state.publicInputs[rank],
		}
		var err error
		state.localResiduals[rank], err = EvaluateLocalQuotientResidual(terminals[rank], state.alpha, context)
		if err != nil {
			return nil, fmt.Errorf("%w: rank %d: %v", ErrInvalidOuterPIOPRecord, rank, err)
		}
		if !state.localResiduals[rank].IsZero() {
			return nil, fmt.Errorf("%w at rank %d", ErrOuterPIOPLocalResidual, rank)
		}
		flattened := terminals[rank].Flatten()
		for terminal := range flattened {
			state.terminalTables[terminal][rank] = flattened[terminal]
		}
	}

	var err error
	state.productTable, err = BuildProductCheckTable(factors)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidOuterPIOPShape, err)
	}
	if err := VerifyProductCheckTable(state.productTable); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrOuterPIOPProductRoot, err)
	}
	state.t0, state.t1, err = SplitProductCheckTable(state.productTable)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidOuterPIOPShape, err)
	}
	return state, nil
}

func validateOuterPIOPReductionContext(m int, context OuterPIOPReductionContext) error {
	if m < 2 || !isPowerOfTwo(m) {
		return fmt.Errorf("%w: terminal count M=%d is not a power of two at least two", ErrInvalidOuterPIOPShape, m)
	}
	logM := bits.Len(uint(m)) - 1
	_, err := validateOuterPIOPEndpointContext(OuterPIOPEndpointContext{
		M:          m,
		T:          context.DomainSize,
		Omega:      context.Omega,
		XStar:      context.XStar,
		Alpha:      context.Alpha,
		Theta:      make([]fr.Element, logM),
		Point:      make([]fr.Element, logM),
		WireCosets: context.WireCosets,
		Challenges: context.Challenges,
	})
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidOuterPIOPShape, err)
	}
	return nil
}

// OuterPIOPSumCheckInstance owns the exact degree-five oracle composition.
// Clear tables stay private and are used only by the honest SumCheck prover.
type OuterPIOPSumCheckInstance struct {
	state         *OuterPIOPCoordinatorState
	zeta          fr.Element
	theta         []fr.Element
	oracleTables  [][]fr.Element
	composition   SumCheckComposition
	scalarContext outerPIOPScalarContext
}

// Notice identifies this object as an honest-prover intermediate.
func (instance *OuterPIOPSumCheckInstance) Notice() string {
	return OuterPIOPReductionNotice
}

// CoordinatorOracleTables returns diagnostic copies and the degree-five
// composition. It exists for honest-prover testing/instrumentation only; a
// public verifier must never receive these M-length arrays.
func (instance *OuterPIOPSumCheckInstance) CoordinatorOracleTables() ([][]fr.Element, SumCheckComposition) {
	if instance == nil {
		return nil, nil
	}
	tables := make([][]fr.Element, len(instance.oracleTables))
	for oracle := range tables {
		tables[oracle] = cloneElements(instance.oracleTables[oracle])
	}
	return tables, instance.composition
}

// BuildSumCheckInstance constructs eq(theta,z)*(R0+zeta*R1+zeta^2*R2)
// from the 21 circuit tables, public slot/PI tables, and four tree views.
func (state *OuterPIOPCoordinatorState) BuildSumCheckInstance(zeta fr.Element, theta []fr.Element) (*OuterPIOPSumCheckInstance, error) {
	if state == nil {
		return nil, fmt.Errorf("%w: nil coordinator state", ErrOuterPIOPSumCheckInstance)
	}
	logM := bits.Len(uint(state.m)) - 1
	if len(theta) != logM {
		return nil, fmt.Errorf("%w: theta has %d coordinates, want %d", ErrOuterPIOPSumCheckInstance, len(theta), logM)
	}
	equality, err := EqualityEvaluationTable(theta)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrOuterPIOPSumCheckInstance, err)
	}

	instance := &OuterPIOPSumCheckInstance{
		state:        state,
		zeta:         zeta,
		theta:        cloneElements(theta),
		oracleTables: make([][]fr.Element, outerOracleCount),
	}
	instance.oracleTables[outerOracleEquality] = equality
	for terminal := 0; terminal < LocalTerminalTotalCount; terminal++ {
		instance.oracleTables[outerOracleTerminalStart+terminal] = cloneElements(state.terminalTables[terminal])
	}
	instance.oracleTables[outerOracleSlotLabel] = cloneElements(state.partitionLabels)
	instance.oracleTables[outerOraclePublicInput] = cloneElements(state.publicInputs)
	instance.oracleTables[outerOracleTreeLeaves] = cloneElements(state.productTable[:state.m])
	instance.oracleTables[outerOracleTreeParents] = cloneElements(state.productTable[state.m:])
	instance.oracleTables[outerOracleTreeEven] = make([]fr.Element, state.m)
	instance.oracleTables[outerOracleTreeOdd] = make([]fr.Element, state.m)
	for row := 0; row < state.m; row++ {
		instance.oracleTables[outerOracleTreeEven][row] = state.productTable[2*row]
		instance.oracleTables[outerOracleTreeOdd][row] = state.productTable[2*row+1]
	}

	instance.scalarContext, err = newOuterPIOPScalarContext(state, zeta)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrOuterPIOPSumCheckInstance, err)
	}
	instance.composition = func(values []fr.Element) fr.Element {
		terminals := outerPIOPTerminalsFromOracleValues(values)
		tree := [4]fr.Element{
			values[outerOracleTreeLeaves],
			values[outerOracleTreeParents],
			values[outerOracleTreeEven],
			values[outerOracleTreeOdd],
		}
		residuals := evaluateOuterPIOPResiduals(
			terminals,
			values[outerOracleSlotLabel],
			values[outerOraclePublicInput],
			tree,
			instance.scalarContext,
		)
		var combined, term fr.Element
		combined = residuals.R0
		term.Mul(&instance.zeta, &residuals.R1)
		combined.Add(&combined, &term)
		term.Mul(&instance.scalarContext.zetaSquared, &residuals.R2)
		combined.Add(&combined, &term)
		combined.Mul(&combined, &values[outerOracleEquality])
		return combined
	}

	values := make([]fr.Element, outerOracleCount)
	for row := 0; row < state.m; row++ {
		for oracle := range instance.oracleTables {
			values[oracle] = instance.oracleTables[oracle][row]
		}
		terminals := outerPIOPTerminalsFromOracleValues(values)
		tree := [4]fr.Element{
			values[outerOracleTreeLeaves], values[outerOracleTreeParents],
			values[outerOracleTreeEven], values[outerOracleTreeOdd],
		}
		residuals := evaluateOuterPIOPResiduals(
			terminals,
			values[outerOracleSlotLabel],
			values[outerOraclePublicInput],
			tree,
			instance.scalarContext,
		)
		if !residuals.R0.Equal(&state.localResiduals[row]) || !residuals.R1.IsZero() || !residuals.R2.IsZero() {
			return nil, fmt.Errorf("%w: Boolean residual mismatch at rank %d", ErrOuterPIOPSumCheckInstance, row)
		}
	}
	claim := sumCompositionTable(instance.oracleTables, instance.composition)
	if !claim.IsZero() {
		return nil, fmt.Errorf("%w: declared zero claim is nonzero", ErrOuterPIOPSumCheckInstance)
	}
	return instance, nil
}

// OuterPIOPSumCheckProof contains only the transcript and terminal values
// that the compiled verifier later authenticates. No M-length table appears.
type OuterPIOPSumCheckProof struct {
	Transcript      SumCheckTranscript
	FoldedCircuit   LocalTerminalEvaluations
	TreeEvaluations [terminalTreeClaims]fr.Element
}

// Prove runs the degree-five SumCheck and emits the 21+5 terminal inventory.
// Round challenges are caller-supplied; transcript derivation and rejection of
// 0/1 challenges are outside this pure slice.
func (instance *OuterPIOPSumCheckInstance) Prove(roundChallenges []fr.Element) (OuterPIOPSumCheckProof, error) {
	var proof OuterPIOPSumCheckProof
	if instance == nil || instance.state == nil || instance.composition == nil {
		return proof, fmt.Errorf("%w: nil instance", ErrOuterPIOPSumCheckInstance)
	}
	logM := bits.Len(uint(instance.state.m)) - 1
	if len(roundChallenges) != logM {
		return proof, fmt.Errorf("%w: got %d round challenges, want %d", ErrOuterPIOPSumCheckInstance, len(roundChallenges), logM)
	}

	var zero fr.Element
	transcript, finalEvaluations, err := ProveDegreeFiveSumCheck(
		zero,
		instance.oracleTables,
		instance.composition,
		roundChallenges,
	)
	if err != nil {
		return proof, fmt.Errorf("%w: %v", ErrOuterPIOPSumCheckInstance, err)
	}
	proof.Transcript = transcript
	proof.FoldedCircuit = outerPIOPTerminalsFromOracleValues(finalEvaluations)

	treePoints := productCheckFunctionalPoints(roundChallenges)
	for claim := 0; claim < terminalTreeClaims; claim++ {
		proof.TreeEvaluations[claim], err = evaluateTreeFunctional(instance.state.t0, instance.state.t1, treePoints[claim])
		if err != nil {
			return OuterPIOPSumCheckProof{}, fmt.Errorf("%w: tree claim %d: %v", ErrOuterPIOPSumCheckInstance, claim, err)
		}
	}
	if !proof.TreeEvaluations[4].IsOne() {
		return OuterPIOPSumCheckProof{}, ErrOuterPIOPProductRoot
	}

	structuredSlot := EvaluatePartitionLabelMLE(roundChallenges)
	if !structuredSlot.Equal(&finalEvaluations[outerOracleSlotLabel]) {
		return OuterPIOPSumCheckProof{}, fmt.Errorf("%w: folded slot label is not the structured i_F MLE", ErrOuterPIOPSumCheckInstance)
	}
	context := instance.EndpointContext(roundChallenges)
	endpoint, err := ReconstructOuterPIOPEndpoint(
		proof.FoldedCircuit,
		proof.TreeEvaluations,
		finalEvaluations[outerOraclePublicInput],
		context,
	)
	if err != nil {
		return OuterPIOPSumCheckProof{}, err
	}
	compositionEndpoint := instance.composition(finalEvaluations)
	if !endpoint.Value.Equal(&compositionEndpoint) {
		return OuterPIOPSumCheckProof{}, fmt.Errorf("%w: terminal reconstruction differs from oracle composition", ErrOuterPIOPEndpoint)
	}
	if err := VerifyDegreeFiveSumCheck(zero, roundChallenges, proof.Transcript, endpoint.Value); err != nil {
		return OuterPIOPSumCheckProof{}, fmt.Errorf("%w: %v", ErrOuterPIOPEndpoint, err)
	}
	return proof, nil
}

// OuterPIOPEndpointContext is public verifier state. PublicInputAtR is kept as
// a separate argument to reconstruction because it must be derived from the
// statement, not accepted as a prover claim.
type OuterPIOPEndpointContext struct {
	M          int
	T          int
	Omega      fr.Element
	XStar      fr.Element
	Alpha      fr.Element
	Zeta       fr.Element
	Theta      []fr.Element
	Point      []fr.Element
	WireCosets [LocalWireCount]fr.Element
	Challenges LocalPIOPChallenges
}

// EndpointContext returns a public context for the supplied final point.
func (instance *OuterPIOPSumCheckInstance) EndpointContext(point []fr.Element) OuterPIOPEndpointContext {
	if instance == nil || instance.state == nil {
		return OuterPIOPEndpointContext{}
	}
	return OuterPIOPEndpointContext{
		M:          instance.state.m,
		T:          instance.state.t,
		Omega:      instance.state.omega,
		XStar:      instance.state.xStar,
		Alpha:      instance.state.alpha,
		Zeta:       instance.zeta,
		Theta:      cloneElements(instance.theta),
		Point:      cloneElements(point),
		WireCosets: instance.state.wireCosets,
		Challenges: instance.state.challenges,
	}
}

// OuterPIOPEndpointEvaluation exposes the three residuals and their equality-
// weighted combination for precise verifier and test diagnostics.
type OuterPIOPEndpointEvaluation struct {
	R0       fr.Element
	R1       fr.Element
	R2       fr.Element
	Equality fr.Element
	Value    fr.Element
}

// ReconstructOuterPIOPEndpoint implements Equations (endpoint-residuals) and
// (endpoint-sumcheck-check) from 21 circuit folds, five tree claims, and the
// verifier-derived public-input fold.
func ReconstructOuterPIOPEndpoint(
	folded LocalTerminalEvaluations,
	tree [terminalTreeClaims]fr.Element,
	publicInputAtR fr.Element,
	context OuterPIOPEndpointContext,
) (OuterPIOPEndpointEvaluation, error) {
	var result OuterPIOPEndpointEvaluation
	scalarContext, err := validateOuterPIOPEndpointContext(context)
	if err != nil {
		return result, err
	}
	if !tree[4].IsOne() {
		return result, ErrOuterPIOPProductRoot
	}
	partitionLabel := EvaluatePartitionLabelMLE(context.Point)
	residuals := evaluateOuterPIOPResiduals(
		folded,
		partitionLabel,
		publicInputAtR,
		[4]fr.Element{tree[0], tree[1], tree[2], tree[3]},
		scalarContext,
	)
	result.R0 = residuals.R0
	result.R1 = residuals.R1
	result.R2 = residuals.R2
	result.Equality = evaluateOuterEquality(context.Theta, context.Point)

	var term fr.Element
	result.Value = result.R0
	term.Mul(&context.Zeta, &result.R1)
	result.Value.Add(&result.Value, &term)
	term.Mul(&scalarContext.zetaSquared, &result.R2)
	result.Value.Add(&result.Value, &term)
	result.Value.Mul(&result.Value, &result.Equality)
	return result, nil
}

// VerifyOuterPIOPSumCheck verifies the round transcript against the endpoint.
// Soundness additionally requires PCS authentication of FoldedCircuit and
// TreeEvaluations and independent derivation of publicInputAtR.
func VerifyOuterPIOPSumCheck(
	proof OuterPIOPSumCheckProof,
	publicInputAtR fr.Element,
	context OuterPIOPEndpointContext,
) error {
	endpoint, err := ReconstructOuterPIOPEndpoint(
		proof.FoldedCircuit,
		proof.TreeEvaluations,
		publicInputAtR,
		context,
	)
	if err != nil {
		return err
	}
	var zero fr.Element
	if err := VerifyDegreeFiveSumCheck(zero, context.Point, proof.Transcript, endpoint.Value); err != nil {
		return fmt.Errorf("%w: %v", ErrOuterPIOPEndpoint, err)
	}
	return nil
}

// EvaluatePartitionLabelMLE evaluates the multilinear extension of i_F in
// little-endian order: upsilon(r)=sum_k 2^k r_k.
func EvaluatePartitionLabelMLE(point []fr.Element) fr.Element {
	var result fr.Element
	weight := fr.One()
	for coordinate := range point {
		var term fr.Element
		term.Mul(&weight, &point[coordinate])
		result.Add(&result, &term)
		weight.Double(&weight)
	}
	return result
}

type outerPIOPResiduals struct {
	R0 fr.Element
	R1 fr.Element
	R2 fr.Element
}

type outerPIOPScalarContext struct {
	t                int
	alpha            fr.Element
	lZero            fr.Element
	lStar            fr.Element
	alphaTMinusOne   fr.Element
	lambda           fr.Element
	lambdaSquared    fr.Element
	etaPart          fr.Element
	etaX             fr.Element
	gamma            fr.Element
	wireCosetAtAlpha [LocalWireCount]fr.Element
	zetaSquared      fr.Element
}

func newOuterPIOPScalarContext(state *OuterPIOPCoordinatorState, zeta fr.Element) (outerPIOPScalarContext, error) {
	context := OuterPIOPEndpointContext{
		M:          state.m,
		T:          state.t,
		Omega:      state.omega,
		XStar:      state.xStar,
		Alpha:      state.alpha,
		Zeta:       zeta,
		Theta:      make([]fr.Element, bits.Len(uint(state.m))-1),
		Point:      make([]fr.Element, bits.Len(uint(state.m))-1),
		WireCosets: state.wireCosets,
		Challenges: state.challenges,
	}
	return validateOuterPIOPEndpointContext(context)
}

func validateOuterPIOPEndpointContext(context OuterPIOPEndpointContext) (outerPIOPScalarContext, error) {
	var result outerPIOPScalarContext
	if context.M < 2 || !isPowerOfTwo(context.M) || context.T < context.M || !isPowerOfTwo(context.T) {
		return result, fmt.Errorf("%w: invalid M=%d,T=%d", ErrOuterPIOPEndpoint, context.M, context.T)
	}
	logM := bits.Len(uint(context.M)) - 1
	if len(context.Theta) != logM || len(context.Point) != logM {
		return result, fmt.Errorf("%w: theta/point arity does not equal log2(M)", ErrOuterPIOPEndpoint)
	}
	domain := fft.NewDomain(uint64(context.T))
	if !context.Omega.Equal(&domain.Generator) || !context.XStar.Equal(&domain.GeneratorInv) {
		return result, fmt.Errorf("%w: noncanonical local domain", ErrOuterPIOPEndpoint)
	}
	if err := validateLocalPIOPIndex(LocalPIOPIndex{WireCosets: context.WireCosets}, context.T); err != nil {
		return result, fmt.Errorf("%w: %v", ErrOuterPIOPEndpoint, err)
	}
	if err := validateAlphaOutsideLocalDomain(context.Alpha, context.T); err != nil {
		return result, fmt.Errorf("%w: %v", ErrOuterPIOPEndpoint, err)
	}

	result.t = context.T
	result.alpha = context.Alpha
	result.lambda = context.Challenges.Lambda
	result.lambdaSquared.Square(&context.Challenges.Lambda)
	result.etaPart = context.Challenges.EtaPart
	result.etaX = context.Challenges.EtaX
	result.gamma = context.Challenges.Gamma
	result.zetaSquared.Square(&context.Zeta)
	for wire := 0; wire < LocalWireCount; wire++ {
		result.wireCosetAtAlpha[wire].Mul(&context.WireCosets[wire], &context.Alpha)
	}

	alphaT := powerLocalField(context.Alpha, context.T)
	result.alphaTMinusOne = alphaT
	one := fr.One()
	result.alphaTMinusOne.Sub(&result.alphaTMinusOne, &one)
	tAsField := fr.NewElement(uint64(context.T))
	var denominator fr.Element
	denominator.Sub(&context.Alpha, &one).Mul(&denominator, &tAsField)
	result.lZero.Div(&result.alphaTMinusOne, &denominator)
	denominator.Sub(&context.Alpha, &context.XStar).Mul(&denominator, &tAsField)
	result.lStar.Mul(&context.XStar, &result.alphaTMinusOne).Div(&result.lStar, &denominator)
	return result, nil
}

func evaluateOuterPIOPResiduals(
	terminals LocalTerminalEvaluations,
	partitionLabel fr.Element,
	publicInput fr.Element,
	tree [4]fr.Element,
	context outerPIOPScalarContext,
) outerPIOPResiduals {
	var wires [LocalWireCount]fr.Element
	for wire := 0; wire < LocalWireCount; wire++ {
		wires[wire] = terminals.Alpha[int(AlphaPhiPrimeA)+wire]
		var term fr.Element
		term.Mul(&context.etaPart, &partitionLabel)
		wires[wire].Sub(&wires[wire], &term)
		term.Mul(&context.etaX, &context.wireCosetAtAlpha[wire])
		wires[wire].Sub(&wires[wire], &term)
		wires[wire].Sub(&wires[wire], &context.gamma)
	}

	var gate, term fr.Element
	term.Mul(&wires[LocalWireA], &wires[LocalWireB]).Mul(&term, &terminals.Alpha[AlphaQM])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireA], &terminals.Alpha[AlphaQL])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireB], &terminals.Alpha[AlphaQR])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireC], &terminals.Alpha[AlphaQO])
	gate.Add(&gate, &term)
	gate.Add(&gate, &terminals.Alpha[AlphaQC])
	gate.Add(&gate, &publicInput)

	fAlpha := multiplyThreeLocalScalars(terminals.Alpha[AlphaPhiA], terminals.Alpha[AlphaPhiB], terminals.Alpha[AlphaPhiC])
	fPrimeAlpha := multiplyThreeLocalScalars(terminals.Alpha[AlphaPhiPrimeA], terminals.Alpha[AlphaPhiPrimeB], terminals.Alpha[AlphaPhiPrimeC])
	fXStar := multiplyThreeLocalScalars(terminals.XStar[XStarPhiA], terminals.XStar[XStarPhiB], terminals.XStar[XStarPhiC])
	fPrimeXStar := multiplyThreeLocalScalars(terminals.XStar[XStarPhiPrimeA], terminals.XStar[XStarPhiPrimeB], terminals.XStar[XStarPhiPrimeC])

	var boundary, transition, endpoint fr.Element
	boundary.Sub(&terminals.Alpha[AlphaZ], valuePointer(fr.One())).Mul(&boundary, &context.lZero)
	transition.Mul(&terminals.Alpha[AlphaZ], &fAlpha)
	term.Mul(&terminals.OmegaAlpha[0], &fPrimeAlpha)
	transition.Sub(&transition, &term)
	endpoint.Mul(&terminals.XStar[XStarZ], &fXStar).Sub(&endpoint, &fPrimeXStar)
	term.Mul(&context.lStar, &endpoint)
	transition.Sub(&transition, &term)

	var result outerPIOPResiduals
	result.R0 = gate
	term.Mul(&context.lambda, &boundary)
	result.R0.Add(&result.R0, &term)
	term.Mul(&context.lambdaSquared, &transition)
	result.R0.Add(&result.R0, &term)
	term.Mul(&context.alphaTMinusOne, &terminals.Alpha[AlphaH])
	result.R0.Sub(&result.R0, &term)

	result.R1.Mul(&tree[0], &fPrimeXStar)
	term.Mul(&terminals.XStar[XStarZ], &fXStar)
	result.R1.Sub(&result.R1, &term)
	term.Mul(&tree[2], &tree[3])
	result.R2.Sub(&tree[1], &term)
	return result
}

func outerPIOPTerminalsFromOracleValues(values []fr.Element) LocalTerminalEvaluations {
	var terminals LocalTerminalEvaluations
	copy(terminals.Alpha[:], values[outerOracleTerminalStart:outerOracleTerminalStart+LocalTerminalAlphaCount])
	terminals.OmegaAlpha[0] = values[outerOracleOmegaZ]
	copy(terminals.XStar[:], values[outerOracleXStarStart:outerOracleXStarStart+LocalTerminalXStarCount])
	return terminals
}

func evaluateOuterEquality(theta, point []fr.Element) fr.Element {
	result := fr.One()
	one := fr.One()
	for coordinate := range theta {
		var oneMinusTheta, oneMinusPoint, factor, term fr.Element
		oneMinusTheta.Sub(&one, &theta[coordinate])
		oneMinusPoint.Sub(&one, &point[coordinate])
		factor.Mul(&oneMinusTheta, &oneMinusPoint)
		term.Mul(&theta[coordinate], &point[coordinate])
		factor.Add(&factor, &term)
		result.Mul(&result, &factor)
	}
	return result
}

func equalOuterPIOPChallenges(left, right LocalPIOPChallenges) bool {
	return left.EtaPart.Equal(&right.EtaPart) &&
		left.EtaX.Equal(&right.EtaX) &&
		left.Gamma.Equal(&right.Gamma) &&
		left.Lambda.Equal(&right.Lambda)
}

func equalOuterPIOPWireCosets(left, right [LocalWireCount]fr.Element) bool {
	for wire := 0; wire < LocalWireCount; wire++ {
		if !left[wire].Equal(&right[wire]) {
			return false
		}
	}
	return true
}

func equalOuterPIOPTerminals(left, right LocalTerminalEvaluations) bool {
	leftValues := left.Flatten()
	rightValues := right.Flatten()
	for index := range leftValues {
		if !leftValues[index].Equal(&rightValues[index]) {
			return false
		}
	}
	return true
}
