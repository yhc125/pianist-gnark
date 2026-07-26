package dlinkzg

// This file is the production-arithmetic implementation of the party-local
// PIOP relation. Transcript derivation, MPI transport, commitments, and the
// PCS remain external so benchmarks can account for those layers separately.

import (
	"errors"
	"fmt"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

const (
	// FastLocalPIOPArithmeticNotice fixes the benchmark boundary of this slice.
	FastLocalPIOPArithmeticNotice = "online production local-PIOP arithmetic only: reusable fixed preprocessing, transcript, MPI, commitments, and PCS are external"

	// A size-4T quotient domain must fit in the BN254 scalar field's 2^28
	// two-adic subgroup.
	FastLocalPIOPMaxDomainSize = 1 << 26
)

var (
	ErrFastLocalPIOPStage          = errors.New("dlinkzg: invalid fast local PIOP stage")
	ErrFastLocalPIOPDomainTooLarge = errors.New("dlinkzg: fast local PIOP domain is too large")
)

// FastLocalPIOPStage records which Fiat--Shamir messages have been fixed.
// A failed state is terminal: callers must restart from PrepareFastLocalPIOP
// rather than retry with different challenges.
type FastLocalPIOPStage uint8

const (
	FastLocalPIOPUninitialized FastLocalPIOPStage = iota
	FastLocalPIOPW0Ready
	FastLocalPIOPW1Ready
	FastLocalPIOPW2Ready
	FastLocalPIOPFailed
)

// FastLocalPIOPW0 contains exactly the witness polynomials committed in W0.
// Fixed selector and permutation polynomials remain in the prepared state.
type FastLocalPIOPW0 struct {
	Wires [LocalWireCount][]fr.Element
}

// FastLocalPIOPW1 is the eta/gamma-dependent state available when W1 is
// committed. It deliberately has no lambda field.
type FastLocalPIOPW1 struct {
	EtaPart fr.Element
	EtaX    fr.Element
	Gamma   fr.Element

	Tags         [LocalWireCount][]fr.Element
	IdentityTags [LocalWireCount][]fr.Element
	Accumulator  []fr.Element

	EndpointCorrection  fr.Element
	BlockFactor         fr.Element
	BoundaryNumerator   fr.Element
	BoundaryDenominator fr.Element
}

// FastLocalPIOPFixed is reusable setup/keygen state. Selector and permutation
// interpolation happens once here, outside the per-proof arithmetic path.
// Its fields are private so completed relations cannot mutate future proofs.
type FastLocalPIOPFixed struct {
	domainSize int
	domain     *fft.Domain
	index      LocalPIOPIndex

	selectors [LocalSelectorCount][]fr.Element
	sigmaX    [LocalWireCount][]fr.Element
	sigmaPart [LocalWireCount][]fr.Element

	sigmaXEvaluations    [LocalWireCount][]fr.Element
	sigmaPartEvaluations [LocalWireCount][]fr.Element
	lZero                []fr.Element
	lStar                []fr.Element
}

// FastLocalPIOPState enforces the W0 -> W1 -> W2 Fiat--Shamir order. The
// online wire rows are retained only until the accumulator has been built.
type FastLocalPIOPState struct {
	stage    FastLocalPIOPStage
	domain   *fft.Domain
	fixed    *FastLocalPIOPFixed
	wires    [LocalWireCount][]fr.Element
	relation *LocalPIOPRelation
}

// PreprocessFastLocalPIOPFixed validates and interpolates the fixed selector
// and permutation columns. Wires and PublicInput are deliberately ignored, so
// one result can be reused for multiple witnesses and public statements.
func PreprocessFastLocalPIOPFixed(table LocalPIOPTable, index LocalPIOPIndex) (*FastLocalPIOPFixed, error) {
	tableSize := len(table.Selectors[LocalSelectorM])
	if err := validateFastLocalPIOPDomainSize(tableSize); err != nil {
		return nil, err
	}
	if err := validateFastLocalPIOPFixedTable(table, tableSize); err != nil {
		return nil, err
	}
	if err := validateLocalPIOPIndex(index, tableSize); err != nil {
		return nil, err
	}

	domain := fft.NewDomain(uint64(tableSize))
	fixed := &FastLocalPIOPFixed{
		domainSize: tableSize,
		domain:     domain,
		index:      index,
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		fixed.sigmaX[wire] = interpolateLocalTable(table.SigmaX[wire], domain)
		fixed.sigmaPart[wire] = interpolateLocalTable(table.SigmaPart[wire], domain)
		fixed.sigmaXEvaluations[wire] = cloneLocalElements(table.SigmaX[wire])
		fixed.sigmaPartEvaluations[wire] = cloneLocalElements(table.SigmaPart[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		fixed.selectors[selector] = interpolateLocalTable(table.Selectors[selector], domain)
	}

	// L_0 has constant coefficients 1/T. For x_star=omega^-1,
	// L_star's coefficient of X^k is omega^k/T.
	tAsField := fr.NewElement(uint64(tableSize))
	var inverseT fr.Element
	inverseT.Inverse(&tAsField)
	fixed.lZero = make([]fr.Element, tableSize)
	fixed.lStar = make([]fr.Element, tableSize)
	power := fr.One()
	for coefficient := 0; coefficient < tableSize; coefficient++ {
		fixed.lZero[coefficient] = inverseT
		fixed.lStar[coefficient].Mul(&inverseT, &power)
		power.Mul(&power, &domain.Generator)
	}
	return fixed, nil
}

// DomainSize reports the fixed local FFT width.
func (fixed *FastLocalPIOPFixed) DomainSize() int {
	if fixed == nil {
		return 0
	}
	return fixed.domainSize
}

// PrepareFastLocalPIOPWithFixed interpolates only online witness/public
// columns and fixes W0. The fixed argument is safe to reuse across proofs.
func PrepareFastLocalPIOPWithFixed(table LocalPIOPTable, fixed *FastLocalPIOPFixed) (*FastLocalPIOPState, error) {
	if err := validateFastLocalPIOPFixed(fixed); err != nil {
		return nil, err
	}
	if err := validateFastLocalPIOPOnlineTable(table, fixed.domainSize); err != nil {
		return nil, err
	}

	t := fixed.domainSize
	relation := &LocalPIOPRelation{
		DomainSize: t,
		Omega:      fixed.domain.Generator,
		XStar:      fixed.domain.GeneratorInv,
		Index:      fixed.index,
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		relation.Wires[wire] = interpolateLocalTable(table.Wires[wire], fixed.domain)
		relation.SigmaX[wire] = cloneLocalElements(fixed.sigmaX[wire])
		relation.SigmaPart[wire] = cloneLocalElements(fixed.sigmaPart[wire])
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		relation.Selectors[selector] = cloneLocalElements(fixed.selectors[selector])
	}
	relation.PublicInput = interpolateLocalTable(table.PublicInput, fixed.domain)

	return &FastLocalPIOPState{
		stage:    FastLocalPIOPW0Ready,
		domain:   fixed.domain,
		fixed:    fixed,
		wires:    cloneFastLocalPIOPWires(table.Wires),
		relation: relation,
	}, nil
}

// PrepareFastLocalPIOP is a convenience wrapper for callers without reusable
// setup state. Production proving and benchmarks should preprocess fixed
// columns once and call PrepareFastLocalPIOPWithFixed for each proof.
func PrepareFastLocalPIOP(table LocalPIOPTable, index LocalPIOPIndex) (*FastLocalPIOPState, error) {
	tableSize, err := validateLocalPIOPTable(table)
	if err != nil {
		return nil, err
	}
	if err := validateFastLocalPIOPDomainSize(tableSize); err != nil {
		return nil, err
	}
	fixed, err := PreprocessFastLocalPIOPFixed(table, index)
	if err != nil {
		return nil, err
	}
	return PrepareFastLocalPIOPWithFixed(table, fixed)
}

// Stage returns the current immutable stage marker.
func (state *FastLocalPIOPState) Stage() FastLocalPIOPStage {
	if state == nil {
		return FastLocalPIOPUninitialized
	}
	return state.stage
}

// W0 returns fresh coefficient slices suitable for the three W0 commitments.
func (state *FastLocalPIOPState) W0() (FastLocalPIOPW0, error) {
	var result FastLocalPIOPW0
	if state == nil || state.stage == FastLocalPIOPUninitialized || state.stage == FastLocalPIOPFailed {
		return result, fmt.Errorf("%w: W0 is unavailable", ErrFastLocalPIOPStage)
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		result.Wires[wire] = cloneLocalElements(state.relation.Wires[wire])
	}
	return result, nil
}

// BuildAccumulator applies eta_part, eta_X, and gamma, then constructs the
// tags, z, endpoint correction, and division-free block factor. No lambda is
// accepted or read in this stage.
func (state *FastLocalPIOPState) BuildAccumulator(etaPart, etaX, gamma fr.Element) (FastLocalPIOPW1, error) {
	var result FastLocalPIOPW1
	if err := state.requireStage(FastLocalPIOPW0Ready, "build W1"); err != nil {
		return result, err
	}

	t := state.relation.DomainSize
	tagEvaluations := [LocalWireCount][]fr.Element{}
	identityTagEvaluations := [LocalWireCount][]fr.Element{}
	for wire := 0; wire < LocalWireCount; wire++ {
		tagEvaluations[wire] = make([]fr.Element, t)
		identityTagEvaluations[wire] = make([]fr.Element, t)
	}

	x := fr.One()
	for row := 0; row < t; row++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			tagEvaluations[wire][row] = state.wires[wire][row]
			var term fr.Element
			term.Mul(&etaPart, &state.fixed.sigmaPartEvaluations[wire][row])
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
			term.Mul(&etaX, &state.fixed.sigmaXEvaluations[wire][row])
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &gamma)

			identityTagEvaluations[wire][row] = state.wires[wire][row]
			term.Mul(&etaPart, &state.relation.Index.SlotLabel)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
			term.Mul(&state.relation.Index.WireCosets[wire], &x).Mul(&term, &etaX)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &gamma)
		}
		x.Mul(&x, &state.relation.Omega)
	}

	tags := [LocalWireCount][]fr.Element{}
	identityTags := [LocalWireCount][]fr.Element{}
	for wire := 0; wire < LocalWireCount; wire++ {
		tags[wire] = cloneLocalElements(state.relation.Wires[wire])
		addScaledLocalPolynomial(tags[wire], state.relation.SigmaPart[wire], etaPart)
		addScaledLocalPolynomial(tags[wire], state.relation.SigmaX[wire], etaX)
		tags[wire][0].Add(&tags[wire][0], &gamma)

		identityTags[wire] = cloneLocalElements(state.relation.Wires[wire])
		var constant, linear fr.Element
		constant.Mul(&etaPart, &state.relation.Index.SlotLabel).Add(&constant, &gamma)
		identityTags[wire][0].Add(&identityTags[wire][0], &constant)
		linear.Mul(&etaX, &state.relation.Index.WireCosets[wire])
		identityTags[wire][1].Add(&identityTags[wire][1], &linear)
	}

	fEvaluations := multiplyLocalEvaluationFamilies(tagEvaluations, t)
	fPrimeEvaluations := multiplyLocalEvaluationFamilies(identityTagEvaluations, t)
	for row := 0; row < t; row++ {
		if fPrimeEvaluations[row].IsZero() {
			state.stage = FastLocalPIOPFailed
			return result, fmt.Errorf("%w at row %d", ErrZeroLocalTagDenominator, row)
		}
	}
	inverseDenominators := fr.BatchInvert(fPrimeEvaluations[:t-1])
	zEvaluations := make([]fr.Element, t)
	zEvaluations[0].SetOne()
	for row := 0; row < t-1; row++ {
		zEvaluations[row+1].Mul(&zEvaluations[row], &fEvaluations[row]).
			Mul(&zEvaluations[row+1], &inverseDenominators[row])
	}
	accumulator := interpolateLocalTable(zEvaluations, state.domain)

	var boundaryNumerator, endpointCorrection, blockFactor fr.Element
	boundaryNumerator.Mul(&zEvaluations[t-1], &fEvaluations[t-1])
	boundaryDenominator := fPrimeEvaluations[t-1]
	endpointCorrection.Sub(&boundaryNumerator, &boundaryDenominator)
	blockFactor.Div(&boundaryNumerator, &boundaryDenominator)

	state.relation.Challenges = LocalPIOPChallenges{
		EtaPart: etaPart,
		EtaX:    etaX,
		Gamma:   gamma,
	}
	state.relation.Tags = tags
	state.relation.IdentityTags = identityTags
	state.relation.Accumulator = accumulator
	state.relation.EndpointCorrection = endpointCorrection
	state.relation.BlockFactor = blockFactor
	state.relation.BoundaryNumerator = boundaryNumerator
	state.relation.BoundaryDenominator = boundaryDenominator
	state.wires = [LocalWireCount][]fr.Element{}
	state.stage = FastLocalPIOPW1Ready

	result = state.snapshotW1()
	return result, nil
}

// BuildQuotient applies lambda and completes the three W2 quotient chunks.
// All products are formed as pointwise products on one size-4T FFT domain.
func (state *FastLocalPIOPState) BuildQuotient(lambda fr.Element) (*LocalPIOPRelation, error) {
	if err := state.requireStage(FastLocalPIOPW1Ready, "build W2"); err != nil {
		return nil, err
	}

	numerator, err := state.fastQuotientNumerator(lambda)
	if err != nil {
		state.stage = FastLocalPIOPFailed
		return nil, err
	}
	quotient, err := fastDivideByLocalVanishing(numerator, state.relation.DomainSize)
	if err != nil {
		state.stage = FastLocalPIOPFailed
		return nil, err
	}

	t := state.relation.DomainSize
	var chunks [3][]fr.Element
	for chunk := range chunks {
		chunks[chunk] = make([]fr.Element, t)
		copy(chunks[chunk], quotient[chunk*t:(chunk+1)*t])
	}
	state.relation.Challenges.Lambda = lambda
	state.relation.Quotient = chunks
	state.stage = FastLocalPIOPW2Ready
	return state.relation, nil
}

func (state *FastLocalPIOPState) fastQuotientNumerator(lambda fr.Element) ([]fr.Element, error) {
	t := state.relation.DomainSize
	if err := validateFastLocalPIOPDomainSize(t); err != nil {
		return nil, err
	}
	extendedDomain := fft.NewDomain(uint64(4 * t))

	// Keep five extended vectors live. After the gate polynomial is formed,
	// the three wire vectors are reused for f, f', and scratch space.
	wireA := fastExtendedEvaluations(state.relation.Wires[LocalWireA], extendedDomain)
	wireB := fastExtendedEvaluations(state.relation.Wires[LocalWireB], extendedDomain)
	wireC := fastExtendedEvaluations(state.relation.Wires[LocalWireC], extendedDomain)
	gate := fastExtendedEvaluations(state.relation.Selectors[LocalSelectorM], extendedDomain)
	for index := range gate {
		gate[index].Mul(&gate[index], &wireA[index]).Mul(&gate[index], &wireB[index])
	}

	scratch := make([]fr.Element, int(extendedDomain.Cardinality))
	fillFastExtendedEvaluations(scratch, state.relation.Selectors[LocalSelectorL], extendedDomain)
	addFastPointwiseProduct(gate, scratch, wireA)
	fillFastExtendedEvaluations(scratch, state.relation.Selectors[LocalSelectorR], extendedDomain)
	addFastPointwiseProduct(gate, scratch, wireB)
	fillFastExtendedEvaluations(scratch, state.relation.Selectors[LocalSelectorO], extendedDomain)
	addFastPointwiseProduct(gate, scratch, wireC)
	fillFastExtendedEvaluations(scratch, state.relation.Selectors[LocalSelectorC], extendedDomain)
	addFastPointwise(gate, scratch)
	fillFastExtendedEvaluations(scratch, state.relation.PublicInput, extendedDomain)
	addFastPointwise(gate, scratch)

	fEvaluations := wireA
	fPrimeEvaluations := wireB
	scratch = wireC
	for index := range fEvaluations {
		fEvaluations[index].SetOne()
		fPrimeEvaluations[index].SetOne()
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		fillFastExtendedEvaluations(scratch, state.relation.Tags[wire], extendedDomain)
		multiplyFastPointwise(fEvaluations, scratch)
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		fillFastExtendedEvaluations(scratch, state.relation.IdentityTags[wire], extendedDomain)
		multiplyFastPointwise(fPrimeEvaluations, scratch)
	}

	zEvaluations := fastExtendedEvaluations(state.relation.Accumulator, extendedDomain)
	for index := range fEvaluations {
		fEvaluations[index].Mul(&fEvaluations[index], &zEvaluations[index])
	}
	zOmega := make([]fr.Element, t)
	power := fr.One()
	for coefficient := range zOmega {
		zOmega[coefficient].Mul(&state.relation.Accumulator[coefficient], &power)
		power.Mul(&power, &state.relation.Omega)
	}
	fillFastExtendedEvaluations(scratch, zOmega, extendedDomain)
	for index := range fEvaluations {
		var term fr.Element
		term.Mul(&scratch[index], &fPrimeEvaluations[index])
		fEvaluations[index].Sub(&fEvaluations[index], &term)
	}
	fillFastExtendedEvaluations(scratch, state.fixed.lStar, extendedDomain)
	for index := range fEvaluations {
		var term fr.Element
		term.Mul(&scratch[index], &state.relation.EndpointCorrection)
		fEvaluations[index].Sub(&fEvaluations[index], &term)
	}

	var lambdaSquared fr.Element
	lambdaSquared.Square(&lambda)
	fillFastExtendedEvaluations(scratch, state.fixed.lZero, extendedDomain)
	one := fr.One()
	for index := range gate {
		var boundary, transition fr.Element
		boundary.Sub(&zEvaluations[index], &one).Mul(&boundary, &scratch[index]).Mul(&boundary, &lambda)
		transition.Mul(&fEvaluations[index], &lambdaSquared)
		gate[index].Add(&gate[index], &boundary).Add(&gate[index], &transition)
	}

	extendedDomain.FFTInverse(gate, fft.DIF)
	fft.BitReverse(gate)
	return gate, nil
}

func (state *FastLocalPIOPState) snapshotW1() FastLocalPIOPW1 {
	result := FastLocalPIOPW1{
		EtaPart:             state.relation.Challenges.EtaPart,
		EtaX:                state.relation.Challenges.EtaX,
		Gamma:               state.relation.Challenges.Gamma,
		Accumulator:         cloneLocalElements(state.relation.Accumulator),
		EndpointCorrection:  state.relation.EndpointCorrection,
		BlockFactor:         state.relation.BlockFactor,
		BoundaryNumerator:   state.relation.BoundaryNumerator,
		BoundaryDenominator: state.relation.BoundaryDenominator,
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		result.Tags[wire] = cloneLocalElements(state.relation.Tags[wire])
		result.IdentityTags[wire] = cloneLocalElements(state.relation.IdentityTags[wire])
	}
	return result
}

func (state *FastLocalPIOPState) requireStage(want FastLocalPIOPStage, operation string) error {
	if state == nil {
		return fmt.Errorf("%w: %s on nil state", ErrFastLocalPIOPStage, operation)
	}
	if state.stage != want {
		return fmt.Errorf("%w: %s requires stage %d, got %d", ErrFastLocalPIOPStage, operation, want, state.stage)
	}
	return nil
}

func validateFastLocalPIOPDomainSize(t int) error {
	if t > FastLocalPIOPMaxDomainSize {
		return fmt.Errorf("%w: T=%d exceeds 2^26", ErrFastLocalPIOPDomainTooLarge, t)
	}
	return nil
}

func validateFastLocalPIOPFixedTable(table LocalPIOPTable, t int) error {
	if t < 2 || !isPowerOfTwo(t) {
		return fmt.Errorf("%w: fixed row count %d is not a power of two at least two", ErrInvalidLocalPIOPTable, t)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if len(table.Selectors[selector]) != t {
			return fmt.Errorf("%w: fixed selector %d does not have T=%d rows", ErrInvalidLocalPIOPTable, selector, t)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if len(table.SigmaX[wire]) != t || len(table.SigmaPart[wire]) != t {
			return fmt.Errorf("%w: fixed permutation table %d does not have T=%d rows", ErrInvalidLocalPIOPTable, wire, t)
		}
	}
	return nil
}

func validateFastLocalPIOPOnlineTable(table LocalPIOPTable, t int) error {
	for wire := 0; wire < LocalWireCount; wire++ {
		if len(table.Wires[wire]) != t {
			return fmt.Errorf("%w: online wire %d does not have T=%d rows", ErrInvalidLocalPIOPTable, wire, t)
		}
	}
	if len(table.PublicInput) != t {
		return fmt.Errorf("%w: online public-input table does not have T=%d rows", ErrInvalidLocalPIOPTable, t)
	}
	return nil
}

func validateFastLocalPIOPFixed(fixed *FastLocalPIOPFixed) error {
	if fixed == nil || fixed.domain == nil || fixed.domainSize < 2 || !isPowerOfTwo(fixed.domainSize) {
		return fmt.Errorf("%w: nil or malformed preprocessed fixed state", ErrInvalidLocalPIOPTable)
	}
	if err := validateFastLocalPIOPDomainSize(fixed.domainSize); err != nil {
		return err
	}
	if err := validateLocalPIOPIndex(fixed.index, fixed.domainSize); err != nil {
		return err
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if len(fixed.selectors[selector]) != fixed.domainSize {
			return fmt.Errorf("%w: malformed preprocessed selector %d", ErrInvalidLocalPIOPTable, selector)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if len(fixed.sigmaX[wire]) != fixed.domainSize || len(fixed.sigmaPart[wire]) != fixed.domainSize ||
			len(fixed.sigmaXEvaluations[wire]) != fixed.domainSize || len(fixed.sigmaPartEvaluations[wire]) != fixed.domainSize {
			return fmt.Errorf("%w: malformed preprocessed permutation table %d", ErrInvalidLocalPIOPTable, wire)
		}
	}
	if len(fixed.lZero) != fixed.domainSize || len(fixed.lStar) != fixed.domainSize {
		return fmt.Errorf("%w: malformed preprocessed Lagrange basis", ErrInvalidLocalPIOPTable)
	}
	return nil
}

func fastExtendedEvaluations(coefficients []fr.Element, domain *fft.Domain) []fr.Element {
	result := make([]fr.Element, int(domain.Cardinality))
	fillFastExtendedEvaluations(result, coefficients, domain)
	return result
}

func fillFastExtendedEvaluations(destination, coefficients []fr.Element, domain *fft.Domain) {
	for index := range destination {
		destination[index].SetZero()
	}
	copy(destination, coefficients)
	domain.FFT(destination, fft.DIF)
	fft.BitReverse(destination)
}

func addFastPointwise(destination, source []fr.Element) {
	for index := range destination {
		destination[index].Add(&destination[index], &source[index])
	}
}

func addFastPointwiseProduct(destination, left, right []fr.Element) {
	for index := range destination {
		var term fr.Element
		term.Mul(&left[index], &right[index])
		destination[index].Add(&destination[index], &term)
	}
}

func multiplyFastPointwise(destination, source []fr.Element) {
	for index := range destination {
		destination[index].Mul(&destination[index], &source[index])
	}
}

// fastDivideByLocalVanishing performs synthetic division by X^T-1. For a
// size-4T numerator it performs O(T) field operations after the FFTs.
func fastDivideByLocalVanishing(numerator []fr.Element, t int) ([]fr.Element, error) {
	if t < 1 || len(numerator) < t {
		return nil, fmt.Errorf("%w: malformed numerator width", ErrLocalQuotientNotDivisible)
	}
	remainder := cloneLocalElements(numerator)
	quotient := make([]fr.Element, len(remainder)-t)
	for degree := len(remainder) - 1; degree >= t; degree-- {
		coefficient := remainder[degree]
		if coefficient.IsZero() {
			continue
		}
		quotient[degree-t] = coefficient
		remainder[degree].SetZero()
		remainder[degree-t].Add(&remainder[degree-t], &coefficient)
	}
	for degree := 0; degree < t; degree++ {
		if !remainder[degree].IsZero() {
			return nil, fmt.Errorf("%w: nonzero remainder coefficient %d", ErrLocalQuotientNotDivisible, degree)
		}
	}
	return quotient, nil
}

func cloneFastLocalPIOPWires(wires [LocalWireCount][]fr.Element) [LocalWireCount][]fr.Element {
	var result [LocalWireCount][]fr.Element
	for wire := 0; wire < LocalWireCount; wire++ {
		result[wire] = cloneLocalElements(wires[wire])
	}
	return result
}
