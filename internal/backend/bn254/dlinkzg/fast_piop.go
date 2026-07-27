package dlinkzg

// This file is the production-arithmetic implementation of the party-local
// PIOP relation. Transcript derivation, MPI transport, commitments, and the
// PCS remain external so benchmarks can account for those layers separately.

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark/internal/utils"
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
	domainSize        int
	domain            *fft.Domain
	extendedDomain    *fft.Domain
	convolutionDomain *fft.Domain
	index             LocalPIOPIndex

	selectors [LocalSelectorCount][]fr.Element
	sigmaX    [LocalWireCount][]fr.Element
	sigmaPart [LocalWireCount][]fr.Element

	sigmaXEvaluations    [LocalWireCount][]fr.Element
	sigmaPartEvaluations [LocalWireCount][]fr.Element
	extendedSelectors    [LocalSelectorCount][]fr.Element
	lZero                []fr.Element
	lStar                []fr.Element
	domainElements       []fr.Element
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
	return preprocessFastLocalPIOPFixedWithDomains(table, index, nil, nil, nil)
}

func preprocessFastLocalPIOPFixedWithDomains(
	table LocalPIOPTable,
	index LocalPIOPIndex,
	domain, extendedDomain, convolutionDomain *fft.Domain,
) (*FastLocalPIOPFixed, error) {
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

	if domain == nil {
		domain = fft.NewDomain(uint64(tableSize))
	}
	if extendedDomain == nil {
		extendedDomain = fft.NewDomain(uint64(4 * tableSize))
	}
	if convolutionDomain == nil {
		convolutionDomain = fft.NewDomain(uint64(2*tableSize - 1))
	}
	fixed := &FastLocalPIOPFixed{
		domainSize:        tableSize,
		domain:            domain,
		extendedDomain:    extendedDomain,
		convolutionDomain: convolutionDomain,
		index:             index,
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
	fixed.domainElements = make([]fr.Element, tableSize)
	power := fr.One()
	for coefficient := 0; coefficient < tableSize; coefficient++ {
		fixed.lZero[coefficient] = inverseT
		fixed.lStar[coefficient].Mul(&inverseT, &power)
		fixed.domainElements[coefficient] = power
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
	tags := [LocalWireCount][]fr.Element{}
	identityTags := [LocalWireCount][]fr.Element{}
	for wire := 0; wire < LocalWireCount; wire++ {
		tagEvaluations[wire] = make([]fr.Element, t)
		identityTagEvaluations[wire] = make([]fr.Element, t)
		tags[wire] = make([]fr.Element, t)
		identityTags[wire] = make([]fr.Element, t)
	}

	var identityConstant fr.Element
	identityConstant.Mul(&etaPart, &state.relation.Index.SlotLabel).Add(&identityConstant, &gamma)
	var identityLinear [LocalWireCount]fr.Element
	for wire := 0; wire < LocalWireCount; wire++ {
		identityLinear[wire].Mul(&etaX, &state.relation.Index.WireCosets[wire])
	}

	parallelFastLoop(t, func(start, end int) {
		for row := start; row < end; row++ {
			for wire := 0; wire < LocalWireCount; wire++ {
				var term fr.Element

				tagEvaluations[wire][row] = state.wires[wire][row]
				term.Mul(&etaPart, &state.fixed.sigmaPartEvaluations[wire][row])
				tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
				term.Mul(&etaX, &state.fixed.sigmaXEvaluations[wire][row])
				tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
				tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &gamma)

				identityTagEvaluations[wire][row] = state.wires[wire][row]
				term.Mul(&etaPart, &state.relation.Index.SlotLabel)
				identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
				term.Mul(&state.relation.Index.WireCosets[wire], &state.fixed.domainElements[row]).Mul(&term, &etaX)
				identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
				identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &gamma)

				tags[wire][row] = state.relation.Wires[wire][row]
				term.Mul(&etaPart, &state.relation.SigmaPart[wire][row])
				tags[wire][row].Add(&tags[wire][row], &term)
				term.Mul(&etaX, &state.relation.SigmaX[wire][row])
				tags[wire][row].Add(&tags[wire][row], &term)
				if row == 0 {
					tags[wire][row].Add(&tags[wire][row], &gamma)
				}

				identityTags[wire][row] = state.relation.Wires[wire][row]
				if row == 0 {
					identityTags[wire][row].Add(&identityTags[wire][row], &identityConstant)
				} else if row == 1 {
					identityTags[wire][row].Add(&identityTags[wire][row], &identityLinear[wire])
				}
			}
		}
	})

	fEvaluations := multiplyFastEvaluationFamilies(tagEvaluations, t)
	fPrimeEvaluations := multiplyFastEvaluationFamilies(identityTagEvaluations, t)
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
	extendedDomain := state.fixed.extendedDomain

	// Keep five extended vectors live. After the gate polynomial is formed,
	// the three wire vectors are reused for f, f', and scratch space.
	wireA := fastExtendedEvaluations(state.relation.Wires[LocalWireA], state.domain, extendedDomain)
	wireB := fastExtendedEvaluations(state.relation.Wires[LocalWireB], state.domain, extendedDomain)
	wireC := fastExtendedEvaluations(state.relation.Wires[LocalWireC], state.domain, extendedDomain)
	gate := state.fastSelectorEvaluations(LocalSelectorM)
	parallelFastLoop(len(gate), func(start, end int) {
		for index := start; index < end; index++ {
			gate[index].Mul(&gate[index], &wireA[index]).Mul(&gate[index], &wireB[index])
		}
	})

	scratch := make([]fr.Element, int(extendedDomain.Cardinality))
	state.fillFastSelectorEvaluations(scratch, LocalSelectorL)
	addFastPointwiseProduct(gate, scratch, wireA)
	state.fillFastSelectorEvaluations(scratch, LocalSelectorR)
	addFastPointwiseProduct(gate, scratch, wireB)
	state.fillFastSelectorEvaluations(scratch, LocalSelectorO)
	addFastPointwiseProduct(gate, scratch, wireC)
	fillFastExtendedEvaluationsSum(
		scratch,
		state.relation.Selectors[LocalSelectorC],
		state.relation.PublicInput,
		extendedDomain,
	)
	addFastPointwise(gate, scratch)

	wireEvaluations := [LocalWireCount][]fr.Element{wireA, wireB, wireC}
	fEvaluations := wireA
	fPrimeEvaluations := wireB
	scratch = wireC
	var identityConstant fr.Element
	identityConstant.Mul(&state.relation.Challenges.EtaPart, &state.relation.Index.SlotLabel).
		Add(&identityConstant, &state.relation.Challenges.Gamma)
	var identityLinear [LocalWireCount]fr.Element
	for wire := 0; wire < LocalWireCount; wire++ {
		identityLinear[wire].Mul(&state.relation.Challenges.EtaX, &state.relation.Index.WireCosets[wire])
	}
	parallelFastLoop(len(fEvaluations), func(start, end int) {
		for index := start; index < end; index++ {
			x := fastDomainElement(extendedDomain, index)
			var identityProduct fr.Element
			identityProduct.SetOne()
			for wire := 0; wire < LocalWireCount; wire++ {
				var identityTag, linearTerm fr.Element
				linearTerm.Mul(&identityLinear[wire], &x)
				identityTag.Add(&wireEvaluations[wire][index], &identityConstant).
					Add(&identityTag, &linearTerm)
				identityProduct.Mul(&identityProduct, &identityTag)
			}
			fEvaluations[index].SetOne()
			fPrimeEvaluations[index] = identityProduct
		}
	})
	for wire := 0; wire < LocalWireCount; wire++ {
		fillFastExtendedEvaluations(scratch, state.relation.Tags[wire], state.domain, extendedDomain)
		multiplyFastPointwise(fEvaluations, scratch)
	}
	zEvaluations := fastExtendedEvaluations(state.relation.Accumulator, state.domain, extendedDomain)
	parallelFastLoop(len(fEvaluations), func(start, end int) {
		for index := start; index < end; index++ {
			fEvaluations[index].Mul(&fEvaluations[index], &zEvaluations[index])
		}
	})
	fillFastShiftedEvaluations(scratch, zEvaluations, len(zEvaluations)/t)
	parallelFastLoop(len(fEvaluations), func(start, end int) {
		for index := start; index < end; index++ {
			var term fr.Element
			term.Mul(&scratch[index], &fPrimeEvaluations[index])
			fEvaluations[index].Sub(&fEvaluations[index], &term)
		}
	})
	fillFastLagrangeEvaluations(
		fPrimeEvaluations,
		scratch,
		extendedDomain,
		t,
		state.relation.XStar,
		state.fixed.lZero[0],
	)
	parallelFastLoop(len(fEvaluations), func(start, end int) {
		for index := start; index < end; index++ {
			var term fr.Element
			term.Mul(&scratch[index], &state.relation.EndpointCorrection)
			fEvaluations[index].Sub(&fEvaluations[index], &term)
		}
	})

	var lambdaSquared fr.Element
	lambdaSquared.Square(&lambda)
	one := fr.One()
	parallelFastLoop(len(gate), func(start, end int) {
		for index := start; index < end; index++ {
			var boundary, transition fr.Element
			boundary.Sub(&zEvaluations[index], &one).Mul(&boundary, &fPrimeEvaluations[index]).Mul(&boundary, &lambda)
			transition.Mul(&fEvaluations[index], &lambdaSquared)
			gate[index].Add(&gate[index], &boundary).Add(&gate[index], &transition)
		}
	})

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
	if fixed == nil || fixed.domain == nil || fixed.extendedDomain == nil || fixed.convolutionDomain == nil ||
		fixed.domainSize < 2 || !isPowerOfTwo(fixed.domainSize) {
		return fmt.Errorf("%w: nil or malformed preprocessed fixed state", ErrInvalidLocalPIOPTable)
	}
	if fixed.domain.Cardinality != uint64(fixed.domainSize) ||
		fixed.extendedDomain.Cardinality != uint64(4*fixed.domainSize) ||
		fixed.convolutionDomain.Cardinality != uint64(2*fixed.domainSize) {
		return fmt.Errorf("%w: malformed preprocessed FFT domains", ErrInvalidLocalPIOPTable)
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
	if len(fixed.lZero) != fixed.domainSize || len(fixed.lStar) != fixed.domainSize || len(fixed.domainElements) != fixed.domainSize {
		return fmt.Errorf("%w: malformed preprocessed Lagrange basis", ErrInvalidLocalPIOPTable)
	}
	return nil
}

func fastExtendedEvaluations(coefficients []fr.Element, baseDomain, extendedDomain *fft.Domain) []fr.Element {
	result := make([]fr.Element, int(extendedDomain.Cardinality))
	fillFastExtendedEvaluations(result, coefficients, baseDomain, extendedDomain)
	return result
}

func (fixed *FastLocalPIOPFixed) preprocessExtendedSelectors() {
	for _, selector := range []LocalSelector{LocalSelectorM, LocalSelectorL, LocalSelectorR, LocalSelectorO} {
		if len(fixed.extendedSelectors[selector]) == int(fixed.extendedDomain.Cardinality) {
			continue
		}
		fixed.extendedSelectors[selector] = fastExtendedEvaluations(
			fixed.selectors[selector],
			fixed.domain,
			fixed.extendedDomain,
		)
	}
}

func (state *FastLocalPIOPState) fastSelectorEvaluations(selector LocalSelector) []fr.Element {
	if cached := state.fixed.extendedSelectors[selector]; len(cached) == int(state.fixed.extendedDomain.Cardinality) {
		return cloneLocalElements(cached)
	}
	return fastExtendedEvaluations(state.relation.Selectors[selector], state.domain, state.fixed.extendedDomain)
}

func (state *FastLocalPIOPState) fillFastSelectorEvaluations(destination []fr.Element, selector LocalSelector) {
	if cached := state.fixed.extendedSelectors[selector]; len(cached) == len(destination) {
		copy(destination, cached)
		return
	}
	fillFastExtendedEvaluations(destination, state.relation.Selectors[selector], state.domain, state.fixed.extendedDomain)
}

// fillFastExtendedEvaluations uses one cached size-4T domain transform. The
// four-FFTPart alternative used by Pianist was measured here as slower due to
// its per-part copies and allocations once fixed selectors were preprocessed.
func fillFastExtendedEvaluations(destination, coefficients []fr.Element, baseDomain, extendedDomain *fft.Domain) {
	_ = baseDomain
	for index := range destination {
		destination[index].SetZero()
	}
	copy(destination, coefficients)
	extendedDomain.FFT(destination, fft.DIF)
	fft.BitReverse(destination)
}

func fillFastExtendedEvaluationsSum(destination, left, right []fr.Element, domain *fft.Domain) {
	for index := range destination {
		destination[index].SetZero()
	}
	copy(destination, left)
	parallelFastLoop(len(right), func(start, end int) {
		for index := start; index < end; index++ {
			destination[index].Add(&destination[index], &right[index])
		}
	})
	domain.FFT(destination, fft.DIF)
	fft.BitReverse(destination)
}

func fastDomainElement(domain *fft.Domain, index int) fr.Element {
	half := int(domain.Cardinality / 2)
	if index <= half {
		return domain.Twiddles[0][index]
	}
	var result fr.Element
	result.Neg(&domain.Twiddles[0][index-half])
	return result
}

func fillFastShiftedEvaluations(destination, source []fr.Element, naturalShift int) {
	mask := len(source) - 1
	parallelFastLoop(len(source), func(start, end int) {
		for index := start; index < end; index++ {
			destination[index] = source[(index+naturalShift)&mask]
		}
	})
}

// fillFastLagrangeEvaluations evaluates L_0 and L_{omega^-1} directly on the
// size-4T quotient domain. One batch inversion of
// (x-1)(x-omega^-1) replaces two complete FFTs.
func fillFastLagrangeEvaluations(
	lZero, lStar []fr.Element,
	domain *fft.Domain,
	baseSize int,
	xStar, inverseBaseSize fr.Element,
) {
	denominatorProducts := lZero
	one := fr.One()
	parallelFastLoop(len(denominatorProducts), func(start, end int) {
		for index := start; index < end; index++ {
			if index%4 == 0 {
				denominatorProducts[index].SetZero()
				continue
			}
			x := fastDomainElement(domain, index)
			var xMinusOne, xMinusStar fr.Element
			xMinusOne.Sub(&x, &one)
			xMinusStar.Sub(&x, &xStar)
			denominatorProducts[index].Mul(&xMinusOne, &xMinusStar)
		}
	})
	inverseProducts := fr.BatchInvert(denominatorProducts)

	quarterRoot := fastDomainElement(domain, baseSize)
	var numeratorByResidue [4]fr.Element
	power := fr.One()
	for residue := 0; residue < 4; residue++ {
		numeratorByResidue[residue].Sub(&power, &one)
		power.Mul(&power, &quarterRoot)
	}
	starIndex := len(lStar) - len(lStar)/baseSize
	parallelFastLoop(len(lZero), func(start, end int) {
		for index := start; index < end; index++ {
			if index%4 == 0 {
				lZero[index].SetZero()
				lStar[index].SetZero()
				if index == 0 {
					lZero[index].SetOne()
				}
				if index == starIndex {
					lStar[index].SetOne()
				}
				continue
			}
			x := fastDomainElement(domain, index)
			var xMinusOne, xMinusStar, scale fr.Element
			xMinusOne.Sub(&x, &one)
			xMinusStar.Sub(&x, &xStar)
			scale.Mul(&numeratorByResidue[index%4], &inverseBaseSize).
				Mul(&scale, &inverseProducts[index])
			lZero[index].Mul(&scale, &xMinusStar)
			lStar[index].Mul(&scale, &xMinusOne).Mul(&lStar[index], &xStar)
		}
	})
}

func addFastPointwise(destination, source []fr.Element) {
	parallelFastLoop(len(destination), func(start, end int) {
		for index := start; index < end; index++ {
			destination[index].Add(&destination[index], &source[index])
		}
	})
}

func addFastPointwiseProduct(destination, left, right []fr.Element) {
	parallelFastLoop(len(destination), func(start, end int) {
		for index := start; index < end; index++ {
			var term fr.Element
			term.Mul(&left[index], &right[index])
			destination[index].Add(&destination[index], &term)
		}
	})
}

func multiplyFastPointwise(destination, source []fr.Element) {
	parallelFastLoop(len(destination), func(start, end int) {
		for index := start; index < end; index++ {
			destination[index].Mul(&destination[index], &source[index])
		}
	})
}

func multiplyFastEvaluationFamilies(families [LocalWireCount][]fr.Element, size int) []fr.Element {
	result := make([]fr.Element, size)
	parallelFastLoop(size, func(start, end int) {
		for row := start; row < end; row++ {
			result[row].SetOne()
			for wire := 0; wire < LocalWireCount; wire++ {
				result[row].Mul(&result[row], &families[wire][row])
			}
		}
	})
	return result
}

func parallelFastLoop(size int, work func(start, end int)) {
	tasks := runtime.GOMAXPROCS(0)
	if size <= 0 {
		return
	}
	if tasks <= 1 {
		work(0, size)
		return
	}
	utils.Parallelize(size, work, tasks)
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
