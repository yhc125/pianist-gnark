package dlinkzg

// This file implements only the exact, party-local polynomial relation used
// by the row-partitioned Plonk PIOP. It deliberately stops before SparseR1CS
// extraction, commitments, Fiat--Shamir, ProductCheck, SumCheck, MPI, and the
// DLinKZG opening protocol. Keeping that boundary explicit prevents this
// algebra slice from being mistaken for a production backend.

import (
	"errors"
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

const (
	// LocalPIOPAlgebraNotice records the exact scope of this implementation.
	LocalPIOPAlgebraNotice = "local PIOP algebra only: explicit row tables, no SparseR1CS adapter, commitments, Fiat-Shamir, ProductCheck/SumCheck transcript, MPI, or PCS"

	LocalWireCount               = 3
	LocalSelectorCount           = 5
	LocalTerminalAlphaCount      = 13
	LocalTerminalOmegaAlphaCount = 1
	LocalTerminalXStarCount      = 7
	LocalTerminalTotalCount      = LocalTerminalAlphaCount + LocalTerminalOmegaAlphaCount + LocalTerminalXStarCount
	LocalCompressedSourceCount   = 3
)

var (
	ErrInvalidLocalPIOPTable         = errors.New("dlinkzg: invalid local PIOP table")
	ErrZeroLocalTagDenominator       = errors.New("dlinkzg: zero local identity-tag denominator")
	ErrLocalQuotientNotDivisible     = errors.New("dlinkzg: local quotient numerator is not divisible by X^T-1")
	ErrInvalidLocalPIOPRelation      = errors.New("dlinkzg: invalid local PIOP relation")
	ErrInvalidLocalBlockFactor       = errors.New("dlinkzg: invalid division-free local block factor")
	ErrInvalidLocalTerminalPoint     = errors.New("dlinkzg: invalid local terminal point")
	ErrInvalidLocalSourceCompression = errors.New("dlinkzg: invalid local terminal source compression")
)

// LocalWire is the fixed three-wire order used throughout the PIOP.
type LocalWire uint8

const (
	LocalWireA LocalWire = iota
	LocalWireB
	LocalWireC
)

// LocalSelector is the fixed selector order q_M,q_L,q_R,q_O,q_C.
type LocalSelector uint8

const (
	LocalSelectorM LocalSelector = iota
	LocalSelectorL
	LocalSelectorR
	LocalSelectorO
	LocalSelectorC
)

// LocalTerminalPoint identifies one of the three terminal opening points.
type LocalTerminalPoint uint8

const (
	LocalTerminalAtAlpha LocalTerminalPoint = iota
	LocalTerminalAtOmegaAlpha
	LocalTerminalAtXStar
)

// LocalTerminalFamily names a polynomial in the canonical 13/1/7 terminal
// inventory. A family may occur at more than one terminal point.
type LocalTerminalFamily string

const (
	LocalTerminalPhiA      LocalTerminalFamily = "phi_a"
	LocalTerminalPhiB      LocalTerminalFamily = "phi_b"
	LocalTerminalPhiC      LocalTerminalFamily = "phi_c"
	LocalTerminalPhiPrimeA LocalTerminalFamily = "phi_prime_a"
	LocalTerminalPhiPrimeB LocalTerminalFamily = "phi_prime_b"
	LocalTerminalPhiPrimeC LocalTerminalFamily = "phi_prime_c"
	LocalTerminalZ         LocalTerminalFamily = "z"
	LocalTerminalQM        LocalTerminalFamily = "q_M"
	LocalTerminalQL        LocalTerminalFamily = "q_L"
	LocalTerminalQR        LocalTerminalFamily = "q_R"
	LocalTerminalQO        LocalTerminalFamily = "q_O"
	LocalTerminalQC        LocalTerminalFamily = "q_C"
	LocalTerminalHAlpha    LocalTerminalFamily = "h^(alpha)"
)

var localTerminalOrders = [LocalCompressedSourceCount][]LocalTerminalFamily{
	{
		LocalTerminalPhiA, LocalTerminalPhiB, LocalTerminalPhiC,
		LocalTerminalPhiPrimeA, LocalTerminalPhiPrimeB, LocalTerminalPhiPrimeC,
		LocalTerminalZ,
		LocalTerminalQM, LocalTerminalQL, LocalTerminalQR, LocalTerminalQO, LocalTerminalQC,
		LocalTerminalHAlpha,
	},
	{LocalTerminalZ},
	{
		LocalTerminalPhiA, LocalTerminalPhiB, LocalTerminalPhiC,
		LocalTerminalPhiPrimeA, LocalTerminalPhiPrimeB, LocalTerminalPhiPrimeC,
		LocalTerminalZ,
	},
}

// AlphaTerminalIndex fixes the array offsets at alpha.
type AlphaTerminalIndex uint8

const (
	AlphaPhiA AlphaTerminalIndex = iota
	AlphaPhiB
	AlphaPhiC
	AlphaPhiPrimeA
	AlphaPhiPrimeB
	AlphaPhiPrimeC
	AlphaZ
	AlphaQM
	AlphaQL
	AlphaQR
	AlphaQO
	AlphaQC
	AlphaH
)

// XStarTerminalIndex fixes the array offsets at x_star.
type XStarTerminalIndex uint8

const (
	XStarPhiA XStarTerminalIndex = iota
	XStarPhiB
	XStarPhiC
	XStarPhiPrimeA
	XStarPhiPrimeB
	XStarPhiPrimeC
	XStarZ
)

// LocalPIOPTable is an explicit local Plonk table in natural domain order:
// entry j is the value at omega^j. SigmaX and SigmaPart contain the two
// destination coordinates bound by the circuit index. PublicInput is derived
// from the statement by a future backend adapter; it is explicit here so the
// local polynomial relation can be tested independently of gnark internals.
type LocalPIOPTable struct {
	Wires       [LocalWireCount][]fr.Element
	Selectors   [LocalSelectorCount][]fr.Element
	PublicInput []fr.Element
	SigmaX      [LocalWireCount][]fr.Element
	SigmaPart   [LocalWireCount][]fr.Element
}

// LocalPIOPIndex contains the public identity-tag data for one partition.
// Cross-partition injectivity of all destination coordinates is an index
// validation obligation and is outside this party-local algebra slice.
type LocalPIOPIndex struct {
	SlotLabel  fr.Element
	WireCosets [LocalWireCount]fr.Element
}

// LocalPIOPChallenges contains the public coins needed before alpha. A full
// backend must derive them from the ordered transcript rather than accept them
// from a caller.
type LocalPIOPChallenges struct {
	EtaPart fr.Element
	EtaX    fr.Element
	Gamma   fr.Element
	Lambda  fr.Element
}

// LocalPIOPRelation holds coefficient-form, degree-<T local oracles. The
// quotient chunks satisfy h=h_0+X^T h_1+X^(2T)h_2. BoundaryNumerator and
// BoundaryDenominator are u=z(x_star)f(x_star) and v=f'(x_star), while
// BlockFactor is bound by the division-free equation rho*v=u.
type LocalPIOPRelation struct {
	DomainSize int
	Omega      fr.Element
	XStar      fr.Element
	Index      LocalPIOPIndex
	Challenges LocalPIOPChallenges

	Wires       [LocalWireCount][]fr.Element
	Selectors   [LocalSelectorCount][]fr.Element
	PublicInput []fr.Element
	SigmaX      [LocalWireCount][]fr.Element
	SigmaPart   [LocalWireCount][]fr.Element

	Tags         [LocalWireCount][]fr.Element
	IdentityTags [LocalWireCount][]fr.Element
	Accumulator  []fr.Element
	Quotient     [3][]fr.Element

	EndpointCorrection  fr.Element
	BlockFactor         fr.Element
	BoundaryNumerator   fr.Element
	BoundaryDenominator fr.Element
}

// LocalTerminalEvaluations is the canonical 13/1/7 inventory from the PIOP
// reduction. Values within each array follow LocalTerminalFamilyOrder.
type LocalTerminalEvaluations struct {
	Alpha      [LocalTerminalAlphaCount]fr.Element
	OmegaAlpha [LocalTerminalOmegaAlphaCount]fr.Element
	XStar      [LocalTerminalXStarCount]fr.Element
}

// Flatten returns a fresh slice in alpha, omega*alpha, x_star order.
func (e LocalTerminalEvaluations) Flatten() []fr.Element {
	result := make([]fr.Element, 0, LocalTerminalTotalCount)
	result = append(result, e.Alpha[:]...)
	result = append(result, e.OmegaAlpha[:]...)
	result = append(result, e.XStar[:]...)
	return result
}

// ValuesAt returns a fresh slice in the canonical order for point.
func (e LocalTerminalEvaluations) ValuesAt(point LocalTerminalPoint) []fr.Element {
	switch point {
	case LocalTerminalAtAlpha:
		return append([]fr.Element(nil), e.Alpha[:]...)
	case LocalTerminalAtOmegaAlpha:
		return append([]fr.Element(nil), e.OmegaAlpha[:]...)
	case LocalTerminalAtXStar:
		return append([]fr.Element(nil), e.XStar[:]...)
	default:
		return nil
	}
}

// LocalQuotientEvaluationContext contains the public data needed to rebuild
// Delta_i(alpha) from the canonical terminal inventory.
type LocalQuotientEvaluationContext struct {
	DomainSize         int
	Omega              fr.Element
	XStar              fr.Element
	Index              LocalPIOPIndex
	Challenges         LocalPIOPChallenges
	PublicInputAtAlpha fr.Element
}

// LocalSourceCompression contains the three ordinary source polynomials and
// their claimed values at alpha, omega*alpha, and x_star. It is still clear
// field data: commitment authentication belongs to DLinKZG.
type LocalSourceCompression struct {
	Points      [LocalCompressedSourceCount]fr.Element
	Challenges  [LocalCompressedSourceCount]fr.Element
	Polynomials [LocalCompressedSourceCount][]fr.Element
	Values      [LocalCompressedSourceCount]fr.Element
}

// LocalTerminalFamilyOrder returns a copy of the canonical family order.
func LocalTerminalFamilyOrder(point LocalTerminalPoint) []LocalTerminalFamily {
	if int(point) >= len(localTerminalOrders) {
		return nil
	}
	return append([]LocalTerminalFamily(nil), localTerminalOrders[point]...)
}

// BuildLocalPIOPRelation interpolates an explicit row table, constructs the
// honest local accumulator, computes the endpoint-corrected numerator, and
// divides it by X^T-1 into exactly three degree-<T chunks. Division is used
// only to construct the honest accumulator and rho; VerifyLocalBlockFactor
// checks the exposed block relation without division.
func BuildLocalPIOPRelation(table LocalPIOPTable, index LocalPIOPIndex, challenges LocalPIOPChallenges) (*LocalPIOPRelation, error) {
	t, err := validateLocalPIOPTable(table)
	if err != nil {
		return nil, err
	}
	if err := validateLocalPIOPIndex(index, t); err != nil {
		return nil, err
	}

	domain := fft.NewDomain(uint64(t))
	relation := &LocalPIOPRelation{
		DomainSize: t,
		Omega:      domain.Generator,
		XStar:      domain.GeneratorInv,
		Index:      index,
		Challenges: challenges,
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		relation.Wires[wire] = interpolateLocalTable(table.Wires[wire], domain)
		relation.SigmaX[wire] = interpolateLocalTable(table.SigmaX[wire], domain)
		relation.SigmaPart[wire] = interpolateLocalTable(table.SigmaPart[wire], domain)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		relation.Selectors[selector] = interpolateLocalTable(table.Selectors[selector], domain)
	}
	relation.PublicInput = interpolateLocalTable(table.PublicInput, domain)

	tagEvaluations := [LocalWireCount][]fr.Element{}
	identityTagEvaluations := [LocalWireCount][]fr.Element{}
	for wire := 0; wire < LocalWireCount; wire++ {
		tagEvaluations[wire] = make([]fr.Element, t)
		identityTagEvaluations[wire] = make([]fr.Element, t)
	}

	x := fr.One()
	for row := 0; row < t; row++ {
		for wire := 0; wire < LocalWireCount; wire++ {
			tagEvaluations[wire][row] = table.Wires[wire][row]
			var term fr.Element
			term.Mul(&challenges.EtaPart, &table.SigmaPart[wire][row])
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
			term.Mul(&challenges.EtaX, &table.SigmaX[wire][row])
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &term)
			tagEvaluations[wire][row].Add(&tagEvaluations[wire][row], &challenges.Gamma)

			identityTagEvaluations[wire][row] = table.Wires[wire][row]
			term.Mul(&challenges.EtaPart, &index.SlotLabel)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
			term.Mul(&index.WireCosets[wire], &x).Mul(&term, &challenges.EtaX)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &term)
			identityTagEvaluations[wire][row].Add(&identityTagEvaluations[wire][row], &challenges.Gamma)
		}
		x.Mul(&x, &relation.Omega)
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		relation.Tags[wire] = interpolateLocalTable(tagEvaluations[wire], domain)
		relation.IdentityTags[wire] = interpolateLocalTable(identityTagEvaluations[wire], domain)
	}

	fEvaluations := multiplyLocalEvaluationFamilies(tagEvaluations, t)
	fPrimeEvaluations := multiplyLocalEvaluationFamilies(identityTagEvaluations, t)
	zEvaluations := make([]fr.Element, t)
	zEvaluations[0].SetOne()
	for row := 0; row < t-1; row++ {
		if fPrimeEvaluations[row].IsZero() {
			return nil, fmt.Errorf("%w at nonfinal row %d", ErrZeroLocalTagDenominator, row)
		}
		var inverseDenominator fr.Element
		inverseDenominator.Inverse(&fPrimeEvaluations[row])
		zEvaluations[row+1].Mul(&zEvaluations[row], &fEvaluations[row]).
			Mul(&zEvaluations[row+1], &inverseDenominator)
	}
	if fPrimeEvaluations[t-1].IsZero() {
		return nil, fmt.Errorf("%w at final row %d", ErrZeroLocalTagDenominator, t-1)
	}
	relation.Accumulator = interpolateLocalTable(zEvaluations, domain)

	core := deriveLocalPIOPCore(relation)
	relation.EndpointCorrection = core.endpoint
	relation.BoundaryNumerator = core.boundaryNumerator
	relation.BoundaryDenominator = core.boundaryDenominator
	relation.BlockFactor.Div(&relation.BoundaryNumerator, &relation.BoundaryDenominator)

	quotient, err := divideByLocalVanishing(core.numerator, t)
	if err != nil {
		return nil, err
	}
	for chunk := range relation.Quotient {
		relation.Quotient[chunk] = make([]fr.Element, t)
		for coefficient := 0; coefficient < t; coefficient++ {
			index := chunk*t + coefficient
			if index < len(quotient) {
				relation.Quotient[chunk][coefficient] = quotient[index]
			}
		}
	}

	if err := VerifyLocalPIOPRelation(relation); err != nil {
		return nil, err
	}
	return relation, nil
}

// VerifyLocalBlockFactor checks rho*v=u without inversion. This is the
// accepted relation even though the honest builder computes rho as u/v.
func VerifyLocalBlockFactor(rho, denominator, numerator fr.Element) error {
	var linked fr.Element
	linked.Mul(&rho, &denominator)
	if !linked.Equal(&numerator) {
		return ErrInvalidLocalBlockFactor
	}
	return nil
}

// VerifyLocalPIOPRelation checks the tag definitions, z(1)=1, endpoint
// correction, division-free block factor, and the exact three-chunk quotient
// identity. It does not validate global wiring injectivity or the product of
// block factors; those are index-validation and ProductCheck obligations.
func VerifyLocalPIOPRelation(relation *LocalPIOPRelation) error {
	if err := validateLocalPIOPRelationShape(relation); err != nil {
		return err
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		expectedTag := cloneLocalElements(relation.Wires[wire])
		addScaledLocalPolynomial(expectedTag, relation.SigmaPart[wire], relation.Challenges.EtaPart)
		addScaledLocalPolynomial(expectedTag, relation.SigmaX[wire], relation.Challenges.EtaX)
		expectedTag[0].Add(&expectedTag[0], &relation.Challenges.Gamma)
		if !equalLocalPolynomials(expectedTag, relation.Tags[wire]) {
			return fmt.Errorf("%w: destination tag %d is inconsistent", ErrInvalidLocalPIOPRelation, wire)
		}

		expectedIdentityTag := cloneLocalElements(relation.Wires[wire])
		var constant, linear fr.Element
		constant.Mul(&relation.Challenges.EtaPart, &relation.Index.SlotLabel)
		constant.Add(&constant, &relation.Challenges.Gamma)
		expectedIdentityTag[0].Add(&expectedIdentityTag[0], &constant)
		linear.Mul(&relation.Challenges.EtaX, &relation.Index.WireCosets[wire])
		expectedIdentityTag[1].Add(&expectedIdentityTag[1], &linear)
		if !equalLocalPolynomials(expectedIdentityTag, relation.IdentityTags[wire]) {
			return fmt.Errorf("%w: identity tag %d is inconsistent", ErrInvalidLocalPIOPRelation, wire)
		}
	}

	accumulatorAtOne := evaluateLocalPolynomial(relation.Accumulator, fr.One())
	one := fr.One()
	if !accumulatorAtOne.Equal(&one) {
		return fmt.Errorf("%w: accumulator does not satisfy z(1)=1", ErrInvalidLocalPIOPRelation)
	}

	core := deriveLocalPIOPCore(relation)
	if !core.endpoint.Equal(&relation.EndpointCorrection) {
		return fmt.Errorf("%w: endpoint correction is inconsistent", ErrInvalidLocalPIOPRelation)
	}
	if !core.boundaryNumerator.Equal(&relation.BoundaryNumerator) ||
		!core.boundaryDenominator.Equal(&relation.BoundaryDenominator) {
		return fmt.Errorf("%w: exposed boundary values are inconsistent", ErrInvalidLocalPIOPRelation)
	}
	if err := VerifyLocalBlockFactor(relation.BlockFactor, relation.BoundaryDenominator, relation.BoundaryNumerator); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidLocalPIOPRelation, err)
	}

	expectedQuotient, err := divideByLocalVanishing(core.numerator, relation.DomainSize)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidLocalPIOPRelation, err)
	}
	for chunk := range relation.Quotient {
		for coefficient := 0; coefficient < relation.DomainSize; coefficient++ {
			index := chunk*relation.DomainSize + coefficient
			var expected fr.Element
			if index < len(expectedQuotient) {
				expected = expectedQuotient[index]
			}
			if !expected.Equal(&relation.Quotient[chunk][coefficient]) {
				return fmt.Errorf("%w: quotient chunk %d coefficient %d is inconsistent", ErrInvalidLocalPIOPRelation, chunk, coefficient)
			}
		}
	}
	for index := 3 * relation.DomainSize; index < len(expectedQuotient); index++ {
		if !expectedQuotient[index].IsZero() {
			return fmt.Errorf("%w: quotient exceeds three chunks", ErrInvalidLocalPIOPRelation)
		}
	}
	return nil
}

// TerminalEvaluations evaluates the exact 13/1/7 terminal inventory. Alpha
// must lie outside the local domain and must be nonzero.
func (relation *LocalPIOPRelation) TerminalEvaluations(alpha fr.Element) (LocalTerminalEvaluations, error) {
	var result LocalTerminalEvaluations
	if err := validateLocalTerminalPoint(relation, alpha); err != nil {
		return result, err
	}

	for wire := 0; wire < LocalWireCount; wire++ {
		result.Alpha[int(AlphaPhiA)+wire] = evaluateLocalPolynomial(relation.Tags[wire], alpha)
		result.Alpha[int(AlphaPhiPrimeA)+wire] = evaluateLocalPolynomial(relation.IdentityTags[wire], alpha)
		result.XStar[int(XStarPhiA)+wire] = evaluateLocalPolynomial(relation.Tags[wire], relation.XStar)
		result.XStar[int(XStarPhiPrimeA)+wire] = evaluateLocalPolynomial(relation.IdentityTags[wire], relation.XStar)
	}
	result.Alpha[AlphaZ] = evaluateLocalPolynomial(relation.Accumulator, alpha)
	for selector := 0; selector < LocalSelectorCount; selector++ {
		result.Alpha[int(AlphaQM)+selector] = evaluateLocalPolynomial(relation.Selectors[selector], alpha)
	}

	hAlpha := combineLocalQuotient(relation.Quotient, alpha, relation.DomainSize)
	result.Alpha[AlphaH] = evaluateLocalPolynomial(hAlpha, alpha)

	var omegaAlpha fr.Element
	omegaAlpha.Mul(&relation.Omega, &alpha)
	result.OmegaAlpha[0] = evaluateLocalPolynomial(relation.Accumulator, omegaAlpha)
	result.XStar[XStarZ] = evaluateLocalPolynomial(relation.Accumulator, relation.XStar)
	return result, nil
}

// EvaluateLocalQuotientResidual reconstructs Delta_i(alpha) solely from the
// canonical terminal values and public context. It is the scalar equation an
// honest coordinator can check before SumCheck/source compression.
func EvaluateLocalQuotientResidual(terminals LocalTerminalEvaluations, alpha fr.Element, context LocalQuotientEvaluationContext) (fr.Element, error) {
	if context.DomainSize < 2 || !isPowerOfTwo(context.DomainSize) {
		return fr.Element{}, fmt.Errorf("%w: invalid domain size %d", ErrInvalidLocalTerminalPoint, context.DomainSize)
	}
	domain := fft.NewDomain(uint64(context.DomainSize))
	if !context.Omega.Equal(&domain.Generator) || !context.XStar.Equal(&domain.GeneratorInv) {
		return fr.Element{}, fmt.Errorf("%w: noncanonical local domain", ErrInvalidLocalTerminalPoint)
	}
	if err := validateAlphaOutsideLocalDomain(alpha, context.DomainSize); err != nil {
		return fr.Element{}, err
	}

	var wires [LocalWireCount]fr.Element
	for wire := 0; wire < LocalWireCount; wire++ {
		wires[wire] = terminals.Alpha[int(AlphaPhiPrimeA)+wire]
		var term fr.Element
		term.Mul(&context.Challenges.EtaPart, &context.Index.SlotLabel)
		wires[wire].Sub(&wires[wire], &term)
		term.Mul(&context.Index.WireCosets[wire], &alpha).Mul(&term, &context.Challenges.EtaX)
		wires[wire].Sub(&wires[wire], &term)
		wires[wire].Sub(&wires[wire], &context.Challenges.Gamma)
	}

	var gate, term fr.Element
	term.Mul(&wires[LocalWireA], &wires[LocalWireB]).
		Mul(&term, &terminals.Alpha[AlphaQM])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireA], &terminals.Alpha[AlphaQL])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireB], &terminals.Alpha[AlphaQR])
	gate.Add(&gate, &term)
	term.Mul(&wires[LocalWireC], &terminals.Alpha[AlphaQO])
	gate.Add(&gate, &term)
	gate.Add(&gate, &terminals.Alpha[AlphaQC])
	gate.Add(&gate, &context.PublicInputAtAlpha)

	fAlpha := multiplyThreeLocalScalars(
		terminals.Alpha[AlphaPhiA], terminals.Alpha[AlphaPhiB], terminals.Alpha[AlphaPhiC],
	)
	fPrimeAlpha := multiplyThreeLocalScalars(
		terminals.Alpha[AlphaPhiPrimeA], terminals.Alpha[AlphaPhiPrimeB], terminals.Alpha[AlphaPhiPrimeC],
	)
	fXStar := multiplyThreeLocalScalars(
		terminals.XStar[XStarPhiA], terminals.XStar[XStarPhiB], terminals.XStar[XStarPhiC],
	)
	fPrimeXStar := multiplyThreeLocalScalars(
		terminals.XStar[XStarPhiPrimeA], terminals.XStar[XStarPhiPrimeB], terminals.XStar[XStarPhiPrimeC],
	)

	one := fr.One()
	alphaT := powerLocalField(alpha, context.DomainSize)
	alphaTMinusOne := alphaT
	alphaTMinusOne.Sub(&alphaTMinusOne, &one)
	tAsField := fr.NewElement(uint64(context.DomainSize))

	var denominator, lZero, lStar fr.Element
	denominator.Sub(&alpha, &one).Mul(&denominator, &tAsField)
	lZero.Div(&alphaTMinusOne, &denominator)
	denominator.Sub(&alpha, &context.XStar).Mul(&denominator, &tAsField)
	lStar.Mul(&context.XStar, &alphaTMinusOne).Div(&lStar, &denominator)

	var boundary, transition, endpoint fr.Element
	boundary.Sub(&terminals.Alpha[AlphaZ], &one).Mul(&boundary, &lZero)
	transition.Mul(&terminals.Alpha[AlphaZ], &fAlpha)
	term.Mul(&terminals.OmegaAlpha[0], &fPrimeAlpha)
	transition.Sub(&transition, &term)
	endpoint.Mul(&terminals.XStar[XStarZ], &fXStar).Sub(&endpoint, &fPrimeXStar)
	term.Mul(&lStar, &endpoint)
	transition.Sub(&transition, &term)

	var residual fr.Element
	residual = gate
	term.Mul(&context.Challenges.Lambda, &boundary)
	residual.Add(&residual, &term)
	var lambdaSquared fr.Element
	lambdaSquared.Square(&context.Challenges.Lambda)
	term.Mul(&lambdaSquared, &transition)
	residual.Add(&residual, &term)
	term.Mul(&alphaTMinusOne, &terminals.Alpha[AlphaH])
	residual.Sub(&residual, &term)
	return residual, nil
}

// QuotientResidualAt evaluates the relation's own public-input polynomial and
// reconstructs Delta_i(alpha) from its terminal inventory.
func (relation *LocalPIOPRelation) QuotientResidualAt(alpha fr.Element) (fr.Element, error) {
	terminals, err := relation.TerminalEvaluations(alpha)
	if err != nil {
		return fr.Element{}, err
	}
	context := LocalQuotientEvaluationContext{
		DomainSize:         relation.DomainSize,
		Omega:              relation.Omega,
		XStar:              relation.XStar,
		Index:              relation.Index,
		Challenges:         relation.Challenges,
		PublicInputAtAlpha: evaluateLocalPolynomial(relation.PublicInput, alpha),
	}
	return EvaluateLocalQuotientResidual(terminals, alpha, context)
}

// CompressLocalTerminalSources applies independent powers of mu to the
// canonical 13/1/7 polynomial groups. It returns the three degree-<T source
// polynomials p_{j,i} and their evaluations w_{j,i}.
func CompressLocalTerminalSources(relation *LocalPIOPRelation, alpha fr.Element, challenges [LocalCompressedSourceCount]fr.Element) (LocalSourceCompression, error) {
	var result LocalSourceCompression
	terminals, err := relation.TerminalEvaluations(alpha)
	if err != nil {
		return result, err
	}
	groups := terminalLocalPolynomialGroups(relation, alpha)

	result.Points[0] = alpha
	result.Points[1].Mul(&relation.Omega, &alpha)
	result.Points[2] = relation.XStar
	result.Challenges = challenges
	for group := 0; group < LocalCompressedSourceCount; group++ {
		result.Polynomials[group] = make([]fr.Element, relation.DomainSize)
		values := terminals.ValuesAt(LocalTerminalPoint(group))
		power := fr.One()
		for family := range groups[group] {
			addScaledLocalPolynomial(result.Polynomials[group], groups[group][family], power)
			var term fr.Element
			term.Mul(&power, &values[family])
			result.Values[group].Add(&result.Values[group], &term)
			power.Mul(&power, &challenges[group])
		}
	}
	return result, nil
}

// VerifyLocalTerminalSourceCompression recomputes the exact ordered source
// compression. This is a clear-algebra check, not PCS authentication.
func VerifyLocalTerminalSourceCompression(relation *LocalPIOPRelation, alpha fr.Element, challenges [LocalCompressedSourceCount]fr.Element, compression LocalSourceCompression) error {
	expected, err := CompressLocalTerminalSources(relation, alpha, challenges)
	if err != nil {
		return err
	}
	for group := 0; group < LocalCompressedSourceCount; group++ {
		if !expected.Points[group].Equal(&compression.Points[group]) ||
			!expected.Challenges[group].Equal(&compression.Challenges[group]) ||
			!expected.Values[group].Equal(&compression.Values[group]) ||
			!equalLocalPolynomials(expected.Polynomials[group], compression.Polynomials[group]) {
			return fmt.Errorf("%w: group %d differs from the canonical compression", ErrInvalidLocalSourceCompression, group)
		}
		openedValue := evaluateLocalPolynomial(compression.Polynomials[group], compression.Points[group])
		if !openedValue.Equal(&compression.Values[group]) {
			return fmt.Errorf("%w: group %d evaluation is inconsistent", ErrInvalidLocalSourceCompression, group)
		}
	}
	return nil
}

type localPIOPCore struct {
	endpoint            fr.Element
	boundaryNumerator   fr.Element
	boundaryDenominator fr.Element
	numerator           []fr.Element
}

func deriveLocalPIOPCore(relation *LocalPIOPRelation) localPIOPCore {
	t := relation.DomainSize
	f := multiplyThreeLocalPolynomials(relation.Tags[0], relation.Tags[1], relation.Tags[2])
	fPrime := multiplyThreeLocalPolynomials(relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2])

	zAtXStar := evaluateLocalPolynomial(relation.Accumulator, relation.XStar)
	fAtXStar := evaluateLocalPolynomial(f, relation.XStar)
	fPrimeAtXStar := evaluateLocalPolynomial(fPrime, relation.XStar)
	var core localPIOPCore
	core.boundaryNumerator.Mul(&zAtXStar, &fAtXStar)
	core.boundaryDenominator = fPrimeAtXStar
	core.endpoint.Sub(&core.boundaryNumerator, &core.boundaryDenominator)

	gate := multiplyThreeLocalPolynomials(relation.Selectors[LocalSelectorM], relation.Wires[LocalWireA], relation.Wires[LocalWireB])
	addLocalPolynomial(gate, multiplyLocalPolynomials(relation.Selectors[LocalSelectorL], relation.Wires[LocalWireA]))
	addLocalPolynomial(gate, multiplyLocalPolynomials(relation.Selectors[LocalSelectorR], relation.Wires[LocalWireB]))
	addLocalPolynomial(gate, multiplyLocalPolynomials(relation.Selectors[LocalSelectorO], relation.Wires[LocalWireC]))
	addLocalPolynomial(gate, relation.Selectors[LocalSelectorC])
	addLocalPolynomial(gate, relation.PublicInput)

	domain := fft.NewDomain(uint64(t))
	firstRow := make([]fr.Element, t)
	firstRow[0].SetOne()
	lastRow := make([]fr.Element, t)
	lastRow[t-1].SetOne()
	lZero := interpolateLocalTable(firstRow, domain)
	lStar := interpolateLocalTable(lastRow, domain)

	zMinusOne := cloneLocalElements(relation.Accumulator)
	one := fr.One()
	zMinusOne[0].Sub(&zMinusOne[0], &one)
	pZero := multiplyLocalPolynomials(lZero, zMinusOne)

	zOmega := make([]fr.Element, len(relation.Accumulator))
	power := fr.One()
	for coefficient := range relation.Accumulator {
		zOmega[coefficient].Mul(&relation.Accumulator[coefficient], &power)
		power.Mul(&power, &relation.Omega)
	}
	transition := multiplyLocalPolynomials(relation.Accumulator, f)
	subtractLocalPolynomial(transition, multiplyLocalPolynomials(zOmega, fPrime))
	endpointTerm := cloneLocalElements(lStar)
	for coefficient := range endpointTerm {
		endpointTerm[coefficient].Mul(&endpointTerm[coefficient], &core.endpoint)
	}
	subtractLocalPolynomial(transition, endpointTerm)

	core.numerator = cloneLocalElements(gate)
	addScaledLocalPolynomialGrowing(&core.numerator, pZero, relation.Challenges.Lambda)
	var lambdaSquared fr.Element
	lambdaSquared.Square(&relation.Challenges.Lambda)
	addScaledLocalPolynomialGrowing(&core.numerator, transition, lambdaSquared)
	return core
}

func validateLocalPIOPTable(table LocalPIOPTable) (int, error) {
	t := len(table.Wires[0])
	if t < 2 || !isPowerOfTwo(t) {
		return 0, fmt.Errorf("%w: row count %d is not a power of two at least two", ErrInvalidLocalPIOPTable, t)
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if len(table.Wires[wire]) != t || len(table.SigmaX[wire]) != t || len(table.SigmaPart[wire]) != t {
			return 0, fmt.Errorf("%w: wire or sigma table %d does not have T=%d rows", ErrInvalidLocalPIOPTable, wire, t)
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if len(table.Selectors[selector]) != t {
			return 0, fmt.Errorf("%w: selector table %d does not have T=%d rows", ErrInvalidLocalPIOPTable, selector, t)
		}
	}
	if len(table.PublicInput) != t {
		return 0, fmt.Errorf("%w: public-input table does not have T=%d rows", ErrInvalidLocalPIOPTable, t)
	}
	return t, nil
}

func validateLocalPIOPIndex(index LocalPIOPIndex, t int) error {
	for wire := 0; wire < LocalWireCount; wire++ {
		if index.WireCosets[wire].IsZero() {
			return fmt.Errorf("%w: wire coset %d is zero", ErrInvalidLocalPIOPTable, wire)
		}
		for previous := 0; previous < wire; previous++ {
			wireCosetPower := powerLocalField(index.WireCosets[wire], t)
			previousCosetPower := powerLocalField(index.WireCosets[previous], t)
			if wireCosetPower.Equal(&previousCosetPower) {
				return fmt.Errorf("%w: wire representatives %d and %d name the same H_X coset", ErrInvalidLocalPIOPTable, previous, wire)
			}
		}
	}
	return nil
}

func validateLocalPIOPRelationShape(relation *LocalPIOPRelation) error {
	if relation == nil || relation.DomainSize < 2 || !isPowerOfTwo(relation.DomainSize) {
		return fmt.Errorf("%w: nil relation or invalid domain size", ErrInvalidLocalPIOPRelation)
	}
	if err := validateLocalPIOPIndex(relation.Index, relation.DomainSize); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidLocalPIOPRelation, err)
	}
	domain := fft.NewDomain(uint64(relation.DomainSize))
	if !relation.Omega.Equal(&domain.Generator) || !relation.XStar.Equal(&domain.GeneratorInv) {
		return fmt.Errorf("%w: noncanonical omega or x_star", ErrInvalidLocalPIOPRelation)
	}
	t := relation.DomainSize
	for wire := 0; wire < LocalWireCount; wire++ {
		if len(relation.Wires[wire]) != t || len(relation.SigmaX[wire]) != t || len(relation.SigmaPart[wire]) != t ||
			len(relation.Tags[wire]) != t || len(relation.IdentityTags[wire]) != t {
			return fmt.Errorf("%w: wire-family %d does not have degree-bound width T", ErrInvalidLocalPIOPRelation, wire)
		}
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if len(relation.Selectors[selector]) != t {
			return fmt.Errorf("%w: selector %d does not have degree-bound width T", ErrInvalidLocalPIOPRelation, selector)
		}
	}
	if len(relation.PublicInput) != t || len(relation.Accumulator) != t {
		return fmt.Errorf("%w: public-input or accumulator width is not T", ErrInvalidLocalPIOPRelation)
	}
	for chunk := range relation.Quotient {
		if len(relation.Quotient[chunk]) != t {
			return fmt.Errorf("%w: quotient chunk %d width is not T", ErrInvalidLocalPIOPRelation, chunk)
		}
	}
	return nil
}

func validateLocalTerminalPoint(relation *LocalPIOPRelation, alpha fr.Element) error {
	if err := validateLocalPIOPRelationShape(relation); err != nil {
		return err
	}
	return validateAlphaOutsideLocalDomain(alpha, relation.DomainSize)
}

func validateAlphaOutsideLocalDomain(alpha fr.Element, t int) error {
	if alpha.IsZero() {
		return fmt.Errorf("%w: alpha is zero", ErrInvalidLocalTerminalPoint)
	}
	alphaT := powerLocalField(alpha, t)
	one := fr.One()
	if alphaT.Equal(&one) {
		return fmt.Errorf("%w: alpha lies in the local domain", ErrInvalidLocalTerminalPoint)
	}
	return nil
}

func terminalLocalPolynomialGroups(relation *LocalPIOPRelation, alpha fr.Element) [LocalCompressedSourceCount][][]fr.Element {
	var groups [LocalCompressedSourceCount][][]fr.Element
	hAlpha := combineLocalQuotient(relation.Quotient, alpha, relation.DomainSize)
	groups[0] = [][]fr.Element{
		relation.Tags[0], relation.Tags[1], relation.Tags[2],
		relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2],
		relation.Accumulator,
		relation.Selectors[0], relation.Selectors[1], relation.Selectors[2], relation.Selectors[3], relation.Selectors[4],
		hAlpha,
	}
	groups[1] = [][]fr.Element{relation.Accumulator}
	groups[2] = [][]fr.Element{
		relation.Tags[0], relation.Tags[1], relation.Tags[2],
		relation.IdentityTags[0], relation.IdentityTags[1], relation.IdentityTags[2],
		relation.Accumulator,
	}
	return groups
}

func combineLocalQuotient(chunks [3][]fr.Element, alpha fr.Element, t int) []fr.Element {
	result := make([]fr.Element, t)
	alphaT := powerLocalField(alpha, t)
	alphaTwoT := alphaT
	alphaTwoT.Square(&alphaTwoT)
	addScaledLocalPolynomial(result, chunks[0], fr.One())
	addScaledLocalPolynomial(result, chunks[1], alphaT)
	addScaledLocalPolynomial(result, chunks[2], alphaTwoT)
	return result
}

func multiplyLocalEvaluationFamilies(families [LocalWireCount][]fr.Element, size int) []fr.Element {
	result := make([]fr.Element, size)
	for row := range result {
		result[row].SetOne()
		for wire := 0; wire < LocalWireCount; wire++ {
			result[row].Mul(&result[row], &families[wire][row])
		}
	}
	return result
}

func interpolateLocalTable(evaluations []fr.Element, domain *fft.Domain) []fr.Element {
	coefficients := cloneLocalElements(evaluations)
	domain.FFTInverse(coefficients, fft.DIF)
	fft.BitReverse(coefficients)
	return coefficients
}

func evaluateLocalPolynomial(coefficients []fr.Element, point fr.Element) fr.Element {
	var result fr.Element
	for coefficient := len(coefficients) - 1; coefficient >= 0; coefficient-- {
		result.Mul(&result, &point)
		result.Add(&result, &coefficients[coefficient])
	}
	return result
}

func multiplyLocalPolynomials(left, right []fr.Element) []fr.Element {
	if len(left) == 0 || len(right) == 0 {
		return nil
	}
	result := make([]fr.Element, len(left)+len(right)-1)
	for i := range left {
		for j := range right {
			var term fr.Element
			term.Mul(&left[i], &right[j])
			result[i+j].Add(&result[i+j], &term)
		}
	}
	return result
}

func multiplyThreeLocalPolynomials(first, second, third []fr.Element) []fr.Element {
	return multiplyLocalPolynomials(multiplyLocalPolynomials(first, second), third)
}

func multiplyThreeLocalScalars(first, second, third fr.Element) fr.Element {
	var result fr.Element
	result.Mul(&first, &second).Mul(&result, &third)
	return result
}

func addLocalPolynomial(destination, source []fr.Element) {
	for coefficient := range source {
		destination[coefficient].Add(&destination[coefficient], &source[coefficient])
	}
}

func subtractLocalPolynomial(destination, source []fr.Element) {
	for coefficient := range source {
		destination[coefficient].Sub(&destination[coefficient], &source[coefficient])
	}
}

func addScaledLocalPolynomial(destination, source []fr.Element, scale fr.Element) {
	for coefficient := range source {
		var term fr.Element
		term.Mul(&source[coefficient], &scale)
		destination[coefficient].Add(&destination[coefficient], &term)
	}
}

func addScaledLocalPolynomialGrowing(destination *[]fr.Element, source []fr.Element, scale fr.Element) {
	if len(*destination) < len(source) {
		*destination = append(*destination, make([]fr.Element, len(source)-len(*destination))...)
	}
	addScaledLocalPolynomial(*destination, source, scale)
}

func divideByLocalVanishing(numerator []fr.Element, t int) ([]fr.Element, error) {
	remainder := cloneLocalElements(numerator)
	if len(remainder) < t {
		remainder = append(remainder, make([]fr.Element, t-len(remainder))...)
	}
	quotientLength := len(remainder) - t
	if quotientLength < 0 {
		quotientLength = 0
	}
	quotient := make([]fr.Element, quotientLength)
	for degree := len(remainder) - 1; degree >= t; degree-- {
		coefficient := remainder[degree]
		if coefficient.IsZero() {
			continue
		}
		quotient[degree-t] = coefficient
		remainder[degree].SetZero()
		remainder[degree-t].Add(&remainder[degree-t], &coefficient)
	}
	for degree := 0; degree < t && degree < len(remainder); degree++ {
		if !remainder[degree].IsZero() {
			return nil, fmt.Errorf("%w: nonzero remainder coefficient %d", ErrLocalQuotientNotDivisible, degree)
		}
	}
	return quotient, nil
}

func equalLocalPolynomials(left, right []fr.Element) bool {
	if len(left) != len(right) {
		return false
	}
	for coefficient := range left {
		if !left[coefficient].Equal(&right[coefficient]) {
			return false
		}
	}
	return true
}

func cloneLocalElements(elements []fr.Element) []fr.Element {
	return append([]fr.Element(nil), elements...)
}

func powerLocalField(base fr.Element, exponent int) fr.Element {
	var result fr.Element
	result.Exp(base, new(big.Int).SetUint64(uint64(exponent)))
	return result
}
