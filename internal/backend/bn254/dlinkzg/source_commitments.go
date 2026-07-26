package dlinkzg

// This file is the commitment-only bridge between the row-partitioned Plonk
// PIOP and the three terminal sources consumed by DLinKZG. Fixed commitments
// are produced offline from LocalPIOPPreprocessing. Online, the three source
// commitments are public linear combinations of W0, W1, W2, and that fixed
// index; no party sends another source-commitment message after mu is known.

import (
	"errors"
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptodlinkzg "github.com/consensys/gnark-crypto/ecc/bn254/fr/dlinkzg"
)

const (
	// LocalPIOPFixedCommitmentsPerSlot is the exact authenticated index share:
	// five selectors, three sigma-X columns, three sigma-part columns, One,
	// and X. Upsilon_i is the public scalar multiple SlotLabel*One.
	LocalPIOPFixedCommitmentsPerSlot = LocalSelectorCount + 2*LocalWireCount + 2

	// LocalPIOPSourceCommitmentNotice fixes the message boundary of this slice.
	LocalPIOPSourceCommitmentNotice = "offline 13-per-slot fixed index and public W0/W1/W2-to-13/1/7 terminal source commitment derivation; no extra proof message"
)

var (
	ErrInvalidLocalPIOPFixedCommitments  = errors.New("dlinkzg: invalid local PIOP fixed commitments")
	ErrInvalidLocalPIOPPublicCommitments = errors.New("dlinkzg: invalid aggregate local PIOP public commitments")
)

// LocalPIOPFixedCommitmentSlot is one manifest slot's exact thirteen-element
// fixed commitment vector. Metadata is public and binds the vector to its
// ordered row-SRS view; it is not an additional group element.
type LocalPIOPFixedCommitmentSlot struct {
	Rank       int
	Parties    int
	DomainSize int
	SlotLabel  fr.Element

	Selectors [LocalSelectorCount]bn254.G1Affine
	SigmaX    [LocalWireCount]bn254.G1Affine
	SigmaPart [LocalWireCount]bn254.G1Affine
	One       bn254.G1Affine
	X         bn254.G1Affine
}

// LocalPIOPAggregatePublicFamilies are the three public-table commitments
// used by the identity tags. Upsilon is derived from the ordered slot vector
// as sum_i SlotLabel_i*Com(Y^i); it is not a fourteenth per-slot commitment.
type LocalPIOPAggregatePublicFamilies struct {
	One     bn254.G1Affine
	Upsilon bn254.G1Affine
	X       bn254.G1Affine
}

// AggregateLocalPIOPFixedCommitments is the verifier's aggregate fixed view.
// Each field is the sum of the corresponding manifest-ordered slot shares.
type AggregateLocalPIOPFixedCommitments struct {
	Parties    int
	DomainSize int

	Selectors [LocalSelectorCount]bn254.G1Affine
	SigmaX    [LocalWireCount]bn254.G1Affine
	SigmaPart [LocalWireCount]bn254.G1Affine
	Public    LocalPIOPAggregatePublicFamilies
}

// AggregateLocalPIOPPublicCommitments contains exactly the public W0--W2
// commitment view: three witnesses, one accumulator, and three quotient
// chunks. Parties and DomainSize bind it to the fixed view.
type AggregateLocalPIOPPublicCommitments struct {
	Parties    int
	DomainSize int
	W0         [LocalWireCount]bn254.G1Affine
	W1         bn254.G1Affine
	W2         [3]bn254.G1Affine
}

// AggregateLocalPIOPCommitmentView is all group data needed to derive the
// terminal source commitments after mu. It contains no W3 evaluations.
type AggregateLocalPIOPCommitmentView struct {
	Fixed  AggregateLocalPIOPFixedCommitments
	Public AggregateLocalPIOPPublicCommitments
}

// LocalTerminalSourceCommitmentChallenges are the post-W2 public scalars
// needed for source construction. Mu follows alpha, omega*alpha, x_star order.
type LocalTerminalSourceCommitmentChallenges struct {
	EtaPart fr.Element
	EtaX    fr.Element
	Gamma   fr.Element
	Alpha   fr.Element
	Mu      [LocalCompressedSourceCount]fr.Element
}

// LocalTerminalSourceCommitments follows the canonical 13/1/7 point order:
// alpha, omega*alpha, and x_star.
type LocalTerminalSourceCommitments struct {
	Commitments [LocalCompressedSourceCount]bn254.G1Affine
}

// SetupLocalPIOPFixedCommitmentSlot commits one preprocessing shard in the
// semantic X coordinate. Interpolation and these thirteen MSMs are offline;
// neither witness wires nor public-input values enter this function.
func SetupLocalPIOPFixedCommitmentSlot(
	preprocessing *LocalPIOPPreprocessing,
	rowSRS *cryptodlinkzg.PartyRowSRS,
) (LocalPIOPFixedCommitmentSlot, error) {
	var result LocalPIOPFixedCommitmentSlot
	if preprocessing == nil {
		return result, fmt.Errorf("%w: nil preprocessing", ErrInvalidLocalPIOPFixedCommitments)
	}
	if err := rowSRS.Validate(); err != nil {
		return result, fmt.Errorf("%w: party row SRS: %v", ErrInvalidLocalPIOPFixedCommitments, err)
	}
	if preprocessing.LocalRows() < 4 || preprocessing.World() > preprocessing.LocalRows() {
		return result, fmt.Errorf(
			"%w: protocol shape requires T>=4 and M<=T, got M=%d,T=%d",
			ErrInvalidLocalPIOPFixedCommitments, preprocessing.World(), preprocessing.LocalRows(),
		)
	}
	if rowSRS.Rank != preprocessing.Rank() || rowSRS.Parties != preprocessing.World() ||
		rowSRS.DegreeBound < preprocessing.LocalRows() {
		return result, fmt.Errorf(
			"%w: preprocessing (%d/%d,T=%d) does not match row SRS (%d/%d,bound=%d)",
			ErrInvalidLocalPIOPFixedCommitments,
			preprocessing.Rank(), preprocessing.World(), preprocessing.LocalRows(),
			rowSRS.Rank, rowSRS.Parties, rowSRS.DegreeBound,
		)
	}

	fixed, err := PreprocessFastLocalPIOPFixed(preprocessing.FixedTable(), preprocessing.Index())
	if err != nil {
		return result, fmt.Errorf("%w: fixed polynomial interpolation: %v", ErrInvalidLocalPIOPFixedCommitments, err)
	}
	result.Rank = preprocessing.Rank()
	result.Parties = preprocessing.World()
	result.DomainSize = preprocessing.LocalRows()
	result.SlotLabel = preprocessing.Index().SlotLabel
	for selector := 0; selector < LocalSelectorCount; selector++ {
		result.Selectors[selector], err = rowSRS.CommitSemantic(fixed.selectors[selector])
		if err != nil {
			return LocalPIOPFixedCommitmentSlot{}, fmt.Errorf("%w: selector %d: %v", ErrInvalidLocalPIOPFixedCommitments, selector, err)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		result.SigmaX[wire], err = rowSRS.CommitSemantic(fixed.sigmaX[wire])
		if err != nil {
			return LocalPIOPFixedCommitmentSlot{}, fmt.Errorf("%w: sigma-X %d: %v", ErrInvalidLocalPIOPFixedCommitments, wire, err)
		}
		result.SigmaPart[wire], err = rowSRS.CommitSemantic(fixed.sigmaPart[wire])
		if err != nil {
			return LocalPIOPFixedCommitmentSlot{}, fmt.Errorf("%w: sigma-part %d: %v", ErrInvalidLocalPIOPFixedCommitments, wire, err)
		}
	}
	result.One, err = rowSRS.CommitSemantic([]fr.Element{fr.One()})
	if err != nil {
		return LocalPIOPFixedCommitmentSlot{}, fmt.Errorf("%w: One: %v", ErrInvalidLocalPIOPFixedCommitments, err)
	}
	xPolynomial := []fr.Element{{}, fr.One()}
	result.X, err = rowSRS.CommitSemantic(xPolynomial)
	if err != nil {
		return LocalPIOPFixedCommitmentSlot{}, fmt.Errorf("%w: X: %v", ErrInvalidLocalPIOPFixedCommitments, err)
	}
	if err := validateLocalPIOPFixedCommitmentSlot(result); err != nil {
		return LocalPIOPFixedCommitmentSlot{}, err
	}
	return result, nil
}

// AggregateLocalPIOPFixedCommitmentSlots checks exact manifest order and
// sums the thirteen slot vectors. The public Upsilon commitment is derived
// from One shares and the canonical rank field embeddings.
func AggregateLocalPIOPFixedCommitmentSlots(
	slots []LocalPIOPFixedCommitmentSlot,
) (AggregateLocalPIOPFixedCommitments, error) {
	var result AggregateLocalPIOPFixedCommitments
	if len(slots) < 2 || !isPowerOfTwo(len(slots)) {
		return result, fmt.Errorf("%w: slot count %d is not a power of two at least two", ErrInvalidLocalPIOPFixedCommitments, len(slots))
	}
	result.Parties = len(slots)
	result.DomainSize = slots[0].DomainSize
	for rank := range slots {
		if err := validateLocalPIOPFixedCommitmentSlot(slots[rank]); err != nil {
			return AggregateLocalPIOPFixedCommitments{}, fmt.Errorf("%w: slot %d: %v", ErrInvalidLocalPIOPFixedCommitments, rank, err)
		}
		if slots[rank].Rank != rank || slots[rank].Parties != len(slots) || slots[rank].DomainSize != result.DomainSize {
			return AggregateLocalPIOPFixedCommitments{}, fmt.Errorf(
				"%w: slot %d metadata is (%d/%d,T=%d), want (%d/%d,T=%d)",
				ErrInvalidLocalPIOPFixedCommitments,
				rank, slots[rank].Rank, slots[rank].Parties, slots[rank].DomainSize,
				rank, len(slots), result.DomainSize,
			)
		}
		for selector := 0; selector < LocalSelectorCount; selector++ {
			sourceCommitmentAdd(&result.Selectors[selector], &slots[rank].Selectors[selector])
		}
		for wire := 0; wire < LocalWireCount; wire++ {
			sourceCommitmentAdd(&result.SigmaX[wire], &slots[rank].SigmaX[wire])
			sourceCommitmentAdd(&result.SigmaPart[wire], &slots[rank].SigmaPart[wire])
		}
		sourceCommitmentAdd(&result.Public.One, &slots[rank].One)
		sourceCommitmentAdd(&result.Public.X, &slots[rank].X)
		sourceCommitmentAddScaled(&result.Public.Upsilon, &slots[rank].One, slots[rank].SlotLabel)
	}
	if err := validateAggregateLocalPIOPFixedCommitments(result); err != nil {
		return AggregateLocalPIOPFixedCommitments{}, err
	}
	return result, nil
}

// DeriveLocalTerminalSourceCommitments constructs the three compressed
// rectangular source commitments by public group arithmetic. The polynomial
// families are exactly 13/1/7 and use the ordering in LocalTerminalFamilyOrder.
func DeriveLocalTerminalSourceCommitments(
	view AggregateLocalPIOPCommitmentView,
	domainSize int,
	wireCosets [LocalWireCount]fr.Element,
	challenges LocalTerminalSourceCommitmentChallenges,
) (LocalTerminalSourceCommitments, error) {
	var result LocalTerminalSourceCommitments
	if err := validateAggregateLocalPIOPCommitmentView(view, domainSize, wireCosets); err != nil {
		return result, err
	}
	if err := validateAlphaOutsideLocalDomain(challenges.Alpha, domainSize); err != nil {
		return result, err
	}

	var phi, phiPrime [LocalWireCount]bn254.G1Affine
	var err error
	for wire := 0; wire < LocalWireCount; wire++ {
		phi[wire], err = sourceCommitmentLinearCombination(
			[]bn254.G1Affine{
				view.Public.W0[wire],
				view.Fixed.SigmaPart[wire],
				view.Fixed.SigmaX[wire],
				view.Fixed.Public.One,
			},
			[]fr.Element{fr.One(), challenges.EtaPart, challenges.EtaX, challenges.Gamma},
		)
		if err != nil {
			return result, err
		}
		var etaXCoset fr.Element
		etaXCoset.Mul(&challenges.EtaX, &wireCosets[wire])
		phiPrime[wire], err = sourceCommitmentLinearCombination(
			[]bn254.G1Affine{
				view.Public.W0[wire],
				view.Fixed.Public.Upsilon,
				view.Fixed.Public.X,
				view.Fixed.Public.One,
			},
			[]fr.Element{fr.One(), challenges.EtaPart, etaXCoset, challenges.Gamma},
		)
		if err != nil {
			return result, err
		}
	}

	alphaT := powerLocalField(challenges.Alpha, domainSize)
	alphaTwoT := alphaT
	alphaTwoT.Square(&alphaTwoT)
	hAlpha, err := sourceCommitmentLinearCombination(
		view.Public.W2[:],
		[]fr.Element{fr.One(), alphaT, alphaTwoT},
	)
	if err != nil {
		return result, err
	}
	alphaFamilies := [LocalTerminalAlphaCount]bn254.G1Affine{
		phi[0], phi[1], phi[2],
		phiPrime[0], phiPrime[1], phiPrime[2],
		view.Public.W1,
		view.Fixed.Selectors[0], view.Fixed.Selectors[1], view.Fixed.Selectors[2],
		view.Fixed.Selectors[3], view.Fixed.Selectors[4],
		hAlpha,
	}
	xStarFamilies := [LocalTerminalXStarCount]bn254.G1Affine{
		phi[0], phi[1], phi[2],
		phiPrime[0], phiPrime[1], phiPrime[2],
		view.Public.W1,
	}
	result.Commitments[LocalTerminalAtAlpha], err = sourceCommitmentPowerFold(alphaFamilies[:], challenges.Mu[LocalTerminalAtAlpha])
	if err != nil {
		return result, err
	}
	// The omega*alpha group contains only z, so mu^0=1.
	result.Commitments[LocalTerminalAtOmegaAlpha] = view.Public.W1
	result.Commitments[LocalTerminalAtXStar], err = sourceCommitmentPowerFold(xStarFamilies[:], challenges.Mu[LocalTerminalAtXStar])
	if err != nil {
		return result, err
	}
	return result, nil
}

func validateLocalPIOPFixedCommitmentSlot(slot LocalPIOPFixedCommitmentSlot) error {
	if slot.Parties < 2 || !isPowerOfTwo(slot.Parties) || slot.Rank < 0 || slot.Rank >= slot.Parties ||
		slot.DomainSize < 4 || !isPowerOfTwo(slot.DomainSize) || slot.Parties > slot.DomainSize {
		return fmt.Errorf("%w: malformed slot metadata", ErrInvalidLocalPIOPFixedCommitments)
	}
	wantLabel := fr.NewElement(uint64(slot.Rank))
	if !slot.SlotLabel.Equal(&wantLabel) {
		return fmt.Errorf("%w: slot %d label is not its canonical field embedding", ErrInvalidLocalPIOPFixedCommitments, slot.Rank)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if !sourceCommitmentPointValid(&slot.Selectors[selector]) {
			return fmt.Errorf("%w: malformed selector commitment %d", ErrInvalidLocalPIOPFixedCommitments, selector)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !sourceCommitmentPointValid(&slot.SigmaX[wire]) || !sourceCommitmentPointValid(&slot.SigmaPart[wire]) {
			return fmt.Errorf("%w: malformed wiring commitment %d", ErrInvalidLocalPIOPFixedCommitments, wire)
		}
	}
	if !sourceCommitmentPointValid(&slot.One) || !sourceCommitmentPointValid(&slot.X) {
		return fmt.Errorf("%w: malformed public-family basis commitment", ErrInvalidLocalPIOPFixedCommitments)
	}
	return nil
}

func validateAggregateLocalPIOPFixedCommitments(fixed AggregateLocalPIOPFixedCommitments) error {
	if fixed.Parties < 2 || !isPowerOfTwo(fixed.Parties) || fixed.DomainSize < 4 ||
		!isPowerOfTwo(fixed.DomainSize) || fixed.Parties > fixed.DomainSize {
		return fmt.Errorf("%w: malformed aggregate metadata", ErrInvalidLocalPIOPFixedCommitments)
	}
	for selector := 0; selector < LocalSelectorCount; selector++ {
		if !sourceCommitmentPointValid(&fixed.Selectors[selector]) {
			return fmt.Errorf("%w: malformed aggregate selector %d", ErrInvalidLocalPIOPFixedCommitments, selector)
		}
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !sourceCommitmentPointValid(&fixed.SigmaX[wire]) || !sourceCommitmentPointValid(&fixed.SigmaPart[wire]) {
			return fmt.Errorf("%w: malformed aggregate wiring commitment %d", ErrInvalidLocalPIOPFixedCommitments, wire)
		}
	}
	if !sourceCommitmentPointValid(&fixed.Public.One) || !sourceCommitmentPointValid(&fixed.Public.Upsilon) ||
		!sourceCommitmentPointValid(&fixed.Public.X) {
		return fmt.Errorf("%w: malformed aggregate public-family commitment", ErrInvalidLocalPIOPFixedCommitments)
	}
	return nil
}

func validateAggregateLocalPIOPCommitmentView(
	view AggregateLocalPIOPCommitmentView,
	domainSize int,
	wireCosets [LocalWireCount]fr.Element,
) error {
	if err := validateAggregateLocalPIOPFixedCommitments(view.Fixed); err != nil {
		return err
	}
	if view.Public.Parties != view.Fixed.Parties || view.Public.DomainSize != view.Fixed.DomainSize ||
		domainSize != view.Fixed.DomainSize {
		return fmt.Errorf("%w: public/fixed/domain metadata mismatch", ErrInvalidLocalPIOPPublicCommitments)
	}
	if err := validateLocalPIOPIndex(LocalPIOPIndex{WireCosets: wireCosets}, domainSize); err != nil {
		return fmt.Errorf("%w: wire cosets: %v", ErrInvalidLocalPIOPPublicCommitments, err)
	}
	for wire := 0; wire < LocalWireCount; wire++ {
		if !sourceCommitmentPointValid(&view.Public.W0[wire]) {
			return fmt.Errorf("%w: malformed W0 commitment %d", ErrInvalidLocalPIOPPublicCommitments, wire)
		}
	}
	if !sourceCommitmentPointValid(&view.Public.W1) {
		return fmt.Errorf("%w: malformed W1 commitment", ErrInvalidLocalPIOPPublicCommitments)
	}
	for chunk := range view.Public.W2 {
		if !sourceCommitmentPointValid(&view.Public.W2[chunk]) {
			return fmt.Errorf("%w: malformed W2 commitment %d", ErrInvalidLocalPIOPPublicCommitments, chunk)
		}
	}
	return nil
}

func sourceCommitmentPointValid(point *bn254.G1Affine) bool {
	return point != nil && point.IsOnCurve() && point.IsInSubGroup()
}

func sourceCommitmentAdd(destination *bn254.G1Affine, source *bn254.G1Affine) {
	var sum bn254.G1Jac
	sum.FromAffine(destination)
	sum.AddMixed(source)
	destination.FromJacobian(&sum)
}

func sourceCommitmentAddScaled(destination *bn254.G1Affine, source *bn254.G1Affine, scalar fr.Element) {
	if scalar.IsZero() {
		return
	}
	var scalarBig big.Int
	scalar.ToBigIntRegular(&scalarBig)
	var scaled bn254.G1Jac
	scaled.ScalarMultiplicationAffine(source, &scalarBig)
	var sum bn254.G1Jac
	sum.FromAffine(destination)
	sum.AddAssign(&scaled)
	destination.FromJacobian(&sum)
}

func sourceCommitmentLinearCombination(points []bn254.G1Affine, scalars []fr.Element) (bn254.G1Affine, error) {
	if len(points) != len(scalars) {
		return bn254.G1Affine{}, fmt.Errorf(
			"%w: linear combination has %d points and %d scalars",
			ErrInvalidLocalPIOPPublicCommitments, len(points), len(scalars),
		)
	}
	var sum bn254.G1Jac
	for index := range points {
		if scalars[index].IsZero() {
			continue
		}
		var scalarBig big.Int
		scalars[index].ToBigIntRegular(&scalarBig)
		var scaled bn254.G1Jac
		scaled.ScalarMultiplicationAffine(&points[index], &scalarBig)
		sum.AddAssign(&scaled)
	}
	var result bn254.G1Affine
	result.FromJacobian(&sum)
	return result, nil
}

func sourceCommitmentPowerFold(points []bn254.G1Affine, challenge fr.Element) (bn254.G1Affine, error) {
	scalars := make([]fr.Element, len(points))
	power := fr.One()
	for index := range scalars {
		scalars[index] = power
		power.Mul(&power, &challenge)
	}
	return sourceCommitmentLinearCombination(points, scalars)
}
