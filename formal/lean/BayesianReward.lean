import PolicyMath
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-! Scalar Gaussian reward heads and prediction from finite mixtures of their draws. -/
namespace StateSetFormal.BayesianReward
open Finset Real
open scoped BigOperators
noncomputable section

def nll (y m l : ℝ) := (l + (m-y)^2 * exp (-l)) / 2

/-- Taking the negative log of the normal density differs from the training
    loss by the same constant for every label, mean and log variance. -/
theorem negative_log_density (y m l : ℝ) :
    -log ((sqrt (2*π*exp l))⁻¹ * exp (-(y-m)^2/(2*exp l))) =
      nll y m l + log (2*π)/2 := by
  rw [log_mul (by positivity) (exp_ne_zero _), log_inv, log_sqrt (by positivity),
    log_mul (by positivity) (exp_ne_zero l), log_exp, log_exp, nll, exp_neg]
  ring

theorem squared_precision_factor (l : ℝ) : exp (l*(-1/2))^2 = exp (-l) := by
  rw [sq, ← exp_add]
  congr 1
  ring

theorem logged_residual {a : ℝ} (ha : 0 < a) (l : ℝ) :
    exp (log a * 2 - l) = a^2 * exp (-l) := by
  rw [show log a * 2 - l = (log a + log a) + (-l) by ring, exp_add, exp_add, exp_log ha, sq]

/-- Slopes for a row paired with its own label, with respect to each head. -/
theorem head_slopes (y m l : ℝ) :
    HasDerivAt (fun t => nll y t l) ((m-y)*exp (-l)) m ∧
    HasDerivAt (nll y m) ((1-(m-y)^2*exp (-l))/2) l := by
  constructor
  · convert (((((hasDerivAt_id m).sub_const y).pow 2).mul_const (exp (-l))).const_add l
      |>.div_const 2) using 1
    simp [nll]
    ring
  · convert (((hasDerivAt_id l).add (((hasDerivAt_id l).neg.exp).const_mul ((m-y)^2))).div_const 2)
      using 1
    simp [nll]
    ring

/-- The two head slopes add along a shared network parameter, with each mean
    paired with its own target and log variance before batch reduction. -/
theorem paired_batch_derivative {n : ℕ} [NeZero n] (y dm dl : Fin n → ℝ)
    (m l : Fin n → ℝ → ℝ) (z : ℝ)
    (hm : ∀ i, HasDerivAt (m i) (dm i) z) (hl : ∀ i, HasDerivAt (l i) (dl i) z) :
    HasDerivAt (fun t => (∑ i, nll (y i) (m i t) (l i t))/n)
      ((∑ i, ((m i z-y i)*exp (-(l i z))*dm i +
        (1-(m i z-y i)^2*exp (-(l i z)))*dl i/2))/n) z := by
  apply HasDerivAt.div_const
  apply HasDerivAt.sum
  intro i _
  convert ((hl i).add ((((hm i).sub_const (y i)).pow 2).mul (hl i).neg.exp)).div_const 2 using 1
  ring

/-- For positive mean squared residual v the unique best log variance is log v. -/
theorem best_log_variance {v l : ℝ} (hv : 0 < v) (hne : l ≠ log v) :
    (log v + 1)/2 < (l + v*exp (-l))/2 := by
  have gap := add_one_lt_exp (x := log v-l) (sub_ne_zero.mpr (Ne.symm hne))
  rw [sub_eq_add_neg, exp_add, exp_log hv] at gap
  linarith

theorem zero_residual_has_no_minimum (y l : ℝ) : nll y y (l-1) < nll y y l := by
  simp [nll]
  linarith

def average {n : ℕ} (w x : Fin n → ℝ) := ∑ i, w i * x i

/-- Centering a finite distribution at any other value adds the squared bias.
    Nonnegative normalized weights give the usual probabilistic interpretation. -/
theorem squared_risk {n : ℕ} (w x : Fin n → ℝ) (hw : ∑ i, w i = 1) (a : ℝ) :
    average w (fun i => (x i-a)^2) =
      average w (fun i => (x i-average w x)^2) + (average w x-a)^2 := by
  let m := average w x
  have expand (c : ℝ) : average w (fun i => (x i-c)^2) =
      average w (fun i => (x i)^2) - 2*c*m + c^2 := by
    have row (i : Fin n) : w i * (x i-c)^2 =
        w i * (x i)^2 - 2*c*(w i*x i) + w i*c^2 := by ring
    simp only [average, row, sum_add_distrib, sum_sub_distrib, ← mul_sum, ← sum_mul, hw,
      one_mul, m]
  rw [expand, expand]
  dsimp [m]
  ring

/-- The expected Gaussian score depends on the label distribution through
    its mean and variance alone. Moving the predicted mean adds a quadratic. -/
theorem expected_nll {n : ℕ} (w y : Fin n → ℝ) (hw : ∑ i, w i = 1) (m l : ℝ) :
    average w (fun i => nll (y i) m l) =
    (l + (average w (fun i => (y i-average w y)^2) + (average w y-m)^2)*exp (-l))/2 := by
  rw [← squared_risk w y hw m]
  have row (i : Fin n) : w i * nll (y i) m l =
      (w i*l + (w i*(y i-m)^2)*exp (-l))/2 := by unfold nll; ring
  simp [average, row, ← sum_div, sum_add_distrib, ← sum_mul, hw]

/-- Jointly fitting both heads selects the empirical mean and log population
    variance whenever the labels have positive variance. -/
theorem unique_best_heads {n : ℕ} (w y : Fin n → ℝ) (hw : ∑ i, w i = 1) (m l : ℝ)
    (hv : 0 < average w (fun i => (y i-average w y)^2)) :
    (average w (fun i => nll (y i) m l) ≤
     average w (fun i => nll (y i) (average w y)
       (log (average w (fun j => (y j-average w y)^2))))) ↔
    m = average w y ∧ l = log (average w (fun i => (y i-average w y)^2)) := by
  let v := average w (fun i => (y i-average w y)^2)
  change 0 < v at hv
  constructor
  · intro h
    simp only [expected_nll w y hw] at h
    change (l + (v + (average w y-m)^2)*exp (-l))/2 ≤
      (log v + (v + (average w y-average w y)^2)*exp (-log v))/2 at h
    rw [sub_self, zero_pow (by decide : 2 ≠ 0), add_zero, exp_neg (log v), exp_log hv,
      mul_inv_cancel₀ hv.ne'] at h
    have bias := mul_nonneg (sq_nonneg (average w y-m)) (exp_pos (-l)).le
    have hl : l = log v := by
      by_contra hne
      nlinarith [best_log_variance hv hne]
    refine ⟨?_, hl⟩
    by_contra hm
    have pos := mul_pos (sq_pos_of_ne_zero (sub_ne_zero.mpr (Ne.symm hm))) (exp_pos (-l))
    have unit : v*exp (-l) = 1 := by rw [hl, exp_neg, exp_log hv, mul_inv_cancel₀ hv.ne']
    nlinarith
  · rintro ⟨rfl,rfl⟩
    exact le_rfl

/-- Broadcasting two label axes charges exactly the target variance at a
    perfect fit with unit noise. Pairing the rows charges zero. -/
theorem broadcast_penalty {n : ℕ} (w y : Fin n → ℝ) (hw : ∑ i, w i = 1) :
    average w (fun i => nll (y i) (y i) 0) = 0 ∧
    average w (fun i => average w (fun j => nll (y j) (y i) 0)) =
      average w (fun i => (y i-average w y)^2) := by
  constructor
  · simp [nll, average]
  · have half (v : Fin n → ℝ) : average w (fun i => v i / 2) = average w v / 2 := by
      simp [average, ← mul_div_assoc, ← sum_div]
    have inner (i : Fin n) : average w (fun j => nll (y j) (y i) 0) =
        (average w (fun j => (y j-average w y)^2) + (y i-average w y)^2)/2 := by
      have row (j : Fin n) : nll (y j) (y i) 0 = (y j-y i)^2/2 := by simp [nll]; ring
      simp_rw [row, half, squared_risk w y hw (y i)]
      ring
    simp_rw [inner, half]
    simp [average, mul_add, sum_add_distrib, ← sum_mul, hw]

/-- The variance of the conditional means of a model/draw mixture contains
    both disagreement within each member and disagreement between member means. -/
theorem total_epistemic {n k : ℕ} (p : Fin n → ℝ) (w : Fin n → Fin k → ℝ)
    (x : Fin n → Fin k → ℝ) (hw : ∀ i, ∑ j, w i j = 1) :
    let member := fun i => average (w i) (x i)
    let grand := average p member
    average p (fun i => average (w i) (fun j => (x i j-grand)^2)) =
      average p (fun i => average (w i) (fun j => (x i j-member i)^2)) +
      average p (fun i => (member i-grand)^2) := by
  dsimp only
  simp_rw [squared_risk (w _) (x _) (hw _) (average p _)]
  simp [average, mul_add, sum_add_distrib]

/-- Log scale arithmetic agrees with the Gaussian posterior-to-prior KL formula. -/
theorem gaussian_kl_formula (m l : ℝ) {p : ℝ} (hp : 0 < p) :
    log (p/exp l) + ((exp l)^2 + m^2)/(2*p^2) - 1/2 =
      ((m/p)^2 + Policy.k3 (2*(l-log p)))/2 := by
  rw [Policy.k3, show 2*(l-log p) = (l-log p)+(l-log p) by ring, exp_add, exp_sub,
    exp_log hp, log_div hp.ne' (exp_ne_zero l), log_exp]
  field_simp
  ring

/-- The constant-free diagonal Gaussian KL is minimized by matching the prior. -/
theorem gaussian_kl_nonnegative (normalizedMean logStdRatio : ℝ) :
    0 ≤ (normalizedMean^2 + Policy.k3 (2*logStdRatio))/2 := by
  linarith [Policy.k3_nonnegative (2*logStdRatio), sq_nonneg normalizedMean]

end
end StateSetFormal.BayesianReward
