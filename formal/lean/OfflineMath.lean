import Returns
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace StateSetFormal.Offline
open Finset Real
open scoped BigOperators
noncomputable section

/-- CQL's uniform-proposal soft backup. The log N correction is needed
    for the upper bound and for equal proposals to keep their Q value. -/
def softBackup {n : ℕ} (tau : ℝ) (q : Fin n → ℝ) :=
  tau * (log (∑ i, exp (q i/tau)) - log n)

/-- The high-temperature branch can evaluate the normalized log partition
    as log1p(mean(expm1)), preserving small deviations from equal scores. -/
theorem logMean_correction {n : ℕ} [NeZero n] (x : Fin n → ℝ) :
    log (1 + (∑ i, (exp (x i)-1)) / n) = log (∑ i, exp (x i)) - log n := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  rw [← log_div (by positivity) hn.ne']
  congr 1
  simp only [sum_sub_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  field_simp

theorem softBackup_bounds {n : ℕ} [NeZero n] {t a b : ℝ} (ht : 0 < t)
    (q : Fin n → ℝ) (lo : ∀ i, a ≤ q i) (hi : ∀ i, q i ≤ b) :
    a ≤ softBackup t q ∧ softBackup t q ≤ b := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  have hsum : 0 < ∑ i, exp (q i / t) := by positivity
  have lower := sum_le_sum (s := (univ : Finset (Fin n))) (fun i _ =>
    exp_le_exp.mpr (div_le_div_of_nonneg_right (lo i) ht.le))
  have upper := sum_le_sum (s := (univ : Finset (Fin n))) (fun i _ =>
    exp_le_exp.mpr (div_le_div_of_nonneg_right (hi i) ht.le))
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at lower upper
  have hl := log_le_log (mul_pos hn (exp_pos _)) lower
  have hu := log_le_log hsum upper
  rw [log_mul hn.ne' (exp_pos _).ne', log_exp] at hl hu
  unfold softBackup
  constructor
  · rw [mul_comm, ← div_le_iff₀ ht]
    linarith
  · rw [mul_comm, ← le_div_iff₀ ht]
    linarith

theorem expectile_weighted_slope (weight target prediction : ℝ) :
    HasDerivAt (fun v => weight * (target-v)^2)
      (-2*weight*(target-prediction)) prediction := by
  convert (((hasDerivAt_id prediction).const_sub target).pow 2).const_mul weight using 1
  simp
  ring

/-- The two expectile quadratics meet with the same zero slope at the target. -/
theorem expectile_derivative (tau target v : ℝ) :
    HasDerivAt (fun x => Returns.expectile tau (target-x))
      (-2 * (if 0 < target-v then tau else 1-tau) * (target-v)) v := by
  rcases lt_trichotomy v target with h | rfl | h
  · rw [if_pos (sub_pos.mpr h)]
    apply (expectile_weighted_slope tau target v).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds h] with x hx
    simp [Returns.expectile, sub_pos.mpr (Set.mem_Iio.mp hx)]
  · simp only [sub_self, mul_zero]
    have branch (w : ℝ) : HasDerivAt (fun x => w*(v-x)^2) 0 v := by
      simpa using expectile_weighted_slope w v v
    rw [← hasDerivWithinAt_univ, ← Set.Iic_union_Ici (a := v)]
    apply HasDerivWithinAt.union
    · apply (branch tau).hasDerivWithinAt.congr_of_mem _ Set.right_mem_Iic
      intro x hx
      rcases (Set.mem_Iic.mp hx).lt_or_eq with h | rfl
      · simp [Returns.expectile, sub_pos.mpr h]
      · simp [Returns.expectile]
    · apply (branch (1-tau)).hasDerivWithinAt.congr_of_mem _ Set.left_mem_Ici
      intro x hx
      simp [Returns.expectile, not_lt.mpr (sub_nonpos.mpr (Set.mem_Ici.mp hx))]
  · rw [if_neg (not_lt.mpr (sub_nonpos.mpr h.le))]
    apply (expectile_weighted_slope (1-tau) target v).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds h] with x hx
    simp [Returns.expectile, not_lt.mpr (sub_nonpos.mpr (Set.mem_Ioi.mp hx).le)]

theorem expectile_batch_derivative {n : ℕ} [NeZero n] (tau v : ℝ) (y : Fin n → ℝ) :
    HasDerivAt (fun u => (∑ i, Returns.expectile tau (y i-u)) / n)
      ((∑ i, -2*(if 0 < y i-v then tau else 1-tau)*(y i-v)) / n) v := by
  exact (HasDerivAt.sum fun i _ => expectile_derivative tau (y i) v).div_const _

/-- Each logged state can have a different value and parameter sensitivity. -/
theorem network_expectile_gradient {n : ℕ} [NeZero n] (tau x : ℝ)
    (v : Fin n → ℝ → ℝ) (j y : Fin n → ℝ) (hv : ∀ i, HasDerivAt (v i) (j i) x) :
    HasDerivAt (fun z => (∑ i, Returns.expectile tau (y i-v i z)) / n)
      ((∑ i, -2*(if 0 < y i-v i x then tau else 1-tau)*(y i-v i x)*j i) / n) x := by
  exact (HasDerivAt.sum fun i _ => (expectile_derivative tau (y i) (v i x)).comp x (hv i)).div_const _

/-- Quadratic lower support, including comparisons that cross the target. -/
theorem expectile_support (t e f : ℝ) :
    Returns.expectile t e + 2*(if 0 < e then t else 1-t)*e*(f-e) +
        min t (1-t)*(f-e)^2 ≤ Returns.expectile t f := by
  let m := min t (1-t)
  have h₁ : 0 ≤ t-m := sub_nonneg.mpr (min_le_left ..)
  have h₂ : 0 ≤ (1-t)-m := sub_nonneg.mpr (min_le_right ..)
  change Returns.expectile t e + _ + m * _ ≤ _
  unfold Returns.expectile
  by_cases he : 0 < e <;> by_cases hf : 0 < f <;> simp only [he, hf, if_true, if_false]
  · nlinarith [mul_nonneg h₁ (sq_nonneg (f-e))]
  · have cross := mul_nonpos_of_nonneg_of_nonpos (mul_nonneg h₁ he.le) (le_of_not_gt hf)
    nlinarith [mul_nonneg h₁ (sq_nonneg e), mul_nonneg h₂ (sq_nonneg f)]
  · have cross := mul_nonpos_of_nonneg_of_nonpos (mul_nonneg h₂ hf.le) (le_of_not_gt he)
    nlinarith [mul_nonneg h₂ (sq_nonneg e), mul_nonneg h₁ (sq_nonneg f)]
  · nlinarith [mul_nonneg h₂ (sq_nonneg (f-e))]

/-- Any fitted scalar that balances the weighted residuals minimizes the
    empirical objective. At an interior expectile the lower bound is strict
    for every other prediction on a nonempty batch. -/
theorem expectile_balanced_lower_bound {n : ℕ} [NeZero n] (y : Fin n → ℝ) (t v u : ℝ)
    (balanced : (∑ i, (if 0 < y i-v then t else 1-t)*(y i-v)) = 0) :
    (∑ i, Returns.expectile t (y i-v)) / n + min t (1-t)*(v-u)^2 ≤
      (∑ i, Returns.expectile t (y i-u)) / n := by
  have point (i : Fin n) : Returns.expectile t (y i-v) +
      (if 0 < y i-v then t else 1-t)*(y i-v)*(2*(v-u)) + min t (1-t)*(v-u)^2 ≤
      Returns.expectile t (y i-u) := by
    have h := expectile_support t (y i-v) (y i-u)
    rw [show y i-u-(y i-v) = v-u by ring] at h
    linarith only [h]
  have bound := sum_le_sum (s := (univ : Finset (Fin n))) fun i _ => point i
  simp only [sum_add_distrib, ← sum_mul, ← mul_sum, balanced, mul_zero, zero_mul,
    add_zero, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at bound
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  apply (le_div_iff₀ hn).mpr
  rw [add_mul, div_mul_cancel₀ _ hn.ne', mul_comm (min t (1-t) * _)]
  simpa only [mul_assoc] using bound

theorem expectile_strict_minimum {n : ℕ} [NeZero n] (y : Fin n → ℝ) {t v u : ℝ}
    (ht : 0 < t) (ht' : t < 1) (hu : u ≠ v)
    (balanced : (∑ i, (if 0 < y i-v then t else 1-t)*(y i-v)) = 0) :
    (∑ i, Returns.expectile t (y i-v)) / n < (∑ i, Returns.expectile t (y i-u)) / n := by
  have curvature := mul_pos (lt_min ht (sub_pos.mpr ht'))
    (sq_pos_of_ne_zero (sub_ne_zero.mpr hu.symm))
  linarith [expectile_balanced_lower_bound y t v u balanced]

/-- Every nonempty empirical sample has a unique interior-expectile fit,
    located between its smallest and largest observed target. -/
theorem expectile_empirical_optimum {n : ℕ} [NeZero n] (y : Fin n → ℝ)
    {t : ℝ} (ht : 0 < t) (ht' : t < 1) :
    ∃ v ∈ Set.Icc (univ.inf' univ_nonempty y) (univ.sup' univ_nonempty y),
      ∀ u ≠ v, (∑ i, Returns.expectile t (y i-v)) / n <
        (∑ i, Returns.expectile t (y i-u)) / n := by
  let lo := univ.inf' univ_nonempty y
  let hi := univ.sup' univ_nonempty y
  have lower (i : Fin n) : lo ≤ y i := inf'_le y (mem_univ i)
  have upper (i : Fin n) : y i ≤ hi := le_sup' y (mem_univ i)
  let score := fun v : ℝ => ∑ i, (t*max (y i-v) 0 + (1-t)*min (y i-v) 0)
  have hc : Continuous score := by
    apply continuous_finset_sum
    intro i _
    have d : Continuous (fun v : ℝ => y i-v) := continuous_const.sub continuous_id
    exact (continuous_const.mul (d.max continuous_const)).add
      (continuous_const.mul (d.min continuous_const))
  have hl : 0 ≤ score lo := by
    apply sum_nonneg
    intro i _
    simp only [min_eq_right (sub_nonneg.mpr (lower i)), mul_zero, add_zero]
    exact mul_nonneg ht.le (le_max_right ..)
  have hh : score hi ≤ 0 := by
    apply sum_nonpos
    intro i _
    simp only [max_eq_right (sub_nonpos.mpr (upper i)), mul_zero, zero_add]
    exact mul_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr ht'.le) (min_le_right ..)
  obtain ⟨v, hv, root⟩ := intermediate_value_Icc' ((lower 0).trans (upper 0)) hc.continuousOn
    (show (0 : ℝ) ∈ Set.Icc (score hi) (score lo) from ⟨hh, hl⟩)
  refine ⟨v, hv, fun u hu => expectile_strict_minimum y ht ht' hu ?_⟩
  have term (e : ℝ) : (if 0 < e then t else 1-t)*e = t*max e 0 + (1-t)*min e 0 := by
    by_cases h : 0 < e
    · simp [h, max_eq_left h.le, min_eq_right h.le]
    · simp [h, max_eq_right (le_of_not_gt h), min_eq_left (le_of_not_gt h)]
  simpa only [term] using root

/-- At the endpoints the minimum is zero on an entire half-line. -/
theorem expectile_endpoint_minima {n : ℕ} [NeZero n] (y : Fin n → ℝ) (v : ℝ) :
    (((∑ i, Returns.expectile 1 (y i-v)) / n = 0) ↔ ∀ i, y i ≤ v) ∧
    (((∑ i, Returns.expectile 0 (y i-v)) / n = 0) ↔ ∀ i, v ≤ y i) := by
  have hn : (n : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne n
  have zero_iff (t : ℝ) (ht : 0 ≤ t) (ht' : t ≤ 1) :
      (∑ i, Returns.expectile t (y i-v)) / n = 0 ↔
        ∀ i, Returns.expectile t (y i-v) = 0 := by
    simp [div_eq_zero_iff, hn, sum_eq_zero_iff_of_nonneg
      (fun i _ => Returns.expectile_nonnegative (y i-v) ht ht')]
  rw [zero_iff 1 zero_le_one le_rfl, zero_iff 0 le_rfl zero_le_one]
  have upper (e : ℝ) : Returns.expectile 1 e = 0 ↔ e ≤ 0 := by
    by_cases h : 0 < e
    · simp [Returns.expectile, h, h.ne', not_le.mpr h]
    · simp [Returns.expectile, h, le_of_not_gt h]
  have lower (e : ℝ) : Returns.expectile 0 e = 0 ↔ 0 ≤ e := by
    rcases lt_trichotomy e 0 with h | rfl | h
    · simp [Returns.expectile, not_lt.mpr h.le, h.ne, not_le.mpr h]
    · simp [Returns.expectile]
    · simp [Returns.expectile, h, h.le]
  simp only [upper, lower, sub_nonpos, sub_nonneg, iff_self, and_self]

/-- The sampled CQL(H) penalty uses the unnormalized partition. Subtracting
    the logged action before restoring the offset preserves its real value. -/
def cqlGap {n : ℕ} (t : ℝ) (q : Fin n → ℝ) (data : ℝ) :=
  t * log (∑ i, exp (q i/t)) - data

theorem cqlGap_centered {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q : Fin n → ℝ) (data center : ℝ) :
    cqlGap t q data = (center-data) + t * log (∑ i, exp ((q i-center)/t)) := by
  unfold cqlGap
  simp_rw [sub_div, exp_sub]
  rw [← sum_div, log_div (by positivity) (exp_pos _).ne', log_exp]
  field_simp
  ring

/-- The proposal gradient is a softmax-weighted average of critic slopes.
    The logged action has coefficient -1, including when its parameters
    are shared with the sampled proposals. -/
theorem cqlGap_derivative {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q : Fin n → ℝ → ℝ) (dq : Fin n → ℝ) (data : ℝ → ℝ) (dd x : ℝ)
    (hq : ∀ i, HasDerivAt (q i) (dq i) x) (hd : HasDerivAt data dd x) :
    HasDerivAt (fun y => cqlGap t (fun i => q i y) (data y))
      ((∑ i, (exp (q i x/t) / ∑ j, exp (q j x/t)) * dq i) - dd) x := by
  have z : 0 < ∑ i, exp (q i x/t) := by positivity
  have part := HasDerivAt.sum (u := univ) fun i _ => ((hq i).div_const t).exp
  convert (((part.log z.ne').const_mul t).sub hd) using 1
  simp only [div_eq_mul_inv, mul_sum, sum_mul]
  apply congrArg (· - dd)
  apply sum_congr rfl
  intro i _
  field_simp
  ring

/-- Subtracting the constant log N normalization leaves all directional
    derivatives at the softmax-weighted average of the individual slopes. -/
theorem softBackup_derivative {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q : Fin n → ℝ → ℝ) (dq : Fin n → ℝ) (x : ℝ)
    (hq : ∀ i, HasDerivAt (q i) (dq i) x) :
    HasDerivAt (fun y => softBackup t (fun i => q i y))
      (∑ i, (exp (q i x/t) / ∑ j, exp (q j x/t)) * dq i) x := by
  have h := (cqlGap_derivative ht q dq (fun _ => 0) 0 x hq (hasDerivAt_const _ _)).sub_const (t*log n)
  simpa [cqlGap, softBackup, mul_sub] using h

/-- Directional derivative of one complete sampled CQL critic objective.
    Targets, proposal actions and the multiplier are held fixed for this update. -/
theorem sampled_critic_gradient {b n : ℕ} [NeZero b] [NeZero n] {t : ℝ} (ht : 0 < t)
    (α x : ℝ) (q : Fin b → Fin n → ℝ → ℝ) (dq : Fin b → Fin n → ℝ)
    (data : Fin b → ℝ → ℝ) (dd y : Fin b → ℝ)
    (hq : ∀ i j, HasDerivAt (q i j) (dq i j) x) (hd : ∀ i, HasDerivAt (data i) (dd i) x) :
    HasDerivAt (fun z => (∑ i, ((data i z-y i)^2 + α*cqlGap t (fun j => q i j z) (data i z))) / b)
      ((∑ i, (2*(data i x-y i)*dd i + α*((∑ j,
        (exp (q i j x/t) / ∑ k, exp (q i k x/t)) * dq i j) - dd i))) / b) x := by
  apply HasDerivAt.div_const
  apply HasDerivAt.sum
  intro i _
  have mse : HasDerivAt (fun z => (data i z-y i)^2) (2*(data i x-y i)*dd i) x := by
    simpa using (((hd i).sub_const (y i)).pow 2)
  exact mse.add ((cqlGap_derivative ht (q i) (dq i) (data i) (dd i) x (hq i) (hd i)).const_mul α)

/-- Evaluate the odd and even parts separately before averaging proposals.
    For equal and opposite small logits, the odd part cancels exactly. -/
theorem expm1_odd_even (z : ℝ) : exp z - 1 = sinh z + 2 * sinh (z/2)^2 := by
  have h := cosh_two_mul (z/2)
  rw [show 2*(z/2) = z by ring, cosh_sq] at h
  linarith [cosh_add_sinh z]

theorem symmetric_softBackup {t : ℝ} (ht : 0 < t) :
    softBackup t (![-t, t] : Fin 2 → ℝ) = t * log ((exp (-1) + exp 1) / 2) := by
  rw [log_div (by positivity) (by norm_num)]
  simp [softBackup, Fin.sum_univ_two, neg_div, ht.ne']

/-- The stabilized logsumexp backward pass gives each original proposal its
    softmax coefficient, even if more than one proposal attains the maximum. -/
theorem softmax_centered {n : ℕ} [NeZero n] (t : ℝ) (q : Fin n → ℝ)
    (center : ℝ) (i : Fin n) :
    exp ((q i-center)/t - log (∑ j, exp ((q j-center)/t))) =
      exp (q i/t) / ∑ j, exp (q j/t) := by
  rw [exp_sub, exp_log (by positivity)]
  simp_rw [sub_div, exp_sub]
  rw [← sum_div]
  field_simp

theorem softBackup_monotone {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q q' : Fin n → ℝ) (h : ∀ i, q i ≤ q' i) : softBackup t q ≤ softBackup t q' := by
  unfold softBackup
  apply mul_le_mul_of_nonneg_left _ ht.le
  apply sub_le_sub_right
  apply log_le_log (by positivity)
  exact sum_le_sum fun i _ => exp_le_exp.mpr (div_le_div_of_nonneg_right (h i) ht.le)

theorem softBackup_shift {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q : Fin n → ℝ) (c : ℝ) : softBackup t (fun i => q i + c) = softBackup t q + c := by
  unfold softBackup
  simp_rw [add_div, exp_add]
  rw [← sum_mul, log_mul (by positivity) (exp_pos _).ne', log_exp]
  field_simp
  ring

/-- The temperature-smoothed backup is nonexpansive for any proposal count.
    Composing with a Bellman discount multiplies this bound by γ. -/
theorem softBackup_lipschitz {n : ℕ} [NeZero n] {t : ℝ} (ht : 0 < t)
    (q q' : Fin n → ℝ) (eps : ℝ) (h : ∀ i, |q i-q' i| ≤ eps) :
    |softBackup t q - softBackup t q'| ≤ eps := by
  have h1 := softBackup_monotone ht q (fun i => q' i + eps)
    (fun i => by change q i ≤ q' i + eps; linarith [(abs_le.mp (h i)).2])
  have h2 := softBackup_monotone ht q' (fun i => q i + eps)
    (fun i => by change q' i ≤ q i + eps; linarith [(abs_le.mp (h i)).1])
  rw [softBackup_shift ht] at h1 h2
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem capped_weight {a cap : ℝ} (hc : 0 < cap) :
    exp (min a (log cap)) = min (exp a) cap := by
  by_cases h : a ≤ log cap
  · rw [min_eq_left h, min_eq_left]
    exact (exp_le_exp.mpr h).trans_eq (exp_log hc)
  · rw [min_eq_right (le_of_not_ge h), exp_log hc, min_eq_right]
    exact (exp_log hc).symm.trans_le (exp_le_exp.mpr (le_of_not_ge h))

/-- Publishing the configured cap exactly in the upper branch preserves
the clipped exponential weight. -/
theorem saturated_weight {x cap : ℝ} (hc : 0 < cap) :
    (if log cap ≤ x then cap else min (exp (min x (log cap))) cap) =
      min (exp x) cap := by
  rw [capped_weight hc, min_assoc, min_self]
  split_ifs with h
  · exact (min_eq_right ((exp_log hc).symm.trans_le (exp_le_exp.mpr h))).symm
  · rfl

/-- The KL contribution per latent coordinate in BCQ's diagonal Gaussian VAE. -/
theorem gaussianKL_nonnegative (mean logVariance : ℝ) :
    0 ≤ -(1 + logVariance - mean^2 - exp logVariance)/2 := by
  linarith [sq_nonneg mean, add_one_le_exp logVariance]

/-- The BCQ perturbation network bounds each coordinate's displacement from
the VAE proposal. Membership of a data distribution's support is separate. -/
theorem bcq_displacement (candidate output : ℝ) {phi : ℝ} (hp : 0 ≤ phi) :
    |candidate + phi*tanh output - candidate| ≤ phi := by
  have hc := cosh_pos output
  have ht : |tanh output| ≤ 1 := by
    rw [tanh_eq_sinh_div_cosh, abs_le]
    constructor
    · apply (le_div_iff₀ hc).mpr
      linarith [cosh_add_sinh output, exp_pos output]
    · apply (div_le_iff₀ hc).mpr
      linarith [cosh_sub_sinh output, exp_pos (-output)]
  simpa [abs_mul, abs_of_nonneg hp] using mul_le_mul_of_nonneg_left ht hp

/-- Given a positive semidefinite Gram form, the biased MMD² estimate is
    nonnegative with coefficients +1/m on policy and -1/n on behavior draws.
    Here the kernel is represented by finite-dimensional feature vectors. -/
theorem mmd_feature_expansion {d : ℕ} (x y : Fin d → ℝ) :
    (∑ i, x i ^ 2) + (∑ i, y i ^ 2) - 2*(∑ i, x i*y i) =
      ∑ i, (x i-y i)^2 := by
  rw [Finset.mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
  apply sum_congr rfl
  intro i _
  ring

theorem mmd_feature_nonnegative {d : ℕ} (x y : Fin d → ℝ) :
    0 ≤ (∑ i, x i ^ 2) + (∑ i, y i ^ 2) - 2*(∑ i, x i*y i) := by
  rw [mmd_feature_expansion]
  exact sum_nonneg fun _ _ => sq_nonneg _

/-- Both radial BEAR kernels have this form, with exponent given by either
    the Euclidean distance or its square, scaled by a positive bandwidth. -/
theorem radial_kernel_range {distance scale : ℝ} (hd : 0 ≤ distance) (hs : 0 < scale) :
    0 < exp (-distance / scale) ∧ exp (-distance / scale) ≤ 1 := by
  exact ⟨exp_pos _, exp_le_one_iff.mpr (div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hd) hs.le)⟩

theorem radial_kernel_diagonal (scale : ℝ) : exp (-(0 : ℝ) / scale) = 1 := by simp

/-- Decision Transformer reads the state slot (3t+1) for action t. A causal
    attention edge from an action slot can therefore only use a past action. -/
theorem causal_action_index (source target : ℕ) :
    3*source+2 ≤ 3*target+1 ↔ source < target := by omega

/-- A positive-length retained trajectory has one fewer successor-state
    target than action targets. This also covers batches of only singletons. -/
theorem trajectory_target_counts {n : ℕ} (length : Fin n → ℕ) (h : ∀ i, 0 < length i) :
    (∑ i, (length i-1)) + n = ∑ i, length i := by
  calc
    _ = ∑ i, ((length i-1) + 1) := by simp [sum_add_distrib]
    _ = _ := sum_congr rfl fun i _ => Nat.sub_add_cancel (h i)

theorem target_interpolation (w current old : ℝ) :
    (w*current+(1-w)*old) - current = (1-w)*(old-current) := by ring

/-- DAPO interpolation inside the overlength cache window. -/
theorem overlong_bounds {penalty start len cache : ℝ} (hp : penalty ≤ 0)
    (hc : 0 < cache) (lower : start ≤ len) (upper : len ≤ start+cache) :
    penalty ≤ penalty*((len-start)/cache) ∧ penalty*((len-start)/cache) ≤ 0 := by
  have h : (len-start)/cache ≤ 1 := (div_le_one hc).mpr (by linarith)
  exact ⟨by simpa using mul_le_mul_of_nonpos_left h hp,
          mul_nonpos_of_nonpos_of_nonneg hp (div_nonneg (sub_nonneg.mpr lower) hc.le)⟩
end
end StateSetFormal.Offline
