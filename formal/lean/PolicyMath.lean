import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Real-arithmetic contracts for the tensor functions in training/objectives.py.
    A stopped gradient is an independent constant in the differentiated function. -/
namespace StateSetFormal.Policy
open Finset Real
open scoped BigOperators
noncomputable section

def ratio (bound logGap : ℝ) := exp (max (-bound) (min bound logGap))

theorem ratio_bounds {b : ℝ} (hb : 0 ≤ b) (x : ℝ) :
    exp (-b) ≤ ratio b x ∧ ratio b x ≤ exp b := by
  exact ⟨exp_le_exp.mpr (le_max_left ..),
         exp_le_exp.mpr (max_le (by linarith) (min_le_left ..))⟩

theorem same_policy_ratio {b : ℝ} (hb : 0 ≤ b) :
    ratio b 0 = 1 := by simp [ratio, hb, neg_nonpos.mpr hb]

def surrogate (lo hi r a : ℝ) := -min (r*a) ((max lo (min hi r))*a)

theorem surrogate_positive {lo hi r a : ℝ} (bounds : lo ≤ hi) (ha : 0 ≤ a) :
    surrogate lo hi r a = -(min hi r * a) := by
  unfold surrogate
  by_cases low : r < lo
  · rw [min_eq_right (le_trans low.le bounds), max_eq_left low.le,
      min_eq_left (mul_le_mul_of_nonneg_right low.le ha)]
  · have h : lo ≤ min hi r := le_min bounds (le_of_not_gt low)
    rw [max_eq_right h, min_eq_right (mul_le_mul_of_nonneg_right (min_le_right ..) ha)]

theorem surrogate_negative {lo hi r a : ℝ} (bounds : lo ≤ hi) (ha : a ≤ 0) :
    surrogate lo hi r a = -(max lo r * a) := by
  unfold surrogate
  by_cases high : hi ≤ r
  · rw [min_eq_left high, max_eq_right bounds, max_eq_right (bounds.trans high),
      min_eq_left (mul_le_mul_of_nonpos_right high ha)]
  · rw [min_eq_right (le_of_not_ge high),
      min_eq_right (mul_le_mul_of_nonpos_right (le_max_right ..) ha)]

/-- StateSet's optional delta cap affects the unrestricted branch. These
identities hold even when delta is below the ordinary upper clip bound. -/
def cappedSurrogate (lo hi cap r a : ℝ) :=
  -min ((min r cap)*a) ((max lo (min hi r))*a)

theorem capped_positive {lo hi cap r a : ℝ} (bounds : lo ≤ hi) (ha : 0 ≤ a) :
    cappedSurrogate lo hi cap r a = -(min cap (min hi r) * a) := by
  unfold cappedSurrogate
  rw [← min_mul_of_nonneg _ _ ha]
  have base := surrogate_positive (r := r) (a := (1:ℝ)) bounds zero_le_one
  simp only [surrogate, mul_one, neg_inj] at base
  rw [min_comm r cap, min_assoc, base]

theorem capped_negative {lo hi cap r a : ℝ} (ha : a ≤ 0) :
    cappedSurrogate lo hi cap r a = -(max (min r cap) (max lo (min hi r)) * a) := by
  unfold cappedSurrogate
  rcases le_total (min r cap) (max lo (min hi r)) with h | h
  · rw [max_eq_right h, min_eq_right (mul_le_mul_of_nonpos_right h ha)]
  · rw [max_eq_left h, min_eq_left (mul_le_mul_of_nonpos_right h ha)]

/-- The ATen minimum splits tied inputs equally. The clamp passes gradients
at both endpoints. Combining these rules gives the usual PPO gate, with an
active derivative exactly at each endpoint. This is a subgradient convention. -/
def clipBackward (lo hi r a : ℝ) :=
  let c := min (max lo r) hi
  let left := if r*a > c*a then 0 else if r*a = c*a then -1/2 else -1
  let right := if r*a < c*a then 0 else if r*a = c*a then -1/2 else -1
  left * a + (if lo ≤ r ∧ r ≤ hi then right*a else 0)

theorem clip_backward_gate {lo hi : ℝ} (r a : ℝ) (bounds : lo ≤ hi) :
    clipBackward lo hi r a =
    (if (r < lo ∧ a < 0) ∨ (hi < r ∧ 0 < a) then 0 else -a) := by
  unfold clipBackward
  rcases lt_or_ge r lo with h | h
  · rw [max_eq_left h.le, min_eq_left bounds]
    have upper := h.le.trans bounds
    rcases lt_trichotomy a 0 with sign | sign | sign
    · have p := mul_lt_mul_of_neg_right h sign
      simp [h, sign, not_le.mpr h, not_lt.mpr upper, p, p.ne', not_lt.mpr p.le]
    · simp [sign]
    · have p := mul_lt_mul_of_pos_right h sign
      simp [h, not_le.mpr h, not_lt.mpr upper, not_lt.mpr sign.le, p, p.ne, not_lt.mpr p.le]
  · rw [max_eq_right h]
    rcases le_or_gt r hi with u | u
    · rw [min_eq_left u]
      simp [h, u, not_lt.mpr h, not_lt.mpr u]
      ring
    · rw [min_eq_right u.le]
      rcases lt_trichotomy a 0 with sign | sign | sign
      · have p := mul_lt_mul_of_neg_right u sign
        simp [u, not_le.mpr u, not_lt.mpr h, not_lt.mpr sign.le, p, p.ne, not_lt.mpr p.le]
      · simp [sign]
      · have p := mul_lt_mul_of_pos_right u sign
        simp [u, sign, not_le.mpr u, not_lt.mpr h, p, p.ne', not_lt.mpr p.le]

theorem cispo_gradient (r cap a x : ℝ) :
    HasDerivAt (fun z => -(min r cap) * a * z) (-(min r cap)*a) x := by
  simpa using (hasDerivAt_id x).const_mul (-(min r cap)*a)

/-- At a frozen sequence ratio, each GSPO-token log probability receives that
    ratio as its local derivative, regardless of response length. -/
theorem sequence_token_value_gradient (s x : ℝ) :
    s * exp (x-x) = s ∧ HasDerivAt (fun z => s * exp (z-x)) s x := by
  constructor
  · simp
  · simpa using (((hasDerivAt_id x).sub_const x).exp.const_mul s)

theorem geometric_ratio {n : ℕ} (hn : 0 < n) (gap : Fin n → ℝ) :
    (exp ((∑ i, gap i) / n)) ^ n = ∏ i, exp (gap i) := by
  rw [← Real.exp_nat_mul, ← Real.exp_sum]
  congr 1
  field_simp

/-- GEPO's log-sum-exp denominator is the sampler-probability-weighted
    average probability on the empirical group. Repeated samples retain multiplicity. -/
theorem gepo_denominator {n : ℕ} [NeZero n] (q : Fin n → ℝ) :
    exp (log (∑ i, exp (2*q i)) - log (∑ i, exp (q i))) =
      (∑ i, (exp (q i))^2) / (∑ i, exp (q i)) := by
  rw [exp_sub, exp_log (by positivity), exp_log (by positivity)]
  simp_rw [two_mul, exp_add, sq]

theorem gepo_denominator_bounds {n : ℕ} [NeZero n] (q : Fin n → ℝ) {lo hi : ℝ}
    (hp : ∀ i, 0 < q i) (hl : ∀ i, lo ≤ q i) (hh : ∀ i, q i ≤ hi) :
    lo ≤ (∑ i, (q i)^2) / (∑ i, q i) ∧ (∑ i, (q i)^2) / (∑ i, q i) ≤ hi := by
  have pos : 0 < ∑ i, q i := sum_pos (fun i _ => hp i) univ_nonempty
  constructor
  · rw [le_div_iff₀ pos, mul_sum]
    exact sum_le_sum fun i _ => by simpa [sq] using mul_le_mul_of_nonneg_right (hl i) (hp i).le
  · rw [div_le_iff₀ pos, mul_sum]
    exact sum_le_sum fun i _ => by simpa [sq] using mul_le_mul_of_nonneg_right (hh i) (hp i).le

def k3 (d : ℝ) := exp d - d - 1

theorem k3_nonnegative (d : ℝ) : 0 ≤ k3 d := by
  unfold k3
  linarith [add_one_le_exp d]

/-- Summing only over the sampling support leaves the difference of the
two restricted masses. A reference with positive mass outside that support
therefore does not satisfy the usual unbiased k3 identity. -/
theorem k3_mass_remainder {n : ℕ} (p q : Fin n → ℝ)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i) :
    ∑ i, p i * k3 (log (q i) - log (p i)) =
    (∑ i, p i * (log (p i) - log (q i))) + ((∑ i, q i) - ∑ i, p i) := by
  have term (i : Fin n) : p i * k3 (log (q i) - log (p i)) =
      (q i - p i) + p i * (log (p i) - log (q i)) := by
    rw [k3, exp_sub, exp_log (hp i), exp_log (hq i)]
    field_simp [(hp i).ne']
    ring
  simp_rw [term]
  rw [sum_add_distrib, sum_sub_distrib, add_comm]

theorem k3_on_policy {n : ℕ} (p q : Fin n → ℝ)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (p_mass : ∑ i, p i = 1) (q_mass : ∑ i, q i = 1) :
    ∑ i, p i * k3 (log (q i) - log (p i)) =
    ∑ i, p i * (log (p i) - log (q i)) := by
  rw [k3_mass_remainder p q hp hq, p_mass, q_mass, sub_self, add_zero]

/-- For integer-grid rewards xᵢ/d the exact cross-multiplied residual
    and variance recover the population moments for every finite group. -/
theorem population_numerators {n : ℕ} (hn : n ≠ 0) (x : Fin n → ℝ)
    {d : ℝ} (hd : 0 < d) (i : Fin n) :
    let t := ∑ j, x j
    let mean := (∑ j, x j/d)/n
    (n*x i-t)/(n*d) = x i/d-mean ∧
    ((n:ℝ)*(∑ j, (x j)^2)-t^2)/(n*d)^2 =
      (∑ j, (x j/d-mean)^2)/n := by
  have h : (n:ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  dsimp only
  rw [← sum_div]
  constructor
  · field_simp
    ring
  · simp_rw [sub_sq, div_pow, sum_add_distrib, sum_sub_distrib,
        ← sum_div, ← sum_mul]
    simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp_rw [← mul_sum, ← sum_div]
    field_simp
    ring

/-- An exact comparison of squared integer numerators implements the
    population-deviation gate, including equality at the cutoff. -/
theorem deviation_gate {v d eps : ℝ} (hv : 0 ≤ v) (hd : 0 < d) (he : 0 ≤ eps) :
    eps < sqrt v/d ↔ (eps*d)^2 < v := by
  rw [lt_div_iff₀ hd]
  constructor
  · intro h
    nlinarith [Real.sq_sqrt hv, mul_nonneg he hd.le, mul_self_lt_mul_self (mul_nonneg he hd.le) h]
  · intro h
    by_contra! le
    nlinarith [Real.sq_sqrt hv, mul_self_le_mul_self (Real.sqrt_nonneg v) le]

/-- A group residual and a possibly different batch variance can use the
    same power-of-two units before the square root and final division. -/
theorem scaled_population_advantage (a eps : ℝ) {v s : ℝ} (hv : 0 ≤ v) (hs : 0 < s) :
    (a/s)/(sqrt (v/s^2)+eps/s) = a/(sqrt v+eps) := by
  rw [sqrt_div hv, sqrt_sq hs.le, ← add_div, div_div_div_cancel_right₀ hs.ne']

/-- Magnitudes of a sum and sum of squares on the binary64 2^-1074 grid
    fit the signed 271-byte and unsigned 533-byte DDP fields respectively. -/
theorem moment_packet_bounds {n x : ℕ} (hn : n ≤ 2^63) (hx : x ≤ 2^2098) :
    n*x < 2^(8*271-1) ∧ n*x*x < 2^(8*533) := by
  constructor
  · apply lt_of_le_of_lt (Nat.mul_le_mul hn hx)
    rw [← pow_add]
    exact pow_lt_pow_right₀ (by decide) (by decide)
  · apply lt_of_le_of_lt (Nat.mul_le_mul (Nat.mul_le_mul hn hx) hx)
    rw [← pow_add, ← pow_add]
    exact pow_lt_pow_right₀ (by decide) (by decide)

theorem centered_normalized {n : ℕ} (hn : n ≠ 0) (r : Fin n → ℝ) (scale : ℝ) :
    ∑ i, (r i - (∑ j, r j) / n) / scale = 0 := by
  rw [← sum_div, sum_sub_distrib]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have h : (n : ℝ) * ((∑ i, r i) / n) = ∑ i, r i := by
    field_simp [Nat.cast_ne_zero.mpr hn]
  rw [h, sub_self, zero_div]

/-- With binary verifier rewards, empirical variance is accuracy*(1-accuracy).
Thus the default DAPO strict interior filter retains exactly nonconstant
binary groups. Separate continuous rewards and length shaping can differ. -/
theorem binary_group_variance {n : ℕ} (hn : n ≠ 0) (r : Fin n → ℝ)
    (binary : ∀ i, r i = 0 ∨ r i = 1) :
    let p := (∑ i, r i) / n
    (∑ i, (r i-p)^2) / n = p*(1-p) := by
  let p := (∑ i, r i) / n
  have term (i : Fin n) : (r i-p)^2 = r i - (2*p)*r i + p^2 := by
    rcases binary i with h | h <;> rw [h] <;> ring
  change (∑ i, (r i-p)^2) / n = p*(1-p)
  simp_rw [term, sum_add_distrib, sum_sub_distrib, ← mul_sum]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, p]
  field_simp [Nat.cast_ne_zero.mpr hn]
  ring

theorem binary_filter_criterion {p : ℝ} (hp : 0 ≤ p) (hp' : p ≤ 1) :
    0 < p*(1-p) ↔ 0 < p ∧ p < 1 := by
  constructor
  · intro h
    constructor
    · exact lt_of_le_of_ne hp (fun eq => by rw [← eq] at h; linarith)
    · exact lt_of_le_of_ne hp' (fun eq => by rw [eq] at h; linarith)
  · exact fun h => mul_pos h.1 (sub_pos.mpr h.2)

theorem leave_one_out {n : ℝ} (hn : 1 < n) (r total : ℝ) :
    r - (total-r)/(n-1) = (n/(n-1)) * (r-total/n) := by
  field_simp [show n ≠ 0 by linarith, show n-1 ≠ 0 by linarith]
  ring

/-- Count-weighted means compose across nonempty microbatches or ranks. -/
theorem merge_means (s t m n : ℝ) (hm : 0 < m) (hn : 0 < n) :
    (m * (s/m) + n * (t/n))/(m+n) = (s+t)/(m+n) := by
  field_simp [hm.ne', hn.ne', (add_pos hm hn).ne']

theorem constant_length_reduction (s tokens rows length : ℝ) (ht : tokens ≠ 0) :
    s / (rows*length) = (s/tokens) * (tokens/(rows*length)) := by
  field_simp

/-- Negative log-sigmoid as softplus of the negated margin. -/
def preferenceLoss (beta margin : ℝ) := log (1 + exp (-beta * margin))

theorem preference_improves {b m m' : ℝ} (hb : 0 < b) (hm : m < m') :
    preferenceLoss b m' < preferenceLoss b m := by
  apply log_lt_log (by positivity)
  apply add_lt_add_left
  exact exp_lt_exp.mpr (mul_lt_mul_of_neg_left hm (neg_lt_zero.mpr hb))

theorem dpo_prompt_cancels (chosen rejected refChosen refRejected prefixScore refPrefix : ℝ) :
    ((prefixScore+chosen) - (refPrefix+refChosen)) -
    ((prefixScore+rejected) - (refPrefix+refRejected)) =
    (chosen - rejected) - (refChosen - refRejected) := by ring

end
end StateSetFormal.Policy
