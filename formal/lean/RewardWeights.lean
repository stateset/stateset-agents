import ScoreFunction

namespace StateSetFormal.RewardWeights
open Finset Real
open scoped BigOperators
noncomputable section

/-- A configured judge score is clipped to its finite rubric, followed by
an affine map into the unit interval. -/
def judgeScore (lo hi x : ℝ) := (min hi (max lo x)-lo)/(hi-lo)

theorem judge_bounds {lo hi : ℝ} (h : lo < hi) (x : ℝ) :
    0 ≤ judgeScore lo hi x ∧ judgeScore lo hi x ≤ 1 := by
  have lower := le_min h.le (le_max_left lo x)
  have upper := min_le_left hi (max lo x)
  exact ⟨div_nonneg (sub_nonneg.mpr lower) (sub_pos.mpr h).le,
    (div_le_one (sub_pos.mpr h)).mpr (sub_le_sub_right upper _)⟩

theorem judge_monotone {lo hi : ℝ} (h : lo < hi) : Monotone (judgeScore lo hi) := by
  intro x y hxy
  exact div_le_div_of_nonneg_right
    (sub_le_sub_right (min_le_min_left hi (max_le_max_left lo hxy)) lo) (sub_pos.mpr h).le

theorem judge_recover {lo hi : ℝ} (h : lo < hi) (x : ℝ) :
    judgeScore lo hi x * (hi-lo) + lo = min hi (max lo x) := by
  rw [judgeScore, div_mul_cancel₀ _ (sub_pos.mpr h).ne']
  ring

/-- Repeating a completion score over its actual action mask is the same
as weighting that score by the count of retained tokens in the row. -/
theorem repeat_rows {b t : ℕ} (mask : Fin b → Finset (Fin t)) (f : Fin b → ℝ) :
    (∑ i, ∑ _j ∈ mask i, f i) = ∑ i, ((mask i).card : ℝ)*f i := by
  simp [nsmul_eq_mul]

def mean {b : ℕ} (c r : Fin b → ℝ) := (∑ i, c i*r i)/(∑ i, c i)

/-- A shared judge offset cancels from every multi-token centered reward,
including when completion lengths differ. -/
theorem offset_cancels {b : ℕ} (c r : Fin b → ℝ) (hc : (∑ i, c i) ≠ 0)
    (offset : ℝ) (i : Fin b) :
    r i+offset-mean c (fun j => r j+offset) = r i-mean c r := by
  unfold mean
  simp_rw [mul_add, sum_add_distrib, ← sum_mul]
  field_simp [hc]
  ring

theorem constant_residual {b : ℕ} (c : Fin b → ℝ) (hc : (∑ i, c i) ≠ 0)
    (r s : ℝ) : (r-mean c (fun _ => r))/s = 0 := by
  simp [mean, ← sum_mul, mul_div_cancel_left₀ r hc]

theorem centered {b : ℕ} (c r : Fin b → ℝ) (hc : (∑ i, c i) ≠ 0) (s : ℝ) :
    (∑ i, c i*((r i-mean c r)/s)) = 0 := by
  simp_rw [← mul_div_assoc, ← sum_div, mul_sub, sum_sub_distrib, ← sum_mul]
  rw [mean]
  field_simp [hc]

/-- The sum of squared deviations agrees with subtracting the squared
weighted first moment from the weighted raw second moment. -/
theorem variance_numerator {b : ℕ} (c r : Fin b → ℝ) (hc : (∑ i, c i) ≠ 0) :
    (∑ i, c i*(r i-mean c r)^2) =
      (∑ i, c i*(r i)^2) - (∑ i, c i*r i)^2/(∑ i, c i) := by
  have term (i : Fin b) : c i*(r i-mean c r)^2 =
      c i*(r i)^2 - c i*r i*(2*mean c r) + c i*(mean c r)^2 := by ring
  simp_rw [term, sum_add_distrib, sum_sub_distrib, ← sum_mul]
  unfold mean
  field_simp [hc]
  ring

/-- Positive scaling preserves every standardized weight when epsilon and
the sample deviation are expressed in the same units. -/
theorem scaled_weight {b : ℕ} (c d : Fin b → ℝ) (hc : ∀ i, 0 ≤ c i)
    (hn : 1 < ∑ i, c i) {s : ℝ} (hs : 0 < s) (eps : ℝ) (i : Fin b) :
    let n := ∑ j, c j
    let sd := sqrt ((∑ j, c j*(d j)^2)/(n-1))
    let scaled := sqrt ((∑ j, c j*(d j/s)^2)/(n-1))
    (d i/s)/(scaled+eps/s) = d i/(sd+eps) := by
  have v : 0 ≤ (∑ j, c j*(d j)^2)/((∑ j, c j)-1) :=
    div_nonneg (sum_nonneg fun j _ => mul_nonneg (hc j) (sq_nonneg _)) (by linarith)
  dsimp only
  simp_rw [div_pow, ← mul_div_assoc, ← sum_div]
  rw [div_right_comm _ (s^2), sqrt_div v, sqrt_sq hs.le, ← add_div, div_div_div_cancel_right₀ hs.ne']

/-- The sample-standardized squared weights have total at most N-1. Adding
epsilon can only shrink them, including when all judged rewards coincide. -/
theorem standardized_mass {b : ℕ} (c d : Fin b → ℝ) (hc : ∀ i, 0 ≤ c i)
    (hn : 1 < ∑ i, c i) {eps : ℝ} (he : 0 < eps) :
    let n := ∑ i, c i
    let sd := sqrt ((∑ i, c i*(d i)^2)/(n-1))
    (∑ i, c i*(d i/(sd+eps))^2) ≤ n-1 := by
  let n := (∑ i, c i)-1
  let q := ∑ i, c i*(d i)^2
  let sd := sqrt (q/n)
  have hn' : 0 < n := sub_pos.mpr hn
  have hq : 0 ≤ q := sum_nonneg fun i _ => mul_nonneg (hc i) (sq_nonneg _)
  have hs : 0 ≤ sd := sqrt_nonneg _
  have hsq : sd^2 = q/n := Real.sq_sqrt (div_nonneg hq hn'.le)
  change (∑ i, c i*(d i/(sd+eps))^2) ≤ n
  simp_rw [div_pow, ← mul_div_assoc, ← sum_div]
  apply (div_le_iff₀ (sq_pos_of_pos (add_pos_of_nonneg_of_pos hs he))).mpr
  have h := (eq_div_iff hn'.ne').mp hsq
  calc
    q = n*sd^2 := by linarith
    _ ≤ _ := mul_le_mul_of_nonneg_left (by nlinarith [mul_nonneg hs he.le, sq_nonneg eps]) hn'.le

/-- The row-wise policy derivative is the mean of all retained token
derivatives under their completion's frozen weight. -/
theorem row_loss_gradient {b t : ℕ} (mask : Fin b → Finset (Fin t)) (w : Fin b → ℝ)
    (logp : Fin b → Fin t → ℝ → ℝ) (slope : Fin b → Fin t → ℝ) (x n : ℝ)
    (hd : ∀ i, ∀ j ∈ mask i, HasDerivAt (logp i j) (slope i j) x) :
    HasDerivAt (fun y => (∑ i, -w i*(∑ j ∈ mask i, logp i j y))/n)
      ((∑ i, ∑ j ∈ mask i, -w i*slope i j)/n) x := by
  simp_rw [← mul_sum]
  exact (HasDerivAt.sum fun i _ => (HasDerivAt.sum (hd i)).const_mul (-w i)).div_const _

/-- For an active action the slope is its logit's slope minus the categorical
average. An inactive position contributes a constant zero and has zero slope. -/
theorem masked_causal_score {v : ℕ} (active : Bool) (action : Fin v)
    (z : Fin v → ℝ → ℝ) (g : Fin v → ℝ) (x : ℝ)
    (hd : ∀ i, HasDerivAt (z i) (g i) x) :
    HasDerivAt (fun t => if active then z action t-log (∑ i, exp (z i t)) else 0)
      (if active then g action-(∑ i, exp (z i x)*g i)/(∑ i, exp (z i x)) else 0) x := by
  cases active with
  | false => simpa using (hasDerivAt_const x (0 : ℝ))
  | true =>
    have hp : 0 < ∑ i, exp (z i x) :=
      sum_pos' (fun i _ => (exp_pos _).le) ⟨action, mem_univ _, exp_pos _⟩
    simpa using (hd action).sub ((HasDerivAt.sum fun i _ => (hd i).exp).log hp.ne')

end
end StateSetFormal.RewardWeights
