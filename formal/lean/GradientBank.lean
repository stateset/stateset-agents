import PolicyMath

namespace StateSetFormal.GradientBank
open Finset
open scoped BigOperators
noncomputable section

/-- A coordinate of one native GRPO gradient bank. Each rank accumulates
local mean gradients weighted by their local counts at a fixed scale. DDP
averages across ranks. Finalization rescales by the true window population. -/
theorem finalize {r b : ℕ} (hr : r ≠ 0)
    (count grad : Fin r → Fin b → ℝ) (scale : ℝ) (hs : scale ≠ 0)
    (hc : (∑ i, ∑ j, count i j) ≠ 0) :
    ((∑ i, ∑ j, grad i j * (count i j / scale)) / r) *
      (r * scale / (∑ i, ∑ j, count i j)) =
    (∑ i, ∑ j, grad i j * count i j) / (∑ i, ∑ j, count i j) := by
  simp_rw [← mul_div_assoc, ← sum_div]
  field_simp [Nat.cast_ne_zero.mpr hr, hs, hc]
  ring

/-- The native reducer now sums the rank banks and applies the population
scale once. A restored window can use any common nonzero scale. -/
theorem finalize_sum {r b : ℕ} (count grad : Fin r → Fin b → ℝ)
    (scale : ℝ) (hs : scale ≠ 0) (hc : (∑ i, ∑ j, count i j) ≠ 0) :
    (∑ i, ∑ j, grad i j * (count i j / scale)) * (scale / (∑ i, ∑ j, count i j)) =
    (∑ i, ∑ j, grad i j * count i j) / (∑ i, ∑ j, count i j) := by
  simp_rw [← mul_div_assoc, ← sum_div]
  field_simp [hs, hc]

/-- Re-averaging an already all-reduced coordinate is idempotent. -/
theorem average_replicated {r : ℕ} (hr : r ≠ 0) (x : ℝ) :
    (∑ _i : Fin r, x) / r = x := by
  simp [nsmul_eq_mul, Nat.cast_ne_zero.mpr hr]

/-- One accepted sample extends the empirical diagonal Fisher's online mean.
An unused parameter on this sample contributes g = 0. -/
theorem fisher_online (total g : ℝ) {n : ℝ} (hn : 0 < n) :
    total/n + (1/(n+1)) * (g^2 - total/n) = (total+g^2)/(n+1) := by
  field_simp [hn.ne', (show 0 < n+1 by linarith).ne']
  ring

theorem fisher_consolidation {decay old newF : ℝ}
    (hd : 0 ≤ decay) (ho : 0 ≤ old) (hn : 0 ≤ newF) :
    0 ≤ old * decay + newF := add_nonneg (mul_nonneg ho hd) hn

def ewc (strength fisher anchor x : ℝ) := strength / 2 * fisher * (x-anchor)^2

theorem ewc_nonnegative {n : ℕ} {s : ℝ} (f a x : Fin n → ℝ) (hs : 0 ≤ s)
    (hf : ∀ i, 0 ≤ f i) : 0 ≤ ∑ i, ewc s (f i) (a i) (x i) := by
  exact sum_nonneg fun i _ =>
    mul_nonneg (mul_nonneg (by positivity) (hf i)) (sq_nonneg _)

theorem ewc_gradient (s f a x : ℝ) :
    HasDerivAt (ewc s f a) (s*f*(x-a)) x := by
  have h := (((hasDerivAt_id x).sub_const a).pow 2).const_mul (s / 2 * f)
  convert h using 1
  dsimp
  ring

theorem ewc_anchor (s f a : ℝ) :
    ewc s f a a = 0 ∧ HasDerivAt (ewc s f a) 0 a := by
  exact ⟨by simp [ewc], by simpa using ewc_gradient s f a a⟩

/-- For scalar supervised reward regression, the excess sum of squares over
    the empirical mean is exactly the squared prediction bias times count. -/
theorem reward_mse_decomposition {n : ℕ} (hn : n ≠ 0) (r : Fin n → ℝ) (x : ℝ) :
    let mean := (∑ i, r i) / n
    ∑ i, (r i-x)^2 = (∑ i, (r i-mean)^2) + n*(x-mean)^2 := by
  let m := (∑ i, r i) / n
  have centered : ∑ i, (r i-m) = 0 := by
    simpa [m] using Policy.centered_normalized hn r 1
  have term (i : Fin n) : (r i-x)^2 =
      (r i-m)^2 + (x-m)^2 - (r i-m)*(2*(x-m)) := by ring
  simp_rw [term, sum_sub_distrib, sum_add_distrib, ← sum_mul, centered]
  simp [m, nsmul_eq_mul]

/-- Welford's update is the difference of exact centered sums of squares.
The online recurrence uses population variance, with a zero M2 at one sample. -/
theorem welford_update {n : ℕ} (hn : n ≠ 0) (r : Fin n → ℝ) (x : ℝ) :
    let m := (∑ i, r i) / n
    let next := m + (x-m) / ((n:ℝ)+1)
    (∑ i, (r i-m)^2) + (x-m)*(x-next) = (∑ i, (r i-next)^2) + (x-next)^2 := by
  let m := (∑ i, r i) / n
  let next := m + (x-m) / ((n:ℝ)+1)
  change (∑ i, (r i-m)^2) + (x-m)*(x-next) = (∑ i, (r i-next)^2) + (x-next)^2
  have h := reward_mse_decomposition hn r next
  change (∑ i, (r i-next)^2) = (∑ i, (r i-m)^2) + n*(next-m)^2 at h
  rw [h, add_assoc]
  congr 1
  dsimp only [next]
  field_simp [show (n:ℝ)+1 ≠ 0 by positivity]
  ring

/-- A frozen affine z-score transform with a positive standard deviation,
nonnegative target scale and projection preserves the ordering of rewards. -/
theorem calibrated_monotone {x y mean sd target scale lo hi : ℝ} (h : x ≤ y)
    (hs : 0 < sd) (ht : 0 ≤ scale) :
    max lo (min hi (((x-mean)/sd)*scale + target)) ≤
    max lo (min hi (((y-mean)/sd)*scale + target)) := by
  exact max_le_max_left _ (min_le_min_left _ (add_le_add_right
    (mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right (sub_le_sub_right h _) hs.le) ht) _))


end
end StateSetFormal.GradientBank
