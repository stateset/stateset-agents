import PolicyMath

namespace StateSetFormal.Optimizer
open Finset Real
open scoped BigOperators
noncomputable section

/-- Adam's moment starting at zero, with the newest observation first. -/
def moment (beta : ℝ) : List ℝ → ℝ
  | [] => 0
  | g :: past => beta * moment beta past + (1-beta)*g

theorem moment_constant (beta g : ℝ) (steps : ℕ) :
    moment beta (List.replicate steps g) = (1-beta^steps)*g := by
  induction steps with
  | zero => simp [moment]
  | succ n ih => rw [List.replicate_succ, moment, ih, pow_succ]; ring

/-- For any gradient history the recurrence is exactly a geometric
weighted sum, with index zero denoting the most recent observation. -/
theorem moment_expansion (beta : ℝ) {n : ℕ} (g : Fin n → ℝ) :
    moment beta (List.ofFn g) = (1-beta) * ∑ i : Fin n, beta^(i:ℕ) * g i := by
  induction n with
  | zero => simp [moment]
  | succ n ih =>
    rw [List.ofFn_succ, moment, ih (fun i => g i.succ), Fin.sum_univ_succ]
    simp only [Fin.val_zero, pow_zero, one_mul, Fin.val_succ, pow_succ]
    have weighted : (∑ i : Fin n, beta^(i:ℕ) * beta * g i.succ) =
        beta * ∑ i : Fin n, beta^(i:ℕ) * g i.succ := by
      rw [mul_sum]
      exact sum_congr rfl fun i _ => by ring
    rw [weighted]
    ring

/-- Bias correction normalizes the weights even at the first update. -/
theorem corrected_constant {b : ℝ} (hb : 0 ≤ b) (hb1 : b < 1) (g : ℝ)
    {n : ℕ} (hn : n ≠ 0) :
    moment b (List.replicate n g) / (1-b^n) = g := by
  rw [moment_constant]
  have h : b^n < 1 := pow_lt_one₀ hb hb1 hn
  field_simp [show 1-b^n ≠ 0 by linarith]

theorem second_moment_nonnegative {b : ℝ} (hb : 0 ≤ b) (hb1 : b ≤ 1) (gs : List ℝ) :
    0 ≤ moment b (gs.map (fun g => g^2)) := by
  induction gs with
  | nil => simp [moment]
  | cons g gs ih =>
    exact add_nonneg (mul_nonneg hb ih) (mul_nonneg (sub_nonneg.mpr hb1) (sq_nonneg g))

def adamW (w lr decay eps mHat vHat : ℝ) :=
  (1-lr*decay)*w - lr*mHat/(sqrt vHat + eps)

theorem adaptive_descent {lr eps m v w decay : ℝ} (hl : 0 < lr) (he : 0 < eps)
    (hm : 0 < m) : adamW w lr decay eps m v < (1-lr*decay)*w := by
  exact sub_lt_self _ (div_pos (mul_pos hl hm) (add_pos_of_nonneg_of_pos (sqrt_nonneg _) he))

/-- The nonnegative PyTorch L2 clipping coefficient bounds the resulting
norm by both the cap and the original norm, including a zero cap. -/
theorem clip_bounds {norm cap eps : ℝ} (hn : 0 ≤ norm) (hc : 0 ≤ cap) (he : 0 < eps) :
    let k := min 1 (cap/(norm+eps))
    0 ≤ k ∧ k*norm ≤ norm ∧ k*norm ≤ cap := by
  let k := min 1 (cap/(norm+eps))
  have hden : 0 < norm+eps := by linarith
  have h0 : 0 ≤ k := le_min zero_le_one (div_nonneg hc hden.le)
  have h1 : k ≤ 1 := min_le_left ..
  have hk : k*(norm+eps) ≤ cap := (le_div_iff₀ hden).mp (min_le_right ..)
  exact ⟨h0, by simpa using mul_le_mul_of_nonneg_right h1 hn,
    (mul_le_mul_of_nonneg_left (by linarith : norm ≤ norm+eps) h0).trans hk⟩

/-- A common positive scale can be divided out of the norm and epsilon
before constructing a clipping coefficient. -/
theorem clip_scaled {s : ℝ} (hs : 0 < s) (cap norm eps : ℝ) :
    min 1 ((cap/s) / (norm/s + eps/s)) = min 1 (cap/(norm+eps)) := by
  rw [← add_div, div_div_div_cancel_right₀ hs.ne']

/-- Splitting off a normal power of two preserves the real product at each
iteration, even when the unsplit coefficient would be below binary64 range. -/
theorem split_coefficient (g m : ℝ) (exponent shift : ℤ) :
    (g*(2:ℝ)^(-shift))*(m*(2:ℝ)^(exponent+shift)) = g*(m*(2:ℝ)^exponent) := by
  rw [zpow_add₀ (by norm_num : (2:ℝ) ≠ 0), zpow_neg]
  field_simp
  ring

/-- Computing the norm of scaled coordinates produces the scaled L2 norm. -/
theorem scaled_l2 {n : ℕ} (g : Fin n → ℝ) {s : ℝ} (hs : 0 < s) :
    sqrt (∑ i, (g i/s)^2) = sqrt (∑ i, (g i)^2)/s := by
  simp_rw [div_pow, ← sum_div]
  rw [Real.sqrt_div (sum_nonneg fun i _ => sq_nonneg (g i)), Real.sqrt_sq hs.le]

/-- Both norm bounds hold for the actual vector of scaled coordinates. -/
theorem clipped_l2 {n : ℕ} (g : Fin n → ℝ) {cap eps : ℝ} (hc : 0 ≤ cap) (he : 0 < eps) :
    let norm := sqrt (∑ i, (g i)^2)
    let k := min 1 (cap/(norm+eps))
    let result := sqrt (∑ i, (k*g i)^2)
    result ≤ norm ∧ result ≤ cap := by
  let norm := sqrt (∑ i, (g i)^2)
  let k := min 1 (cap/(norm+eps))
  have bounds := clip_bounds (norm := norm) (sqrt_nonneg _) hc he
  change sqrt (∑ i, (k*g i)^2) ≤ norm ∧ sqrt (∑ i, (k*g i)^2) ≤ cap
  simp_rw [mul_pow, ← mul_sum]
  rw [Real.sqrt_mul (sq_nonneg k), Real.sqrt_sq bounds.1]
  exact bounds.2

/-- Standard Adam first step, including the epsilon outside the square root. -/
theorem first_step {b1 b2 : ℝ} (h1 : 0 ≤ b1) (h1' : b1 < 1) (h2 : 0 ≤ b2)
    (h2' : b2 < 1) (w g lr decay eps : ℝ) :
    adamW w lr decay eps (moment b1 [g] / (1-b1)) (moment b2 [g^2] / (1-b2)) =
      (1-lr*decay)*w - lr*g/(|g|+eps) := by
  have m := corrected_constant h1 h1' g (n := 1) one_ne_zero
  have v := corrected_constant h2 h2' (g^2) (n := 1) one_ne_zero
  simp only [List.replicate_succ, List.replicate_zero, pow_one] at m v
  rw [m, v, adamW, sqrt_sq_eq_abs]

end
end StateSetFormal.Optimizer
