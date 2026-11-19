import ScoreFunction
import Optimizer

namespace StateSetFormal.ActorCritic
open Finset Real
open scoped BigOperators
noncomputable section

/-- The derivative of negative categorical entropy along a normalized policy.
The constant part of the derivative of p log p cancels across the actions. -/
theorem neg_entropy_gradient {n : ℕ} (p : Fin n → ℝ → ℝ) (dp : Fin n → ℝ) (x : ℝ)
    (hp : ∀ i, 0 < p i x) (hd : ∀ i, HasDerivAt (p i) (dp i) x)
    (mass : ∀ y, ∑ i, p i y = 1) :
    HasDerivAt (fun y => ∑ i, p i y * log (p i y))
      (∑ i, dp i * log (p i x)) x := by
  have term (i : Fin n) : HasDerivAt (fun y => p i y * log (p i y))
      (dp i * log (p i x) + dp i) x := by
    convert (hd i).mul ((hd i).log (hp i).ne') using 1
    field_simp [(hp i).ne']
  have h := HasDerivAt.sum (u := univ) (fun i _ => term i)
  simpa only [sum_add_distrib, ScoreFunction.mass_derivative hd mass, add_zero] using h

/-- The sampled actor, critic and entropy terms may share parameters. The
rollout advantage and lambda-return target are constants during this update. -/
theorem shared_gradient {n : ℕ} (p : Fin n → ℝ → ℝ) (dp : Fin n → ℝ)
    (v : ℝ → ℝ) (dv x advantage target critic entropy : ℝ) (action : Fin n)
    (hp : ∀ i, 0 < p i x) (hd : ∀ i, HasDerivAt (p i) (dp i) x)
    (mass : ∀ y, ∑ i, p i y = 1) (hv : HasDerivAt v dv x) :
    HasDerivAt (fun y => -advantage * log (p action y) + critic * (v y-target)^2 +
        entropy * ∑ i, p i y * log (p i y))
      (-advantage * (dp action / p action x) + 2 * critic * (v x-target)*dv +
        entropy * ∑ i, dp i * log (p i x)) x := by
  have ha := ((hd action).log (hp action).ne').const_mul (-advantage)
  have hc := (((hv.sub_const target).pow 2).const_mul critic)
  have he := (neg_entropy_gradient p dp x hp hd mass).const_mul entropy
  convert (ha.add hc).add he using 1
  ring

def prob {n : ℕ} (z : Fin n → ℝ) (i : Fin n) := exp (z i) / ∑ j, exp (z j)

/-- The derivative of each categorical probability as a single logit moves. -/
theorem coordinate_derivative {n : ℕ} [NeZero n] (z : Fin n → ℝ) (k i : Fin n) :
    HasDerivAt (fun y => prob (fun j => z j + if j = k then y else 0) i)
      (prob z i * ((if i = k then 1 else 0) - prob z k)) 0 := by
  have atom (j : Fin n) : HasDerivAt (fun y : ℝ => exp (z j + if j = k then y else 0))
      (if j = k then exp (z j) else 0) 0 := by
    by_cases h : j = k
    · subst j
      simpa using (((hasDerivAt_id (0:ℝ)).const_add (z k)).exp)
    · simp [h, hasDerivAt_const]
  have den := HasDerivAt.sum (u := univ) (fun j _ => atom j)
  have nz : (∑ j : Fin n, exp (z j)) ≠ 0 := by positivity
  unfold prob
  convert (atom i).div den (by simpa using nz) using 1
  by_cases h : i = k <;> simp [h] <;> field_simp [nz] <;> ring

/-- Contracting the categorical Jacobian with log probabilities gives the
negative-entropy sensitivity used by the actor head. -/
theorem entropy_coordinate {n : ℕ} (p : Fin n → ℝ) (k : Fin n) :
    (∑ i, p i * ((if i = k then 1 else 0) - p k) * log (p i)) =
      p k * (log (p k) - ∑ i, p i * log (p i)) := by
  have term (i : Fin n) : p i * ((if i = k then 1 else 0) - p k) * log (p i) =
      (if i = k then p i * log (p i) else 0) - p k * (p i * log (p i)) := by
    by_cases h : i = k <;> simp [h] <;> ring
  simp_rw [term, sum_sub_distrib, ← mul_sum]
  simp [mul_sub]

/-- Moving logit k changes the fixed-draw loss by the signed policy residual
plus the diversity term. A value head depending on the same coordinate also
contributes its prediction-error slope. -/
theorem logit_loss_gradient {n : ℕ} [NeZero n] (z : Fin n → ℝ) (action k : Fin n)
    (v : ℝ → ℝ) (dv advantage target critic entropy : ℝ) (hv : HasDerivAt v dv 0) :
    let p := fun y => prob (fun j => z j + if j = k then y else 0)
    HasDerivAt (fun y => -advantage * log (p y action) + critic * (v y-target)^2 +
        entropy * ∑ i, p y i * log (p y i))
      (advantage * (prob z k - (if action = k then 1 else 0)) +
        2 * critic * (v 0 - target)*dv +
        entropy * prob z k * (log (prob z k) - ∑ i, prob z i * log (prob z i))) 0 := by
  let p := fun (i : Fin n) (y : ℝ) => prob (fun j => z j + if j = k then y else 0) i
  have atZero (i : Fin n) : p i 0 = prob z i := by simp [p]
  have hp (i : Fin n) : 0 < p i 0 := by rw [atZero, prob]; positivity
  have hm (y : ℝ) : ∑ i, p i y = 1 := by
    simp only [p, prob, ← sum_div, div_self (by positivity : (∑ j, exp (z j + if j = k then y else 0)) ≠ 0)]
  have h := shared_gradient p _ v dv 0 advantage target critic entropy action hp
    (coordinate_derivative z k) hm hv
  simp_rw [atZero] at h
  rw [entropy_coordinate] at h
  convert h using 1
  have pos := hp action
  rw [atZero] at pos
  field_simp [pos.ne']
  ring

/-- Each uncentered RMSprop coordinate has bounded displacement even with
arbitrary previous gradients, when the current second moment includes g². -/
theorem rmsprop_bound (g v lr alpha eps : ℝ) (hv : 0 ≤ v) (hl : 0 ≤ lr)
    (ha : 0 ≤ alpha) (ha' : alpha < 1) (he : 0 < eps) :
    |lr*g/(sqrt (alpha*v+(1-alpha)*g^2)+eps)| ≤ lr/sqrt (1-alpha) := by
  have h := sub_pos.mpr ha'
  have hs := sqrt_pos.mpr h
  have hd : 0 < sqrt (alpha*v+(1-alpha)*g^2)+eps := by positivity
  have lower : sqrt ((1-alpha)*g^2) ≤ sqrt (alpha*v+(1-alpha)*g^2) :=
    sqrt_le_sqrt (le_add_of_nonneg_left (mul_nonneg ha hv))
  rw [sqrt_mul h.le, sqrt_sq_eq_abs] at lower
  rw [abs_div, abs_mul, abs_of_nonneg hl, abs_of_pos hd, div_le_div_iff₀ hd hs]
  calc
    lr * |g| * sqrt (1-alpha) = lr * (sqrt (1-alpha) * |g|) := by ring
    _ ≤ lr * (sqrt (alpha*v+(1-alpha)*g^2)+eps) :=
      mul_le_mul_of_nonneg_left (by linarith) hl

end
end StateSetFormal.ActorCritic
