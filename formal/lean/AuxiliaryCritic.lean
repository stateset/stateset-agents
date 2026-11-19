import LoggedBellman
import AdamHistory

/-! A frozen-label auxiliary TD update with independent value and Q optimizers. -/
namespace StateSetFormal.AuxiliaryCritic
open Finset
open scoped BigOperators
noncomputable section

private theorem relu_slope {f : ℝ → ℝ} {j a : ℝ} (h : HasDerivAt f j a) (hz : f a ≠ 0) :
    HasDerivAt (fun t => max (f t) 0) (if 0 < f a then j else 0) a := by
  rcases lt_or_gt_of_ne hz with hn | hp
  · rw [if_neg (not_lt.mpr hn.le)]
    apply (hasDerivAt_const a (0:ℝ)).congr_of_eventuallyEq
    exact (h.continuousAt.eventually_lt continuousAt_const hn).mono fun t ht => max_eq_right ht.le
  · rw [if_pos hp]
    apply h.congr_of_eventuallyEq
    exact (continuousAt_const.eventually_lt h.continuousAt hp).mono fun t ht => max_eq_left ht.le

/-- The analytic Jacobian for a two-layer ReLU critic along a parameter
direction. The hidden preactivations and output weights are affine on the
line; their derivatives can both contribute. Kinks are excluded. -/
theorem relu_head {d : ℕ} (z dz w dw : Fin d → ℝ) (b db : ℝ) (hn : ∀ i, z i ≠ 0) :
    HasDerivAt (fun t => (b+db*t) + ∑ i, (w i+dw i*t)*max (z i+dz i*t) 0)
      (db + ∑ i, (dw i*max (z i) 0 + w i*(if 0 < z i then dz i else 0))) 0 := by
  have affine (x v : ℝ) : HasDerivAt (fun t : ℝ => x+v*t) v 0 := by
    simpa using (((hasDerivAt_id (0:ℝ)).const_mul v).const_add x)
  apply (affine b db).add
  apply HasDerivAt.sum
  intro i _
  have h := relu_slope (affine (z i) (dz i)) (by simpa using hn i)
  simpa using (affine (w i) (dw i)).mul h

/-- Finite prediction drift remains bounded even when an optimizer step
crosses a ReLU kink. The preactivations include the hidden biases. -/
theorem relu_drift {d : ℕ} (z z' w w' : Fin d → ℝ) (b b' : ℝ) :
    |(b' + ∑ i, w' i*max (z' i) 0) - (b + ∑ i, w i*max (z i) 0)| ≤
      |b'-b| + ∑ i, (|w' i-w i| * max (z' i) 0 + |w i| * |z' i-z i|) := by
  have lip (x y : ℝ) : |max x 0 - max y 0| ≤ |x-y| := by
    have h₁ := le_abs_self (x-y)
    have h₂ := neg_le_abs (x-y)
    rw [abs_le]
    simp only [max_def]
    split_ifs <;> constructor <;> linarith
  rw [add_sub_add_comm, ← sum_sub_distrib]
  apply (abs_add_le ..).trans
  apply add_le_add_left
  apply (abs_sum_le_sum_abs _ _).trans
  apply sum_le_sum
  intro i _
  rw [show w' i*max (z' i) 0 - w i*max (z i) 0 =
    (w' i-w i)*max (z' i) 0 + w i*(max (z' i) 0-max (z i) 0) by ring]
  apply (abs_add_le ..).trans
  rw [abs_mul, abs_mul, abs_of_nonneg (le_max_right (z' i) 0)]
  exact add_le_add_left (mul_le_mul_of_nonneg_left (lip _ _) (abs_nonneg _)) _

/-- The targets used by both regressions are fixed during backpropagation.
The supplied Jacobian can describe a linear head or a differentiable network. -/
theorem frozen_gradient {n : ℕ} (f : Fin n → ℝ → ℝ) (y j : Fin n → ℝ) (a : ℝ)
    (hj : ∀ i, HasDerivAt (f i) (j i) a) :
    HasDerivAt (fun t => (∑ i, (f i t-y i)^2)/(n:ℝ))
      ((∑ i, 2*(f i a-y i)*j i)/(n:ℝ)) a := by
  apply HasDerivAt.div_const
  apply HasDerivAt.sum
  intro i _
  convert (((hj i).sub_const (y i)).pow 2) using 1
  ring

/-- Substituting labels from another target snapshot shifts the mean gradient
by the label difference weighted by each row's prediction derivative. -/
theorem snapshot_difference {n : ℕ} (u y z j : Fin n → ℝ) :
    (∑ i, 2*(u i-y i)*j i)/(n:ℝ) - (∑ i, 2*(u i-z i)*j i)/(n:ℝ) =
    (∑ i, 2*(z i-y i)*j i)/(n:ℝ) := by
  rw [← sub_div, ← sum_sub_distrib]
  congr 1
  apply sum_congr rfl
  intro i _
  ring

def advance (s : AdamHistory.LocalState) (g b₁ b₂ lr decay eps : ℝ) : AdamHistory.LocalState :=
  let m := b₁*s.m+(1-b₁)*g
  let v := b₂*s.v+(1-b₂)*g^2
  ⟨Optimizer.adamW s.w lr decay eps (m/(1-b₁^(s.t+1))) (v/(1-b₂^(s.t+1))),
    m,v,s.t+1⟩

/-- Restoring a coordinate's saved weight, moments and clock gives exactly
the update from its independent full gradient history, including clipping. -/
theorem restored_coordinate (w b₁ b₂ lr decay eps clip raw : ℝ)
    (past : List AdamHistory.Event) :
    let next := advance (AdamHistory.run w b₁ b₂ decay eps past) (clip*raw) b₁ b₂ lr decay eps
    let history := (some (clip*raw), lr)::past
    next.w = AdamHistory.affine w (AdamHistory.terms b₁ b₂ decay eps history) ∧
    next.t = (AdamHistory.gradients history).length ∧
    next.m = Optimizer.moment b₁ (AdamHistory.gradients history) ∧
    next.v = Optimizer.moment b₂ ((AdamHistory.gradients history).map fun g => g^2) := by
  let history : List AdamHistory.Event := (some (clip*raw), lr)::past
  exact ⟨AdamHistory.weight_from_history w b₁ b₂ decay eps history,
    AdamHistory.local_moments w b₁ b₂ decay eps history⟩

def linear {d : ℕ} (w x : Fin d → ℝ) : ℝ := ∑ i, w i*x i

/-- Publishing the *updated* linear value parameters gives the corresponding
interpolation of predictions. Using pre-update weights incurs exactly tau
times the fitted prediction's movement on that state. -/
theorem published_prediction {d : ℕ} (old live updated x : Fin d → ℝ) (τ : ℝ) :
    linear (fun i => (1-τ)*old i+τ*updated i) x -
    linear (fun i => (1-τ)*old i+τ*live i) x = τ*(linear updated x-linear live x) := by
  simp only [linear, add_mul, sum_add_distrib, mul_assoc, ← mul_sum]
  ring

/-- Transport a pre-update value certificate to a linear head after the
optimizer step. The supplied coordinate bounds can include native rounding. -/
theorem after_update {n d : ℕ} (x : Fin n → Fin d → ℝ) (old updated : Fin d → ℝ)
    (V : Fin n → ℝ) (move : Fin d → ℝ) {ε D : ℝ} (hD : 0 ≤ D)
    (prior : dist (fun s => linear old (x s)) V ≤ ε)
    (coord : ∀ i, |updated i-old i| ≤ move i)
    (budget : ∀ s, (∑ i, move i*|x s i|) ≤ D) :
    dist (fun s => linear updated (x s)) V ≤ D+ε := by
  apply (dist_triangle _ (fun s => linear old (x s)) _).trans
  apply add_le_add _ prior
  apply (dist_pi_le_iff hD).mpr
  intro s
  rw [Real.dist_eq]
  change |(∑ i, updated i*x s i)-(∑ i, old i*x s i)| ≤ D
  rw [← sum_sub_distrib]
  apply (abs_sum_le_sum_abs _ _).trans
  apply le_trans _ (budget s)
  apply sum_le_sum
  intro i _
  rw [← sub_mul, abs_mul]
  exact mul_le_mul_of_nonneg_right (coord i) (abs_nonneg _)

end
end StateSetFormal.AuxiliaryCritic
