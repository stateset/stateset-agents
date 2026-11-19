import PolicyMath
import Returns

namespace StateSetFormal.PPOBatch
open StateSetFormal.Returns StateSetFormal.Policy Finset Real
open scoped BigOperators
noncomputable section

/-- Reward-only and value-only contributions to a lambda return. A time
    limit supplies the last bootstrap, and a terminal transition cuts both. -/
def rewardTrace (γ lam : ℝ) : List Step → ℝ
  | [] => 0
  | s :: rest => s.reward + if s.terminal then 0 else γ * lam * rewardTrace γ lam rest

def valueTrace (γ lam b : ℝ) : List Step → ℝ
  | [] => b
  | s :: rest => if s.terminal then 0 else
      γ * ((1-lam) * nextValue b rest + lam * valueTrace γ lam b rest)

theorem split_return (γ lam b : ℝ) (steps : List Step) :
    rewardTrace γ lam steps + valueTrace γ lam b steps = lambdaReturn γ lam b steps := by
  induction steps with
  | nil => simp [rewardTrace, valueTrace, lambdaReturn]
  | cons s rest ih =>
    simp only [rewardTrace, valueTrace, lambdaReturn, ← ih]
    cases s.terminal <;> simp
    ring

theorem split_advantage (γ lam b : ℝ) (steps : List Step) :
    rewardTrace γ lam steps + (valueTrace γ lam b steps - nextValue b steps) = gae γ lam b steps := by
  linarith [split_return γ lam b steps, gae_return γ lam b steps]

theorem recover_residual (r v rounded current : ℝ) :
    (rounded-current) + ((r-rounded)+v) = r+v-current := by ring

def valueError (v old y radius : ℝ) : ℝ :=
  max ((v-y)^2) ((old + max (-radius) (min radius (v-old)) - y)^2)

theorem projected_value (v old radius : ℝ) :
    old + max (-radius) (min radius (v-old)) =
      max (old-radius) (min (old+radius) v) := by
  rw [add_max, add_min]
  simp only [add_sub_cancel, sub_eq_add_neg]
  ring_nf

theorem pessimistic_value (v old y radius : ℝ) : (v-y)^2 ≤ valueError v old y radius :=
  le_max_left _ _

theorem scaled_value_error (v old y radius s : ℝ) :
    max (((v-y)*s)^2) (((max (old-radius) (min (old+radius) v) - y)*s)^2) =
      valueError v old y radius*s^2 := by
  rw [valueError, projected_value, max_mul_of_nonneg _ _ (sq_nonneg s)]
  simp only [mul_pow]

theorem value_inside {v old y radius : ℝ} (lo : old-radius ≤ v) (hi : v ≤ old+radius) :
    valueError v old y radius = (v-y)^2 := by
  simp [valueError, projected_value, min_eq_right hi, max_eq_right lo]

/-- For fixed visited prefixes, the expected token-average stopping statistic
    is the behavior-to-current categorical KL averaged over those prefixes. -/
theorem behavior_drift {n k : ℕ} [NeZero n] (p q : Fin n → Fin k → ℝ)
    (hp : ∀ i j, 0 < p i j) (hq : ∀ i j, 0 < q i j)
    (pm : ∀ i, ∑ j, p i j = 1) (qm : ∀ i, ∑ j, q i j = 1) :
    (∑ i, ∑ j, p i j * k3 (log (q i j) - log (p i j))) / n =
    (∑ i, ∑ j, p i j * (log (p i j) - log (q i j))) / n := by
  simp_rw [fun i => k3_on_policy (p i) (q i) (hp i) (hq i) (pm i) (qm i)]

/-- The integer enclosures in PPOEpochs.tla bound the true divergence of
    both nontrivial action ratios in the model. -/
theorem epoch_table :
    (2703 < 3000 * k3 (log 3) ∧ 3000 * k3 (log 3) < 2706) ∧
    (1294 < 3000 * k3 (log (3:ℝ)⁻¹) ∧ 3000 * k3 (log (3:ℝ)⁻¹) < 1297) := by
  have pos : 0 < (2:ℝ)/3 := by norm_num
  have h := Real.abs_log_sub_add_sum_range_le (x := (2:ℝ)/3) (by rw [abs_of_pos pos]; norm_num) 24
  rw [show 1-(2:ℝ)/3 = (3:ℝ)⁻¹ by norm_num, Real.log_inv] at h
  rw [abs_of_pos (by norm_num : 0 < (2:ℝ)/3)] at h
  norm_num [Finset.sum_range_succ] at h
  have bounds := abs_le.mp h
  unfold k3
  rw [Real.exp_log (by norm_num : (0:ℝ) < 3), Real.exp_log (by norm_num : (0:ℝ) < (3:ℝ)⁻¹)]
  simp only [Real.log_inv]
  constructor <;> constructor <;> linarith [bounds.1, bounds.2]

theorem single_token_stops {n : ℕ} [NeZero n] (gap : Fin n → ℝ) (cap : ℝ) (j : Fin n)
    (h : (n : ℝ) * cap < k3 (gap j)) : cap < (∑ i, k3 (gap i)) / n := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  apply (lt_div_iff₀ hn).mpr
  rw [mul_comm]
  exact h.trans_le (single_le_sum (fun i _ => k3_nonnegative (gap i)) (mem_univ j))

/-- Clipping the probability ratio cannot enforce a KL limit. Positive
    advantages already saturate above the clip although divergence can grow. -/
theorem clipping_allows_drift {hi ratio1 ratio2 adv : ℝ}
    (hh : 1 ≤ hi) (h₁ : hi ≤ ratio1) (h₂ : hi ≤ ratio2) (ha : 0 ≤ adv) :
    surrogate 1 hi ratio1 adv = surrogate 1 hi ratio2 adv := by
  simp [surrogate_positive hh ha, min_eq_left h₁, min_eq_left h₂]

theorem stop_at_first_crossing (cap failed : ℝ) (accepted later : List ℝ)
    (passes : ∀ x ∈ accepted, x ≤ cap) (crossed : cap < failed) :
    ((accepted ++ failed :: later).takeWhile (fun x => decide (x ≤ cap))) = accepted := by
  induction accepted with
  | nil => simp [List.takeWhile, not_le.mpr crossed]
  | cons x xs ih =>
    have hx := passes x (List.mem_cons_self _ _)
    have hr := ih (fun y hy => passes y (List.mem_cons_of_mem _ hy))
    simp [List.takeWhile, hx, hr]

end
end StateSetFormal.PPOBatch
