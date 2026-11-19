import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace StateSetFormal.Returns
open Real
noncomputable section

structure Step where
  reward : ℝ
  value : ℝ
  terminal : Bool

def nextValue (bootstrap : ℝ) : List Step → ℝ
  | [] => bootstrap
  | s :: _ => s.value

def gae (g l bootstrap : ℝ) : List Step → ℝ
  | [] => 0
  | s :: tail =>
    s.reward + (if s.terminal then 0 else g * nextValue bootstrap tail) - s.value +
      (if s.terminal then 0 else g*l*gae g l bootstrap tail)

def lambdaReturn (g l bootstrap : ℝ) : List Step → ℝ
  | [] => bootstrap
  | s :: tail => s.reward + (if s.terminal then 0 else
      g*((1-l)*nextValue bootstrap tail + l*lambdaReturn g l bootstrap tail))

theorem gae_return (g l bootstrap : ℝ) (steps : List Step) :
    gae g l bootstrap steps + nextValue bootstrap steps = lambdaReturn g l bootstrap steps := by
  induction steps with
  | nil => simp [gae, nextValue, lambdaReturn]
  | cons s tail ih =>
    simp only [gae, nextValue, lambdaReturn, ← ih]
    cases s.terminal <;> simp
    ring

theorem terminal_target (g l bootstrap : ℝ) (s : Step) (ht : s.terminal = true)
    (rest : List Step) : gae g l bootstrap (s :: rest) + s.value = s.reward := by
  simp [gae, ht]

def discounted (g bootstrap : ℝ) : List Step → ℝ
  | [] => bootstrap
  | s :: tail => s.reward + (if s.terminal then 0 else g * discounted g bootstrap tail)

theorem lambda_one_mc (g b : ℝ) (steps : List Step) :
    lambdaReturn g 1 b steps = discounted g b steps := by
  induction steps with
  | nil => rfl
  | cons s tail ih => simp [lambdaReturn, discounted, ih]

theorem lambda_zero_td (g b : ℝ) (s : Step) (rest : List Step) :
    lambdaReturn g 0 b (s::rest) =
      s.reward + (if s.terminal then 0 else g*nextValue b rest) := by simp [lambdaReturn]

def adaptiveLambda (alpha length : ℝ) := 1 - 1 / (alpha*length+1)

theorem adaptiveLambda_ratio {a n : ℝ} (h : 0 ≤ a*n) :
    adaptiveLambda a n = a*n/(1+a*n) := by
  unfold adaptiveLambda
  rw [add_comm (a*n) 1]
  field_simp [show (1+a*n) ≠ 0 by linarith]

theorem adaptiveLambda_bounds {a n : ℝ} (ha : 0 < a) (hn : 0 ≤ n) :
    0 ≤ adaptiveLambda a n ∧ adaptiveLambda a n < 1 := by
  have h : 0 < a*n+1 := by positivity
  unfold adaptiveLambda
  constructor
  · have := (div_le_one h).mpr (show 1 ≤ a*n+1 by linarith [mul_nonneg ha.le hn])
    linarith
  · linarith [one_div_pos.mpr h]

/-- Bellman backups with any finite next-action selector (CQL/IQL/BCQ/BEAR). -/
def backup (g reward next : ℝ) (done : Bool) := reward + if done then 0 else g*next

theorem backup_terminal (g reward next : ℝ) : backup g reward next true = reward := by
  simp [backup]

theorem backup_lipschitz (g reward x y : ℝ) (hg : 0 ≤ g) (done : Bool) :
    |backup g reward x done - backup g reward y done| ≤ g*|x-y| := by
  cases done
  · simp only [backup, Bool.false_eq_true, if_false]
    rw [show reward+g*x-(reward+g*y) = g*(x-y) by ring, abs_mul, abs_of_nonneg hg]
  · simp [backup, mul_nonneg hg (abs_nonneg _)]

/-- Expected continuation value under one stochastic transition row is
nonexpansive in sup norm, including transitions of zero probability. -/
theorem expectation_lipschitz {n : ℕ} (p v w : Fin n → ℝ) (eps : ℝ)
    (hp : ∀ i, 0 ≤ p i) (hm : ∑ i, p i = 1) (h : ∀ i, |v i-w i| ≤ eps) :
    |(∑ i, p i*v i) - (∑ i, p i*w i)| ≤ eps := by
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ i, |p i * (v i-w i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, p i*eps := Finset.sum_le_sum fun i _ => by
      rw [abs_mul, abs_of_nonneg (hp i)]
      exact mul_le_mul_of_nonneg_left (h i) (hp i)
    _ = eps := by rw [← Finset.sum_mul, hm, one_mul]

/-- The maximizing candidate may change between updates. Only attainment
and maximality are needed, so ties may be broken either way. -/
theorem greedy_lipschitz {m : ℕ} (v w : Fin m → ℝ) (i j : Fin m) (eps : ℝ)
    (hi : ∀ k, v k ≤ v i) (hj : ∀ k, w k ≤ w j) (h : ∀ k, |v k-w k| ≤ eps) :
    |v i-w j| ≤ eps := by
  exact abs_le.mpr ⟨by linarith [(abs_le.mp (h j)).1, hi j],
                     by linarith [(abs_le.mp (h i)).2, hj i]⟩

def actionBackup {n : ℕ} (γ r : ℝ) (p v : Fin n → ℝ) :=
  r + γ * ∑ i, p i*v i

/-- One row of the finite-MDP Bellman optimality operator contracts by γ
when γ<1. The same bound holds componentwise for every current state. -/
theorem bellman_optimality_lipschitz {m n : ℕ} (γ eps : ℝ) (hg : 0 ≤ γ)
    (p : Fin m → Fin n → ℝ) (r : Fin m → ℝ) (v w : Fin n → ℝ) (i j : Fin m)
    (hp : ∀ a s, 0 ≤ p a s) (hm : ∀ a, ∑ s, p a s = 1)
    (h : ∀ s, |v s-w s| ≤ eps)
    (hi : ∀ a, actionBackup γ (r a) (p a) v ≤ actionBackup γ (r i) (p i) v)
    (hj : ∀ a, actionBackup γ (r a) (p a) w ≤ actionBackup γ (r j) (p j) w) :
    |actionBackup γ (r i) (p i) v - actionBackup γ (r j) (p j) w| ≤ γ*eps := by
  apply greedy_lipschitz _ _ i j (γ*eps) hi hj
  intro a
  unfold actionBackup
  rw [add_sub_add_left_eq_sub, ← mul_sub, abs_mul, abs_of_nonneg hg]
  exact mul_le_mul_of_nonneg_left (expectation_lipschitz (p a) v w eps (hp a) (hm a) h) hg

/-- Polyak target updates and BCQ's convex mixture of ordered twin Q values. -/
theorem convex_bounds {w x y : ℝ} (hw : 0 ≤ w) (hw' : w ≤ 1) (h : x ≤ y) :
    x ≤ w*x+(1-w)*y ∧ w*x+(1-w)*y ≤ y := by
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr hw'),
                              mul_nonneg (sub_nonneg.mpr h) hw]

def expectile (tau error : ℝ) := (if 0 < error then tau else 1-tau) * error^2

theorem expectile_nonnegative (e : ℝ) {t : ℝ} (ht : 0 ≤ t) (ht' : t ≤ 1) :
    0 ≤ expectile t e := by
  unfold expectile
  split <;> apply mul_nonneg _ (sq_nonneg _) <;> linarith

theorem expectile_half (e : ℝ) : expectile (1/2) e = e^2/2 := by
  unfold expectile
  split <;> ring

/-- Minimizing the negative dual objective uses the opposite sign from
    minimizing the actor's penalty. The residual is held fixed. -/
theorem dual_loss_derivative (x violation : ℝ) :
    HasDerivAt (fun z => -exp z * violation) (-exp x * violation) x := by
  exact (Real.hasDerivAt_exp x).neg.mul_const violation

theorem dual_step_increases {x residual eta : ℝ} (h : 0 < residual) (he : 0 < eta) :
    x < x - eta * (-exp x * residual) := by
  nlinarith [mul_pos he (mul_pos (exp_pos x) h)]

end
end StateSetFormal.Returns
