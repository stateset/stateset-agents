import GradientBank
import ScoreFunction

/-! Observed trajectory scores used in native elastic weight consolidation,
and the population curvature of a normalized categorical policy. -/
namespace StateSetFormal.FisherInformation
open Finset Real
open scoped BigOperators
noncomputable section

/-- A coordinate of the negative mean log likelihood derivative. The token
index ranges only over the responses, across all turns of one trajectory. -/
theorem trajectory_gradient {n : ℕ} (logp : Fin n → ℝ → ℝ) (score : Fin n → ℝ)
    {x : ℝ} (h : ∀ i, HasDerivAt (logp i) (score i) x) :
    HasDerivAt (fun y => -(∑ i, logp i y) / n) (-(∑ i, score i) / n) x :=
  ((HasDerivAt.sum fun i _ => h i).neg).div_const _

/-- The gap between averaging squared token scores and squaring the observed
trajectory's mean score is exactly the within-trajectory population variance. -/
theorem token_gap {n : ℕ} (hn : n ≠ 0) (s : Fin n → ℝ) :
    let m := (∑ i, s i) / n
    (∑ i, (s i)^2) / n - (-m)^2 = (∑ i, (s i-m)^2) / n := by
  dsimp only
  have h := GradientBank.reward_mse_decomposition hn s 0
  simp only [sub_zero, zero_sub, neg_sq] at h ⊢
  generalize (∑ i, s i) / (n : ℝ) = m at h ⊢
  rw [h]
  field_simp [Nat.cast_ne_zero.mpr hn]

/-- Uniform trajectory sampling gives each accepted trajectory one vote,
including trajectories whose signed token scores cancel. Lengths may differ. -/
theorem dataset_token_bound {m : ℕ} (len : Fin m → ℕ) (hl : ∀ j, len j ≠ 0)
    (s : (j : Fin m) → Fin (len j) → ℝ) :
    (∑ j, (-(∑ i, s j i) / len j)^2) / m ≤
      (∑ j, (∑ i, (s j i)^2) / len j) / m := by
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply sum_le_sum
  intro j _
  have h := token_gap (hl j) (s j)
  dsimp only at h
  rw [neg_div]
  have hp : 0 ≤ (∑ i, (s j i - (∑ k, s j k) / len j)^2) / len j := by positivity
  linarith

/-- At an anchor x, the curvature of KL(p(x) || p(y)) in a scalar
parameter is the model expectation of the squared score. The expectation
uses the policy's own action probabilities at x. -/
theorem kl_curvature {n : ℕ} (p dp : Fin n → ℝ → ℝ) (ddp : Fin n → ℝ) (x : ℝ)
    (hp : ∀ i y, 0 < p i y) (hm : ∀ y, ∑ i, p i y = 1)
    (hd : ∀ i y, HasDerivAt (p i) (dp i y) y)
    (hdd : ∀ i, HasDerivAt (dp i) (ddp i) x) :
    HasDerivAt (fun y => deriv (fun z => ∑ i, p i x * (log (p i x) - log (p i z))) y)
      (∑ i, p i x * (deriv (fun z => log (p i z)) x)^2) x := by
  have first (y : ℝ) :
      deriv (fun z => ∑ i, p i x * (log (p i x) - log (p i z))) y =
        ∑ i, p i x * (0 - dp i y / p i y) := by
    exact (HasDerivAt.sum fun i _ =>
      ((hasDerivAt_const y (log (p i x))).sub
        ((hd i y).log (hp i y).ne')).const_mul (p i x)).deriv
  simp_rw [first]
  have second (i : Fin n) :
      HasDerivAt (fun y => p i x * (0 - dp i y / p i y))
        (p i x * (deriv (fun z => log (p i z)) x)^2 - ddp i) x := by
    rw [((hd i x).log (hp i x).ne').deriv]
    convert ((hasDerivAt_const x (0:ℝ)).sub
      ((hdd i).div (hd i x) (hp i x).ne')).const_mul (p i x) using 1
    field_simp [(hp i x).ne']
    ring
  have dz (y : ℝ) : ∑ i, dp i y = 0 :=
    ScoreFunction.mass_derivative (fun i => hd i y) hm
  have ddz : ∑ i, ddp i = 0 := by
    have h := (HasDerivAt.sum fun i (_ : i ∈ univ) => hdd i).deriv
    simpa only [dz, deriv_const] using h.symm
  simpa only [sum_sub_distrib, ddz, sub_zero] using
    (HasDerivAt.sum fun i (_ : i ∈ univ) => second i)

/-- A prefix score is orthogonal to a later action score when the latter
has conditional mean zero at every prefix. This permits dependent actions. -/
theorem conditional_cross {n m : ℕ} (p a : Fin n → ℝ) (q b : Fin n → Fin m → ℝ)
    (hz : ∀ i, ∑ j, q i j * b i j = 0) :
    ∑ i, ∑ j, p i * q i j * a i * b i j = 0 := by
  have term (i : Fin n) (j : Fin m) :
      p i * q i j * a i * b i j = (p i * a i) * (q i j * b i j) := by ring
  simp_rw [term, ← mul_sum, hz, mul_zero, sum_const_zero]

/-- Under a path law with zero cross-score moments, squaring a fixed-length
trajectory's mean score divides the sum of the token Fisher diagonals by
length squared. Logged actions need not satisfy the cross-moment hypothesis. -/
theorem fixed_length_information {n paths : ℕ} (p : Fin paths → ℝ)
    (s : Fin n → Fin paths → ℝ)
    (h : ∀ i j, i ≠ j → ∑ k, p k * s i k * s j k = 0) :
    (∑ k, p k * (-(∑ i, s i k) / n)^2) =
      (∑ i, ∑ k, p k * (s i k)^2) / (n : ℝ)^2 := by
  have per (k : Fin paths) : p k * (-(∑ i, s i k) / n)^2 =
      (∑ i, ∑ j, p k * s i k * s j k) / (n : ℝ)^2 := by
    simp_rw [← mul_sum, mul_assoc, ← mul_sum, ← sum_mul, div_pow, neg_sq, sq]
    ring
  simp_rw [per, ← sum_div]
  congr 1
  rw [sum_comm]
  apply sum_congr rfl
  intro i _
  rw [sum_comm, sum_eq_single i]
  · simp_rw [sq, mul_assoc]
  · intro j _ hj
    exact h i j (Ne.symm hj)
  · simp

/-- Bernoulli logit score for a sampled bit b is b-p. A one-action log
may underestimate or overestimate the policy expectation. -/
theorem binary_mismatch :
    let p : ℝ := 1/4
    let model := p*(1-p)^2 + (1-p)*(0-p)^2
    model = 3/16 ∧ (0-p)^2 = 1/16 ∧ (1-p)^2 = 9/16 := by norm_num

/-- For a two-logit categorical policy with probabilities p and 1-p, the
directional score of either action is its logit slope minus the mean slope. -/
theorem logit_score (a b u v : ℝ) :
    let p := exp a / (exp a + exp b)
    HasDerivAt (fun t => a+t*u - log (exp (a+t*u) + exp (b+t*v)))
      ((1-p)*(u-v)) 0 := by
  dsimp only
  have ha := ((hasDerivAt_id (0:ℝ)).mul_const u).const_add a
  have hb := ((hasDerivAt_id (0:ℝ)).mul_const v).const_add b
  have h := ha.sub ((ha.exp.add hb.exp).log (by dsimp; positivity))
  convert h using 1
  dsimp
  simp only [zero_mul, add_zero, one_mul]
  field_simp [show exp a + exp b ≠ 0 by positivity]
  ring

/-- Discarding the two off-diagonal Fisher entries can increase or decrease
the quadratic form, even at the exact on-policy population. -/
theorem diagonal_gap (p u v : ℝ) :
    let full := p*((1-p)*(u-v))^2 + (1-p)*(p*(v-u))^2
    let diagonal := p*(1-p)*(u^2+v^2)
    full = p*(1-p)*(u-v)^2 ∧ diagonal-full = 2*p*(1-p)*u*v := by
  constructor <;> ring

theorem cancelling_pair (s : ℝ) :
    ((s + -s)/2)^2 = 0 ∧ (s^2 + (-s)^2)/2 = s^2 := by constructor <;> ring

end
end StateSetFormal.FisherInformation
