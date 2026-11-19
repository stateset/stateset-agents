import ScoreFunction

/-! Finite replay laws for group-relative score gradients at a fixed prompt.
    The peer outcome can encode an entire tuple of other sampled responses. -/
namespace StateSetFormal.ReplayGradient
open Finset
open scoped BigOperators
noncomputable section

def average {n m : ℕ} (p : Fin n → ℝ) (q : Fin m → ℝ) (f : Fin n → Fin m → ℝ) :=
  ∑ i, ∑ j, p i * q j * f i j

/-- An exact ratio for the scored response changes its marginal. The old
    distribution of the peer group still enters every group-relative advantage. -/
theorem row_ratio {n m : ℕ} (p mu : Fin n → ℝ) (peers : Fin m → ℝ)
    (f : Fin n → Fin m → ℝ) (support : ∀ i, mu i ≠ 0) :
    average mu peers (fun i j => p i / mu i * f i j) = average p peers f := by
  unfold average
  apply sum_congr rfl
  intro i _
  apply sum_congr rfl
  intro j _
  field_simp [support i]
  ring

/-- Exact signed replay bias per coordinate, allowing every response and peer
    outcome to have a different score-times-advantage contribution. -/
theorem peer_bias {n m : ℕ} (p : Fin n → ℝ) (old fresh : Fin m → ℝ)
    (f : Fin n → Fin m → ℝ) :
    average p old f - average p fresh f = average p (fun j => old j-fresh j) f := by
  simp only [average, mul_sub, sub_mul, sum_sub_distrib]

/-- A response-independent peer baseline cancels when the target score has
    zero mean. This includes leave-one-out with any independent peer law. -/
theorem independent_peer_baseline {n m : ℕ} (p s r : Fin n → ℝ) (q b : Fin m → ℝ)
    (norm : ∑ j, q j = 1) (score : ∑ i, p i*s i = 0) :
    average p q (fun i j => s i*(r i-b j)) = ∑ i, p i*s i*r i := by
  unfold average
  rw [sum_comm]
  have term (i : Fin n) (j : Fin m) :
      p i*q j*(s i*(r i-b j)) = q j*((p i*s i)*r i-(p i*s i)*b j) := by ring
  simp_rw [term, ← mul_sum, sum_sub_distrib, ← sum_mul, score, zero_mul, sub_zero]
  rw [← sum_mul, norm, one_mul]

/-- Keeping the scored response in a raw group mean attenuates the gradient
    by (G−1)/G under either peer law. No variance normalization is used. -/
theorem raw_group_mean {n m : ℕ} (p s r : Fin n → ℝ) (q others : Fin m → ℝ) (g : ℝ)
    (size : 1 < g) (norm : ∑ j, q j = 1) (score : ∑ i, p i*s i = 0) :
    average p q (fun i j => s i*(r i-(r i+others j)/g)) =
      ((g-1)/g) * ∑ i, p i*s i*r i := by
  rw [← independent_peer_baseline p s r q (fun j => others j/(g-1)) norm score]
  unfold average
  simp only [mul_sum]
  apply sum_congr rfl
  intro i _
  apply sum_congr rfl
  intro j _
  rw [ScoreFunction.self_including_baseline _ _ size]
  ring

/-- For an arbitrary (possibly priority-selected) joint replay law, multiplying
    by the target joint mass divided by the actual draw mass recovers the fresh
    group expectation on its support. The second sum is the exact missing term. -/
theorem joint_support {n m : ℕ} (p : Fin n → ℝ) (q : Fin m → ℝ)
    (draw f : Fin n → Fin m → ℝ) :
    average p q f =
      (∑ i, ∑ j, draw i j * (p i*q j/draw i j) * f i j) +
      ∑ i, ∑ j, (if draw i j = 0 then p i*q j*f i j else 0) := by
  unfold average
  rw [← sum_add_distrib]
  apply sum_congr rfl
  intro i _
  rw [← sum_add_distrib]
  apply sum_congr rfl
  intro j _
  by_cases h : draw i j = 0
  · simp [h]
  · rw [if_neg h, add_zero]
    field_simp [h]

/-- Product of the individual likelihood ratios is the joint ratio for any
    finite set of independent samples with positive behavior mass. -/
theorem group_ratio {k : ℕ} (p mu : Fin k → ℝ) (positive : ∀ i, mu i ≠ 0) :
    (∏ i, mu i) * (∏ i, p i/mu i) = ∏ i, p i := by
  rw [← prod_mul_distrib]
  apply prod_congr rfl
  intro i _
  field_simp [positive i]

/-- A range bound on peer-dependent contributions gives a total-variation
    bound for each coordinate of the replay update after row correction. -/
theorem bias_bound {n m : ℕ} (p : Fin n → ℝ) (old fresh : Fin m → ℝ)
    (f : Fin n → Fin m → ℝ) (center : Fin n → ℝ) (radius : ℝ)
    (hp : ∀ i, 0 ≤ p i) (mass : ∑ i, p i = 1) (peers : ∑ j, old j = ∑ j, fresh j)
    (bounded : ∀ i j, |f i j-center i| ≤ radius) :
    |average p old f-average p fresh f| ≤ (∑ j, |old j-fresh j|)*radius := by
  have shifted (i : Fin n) : ∑ j, p i*(old j-fresh j)*(f i j-center i) =
      ∑ j, p i*(old j-fresh j)*f i j := by
    simp only [mul_sub, sum_sub_distrib, ← sum_mul, ← mul_sum, peers, sub_self, zero_mul, sub_zero]
  rw [peer_bias, average]
  simp_rw [← shifted]
  calc
    _ ≤ ∑ i, |∑ j, p i*(old j-fresh j)*(f i j-center i)| := abs_sum_le_sum_abs ..
    _ ≤ ∑ i, ∑ j, p i * |old j-fresh j| * radius := by
      apply sum_le_sum
      intro i _
      apply (abs_sum_le_sum_abs ..).trans
      apply sum_le_sum
      intro j _
      simp only [abs_mul, abs_of_nonneg (hp i)]
      exact mul_le_mul_of_nonneg_left (bounded i j) (mul_nonneg (hp i) (abs_nonneg _))
    _ = _ := by simp only [← sum_mul, ← mul_sum, mass, one_mul]

/-- Conditional laws around a scored token. The last outcome may include
    later tokens, rewards and all other responses in the group. -/
def tokenAverage {h a z : ℕ} (prefixLaw : Fin h → ℝ) (next : Fin h → Fin a → ℝ)
    (suffix f : Fin h → Fin a → Fin z → ℝ) :=
  ∑ t, ∑ i, ∑ j, prefixLaw t * next t i * suffix t i j * f t i j

/-- Exact conditional-token importance weighting leaves separate errors from
    the visited prefixes and from the future/peer law used for the advantage. -/
theorem token_bias {h a z : ℕ} (oldPrefix freshPrefix : Fin h → ℝ)
    (mu p : Fin h → Fin a → ℝ) (oldTail freshTail f : Fin h → Fin a → Fin z → ℝ)
    (support : ∀ t i, mu t i ≠ 0) :
    tokenAverage oldPrefix mu oldTail (fun t i j => p t i/mu t i*f t i j) -
      tokenAverage freshPrefix p freshTail f =
    tokenAverage (fun t => oldPrefix t-freshPrefix t) p freshTail f +
      tokenAverage oldPrefix p (fun t i j => oldTail t i j-freshTail t i j) f := by
  unfold tokenAverage
  simp only [← sum_sub_distrib, ← sum_add_distrib]
  apply sum_congr rfl
  intro t _
  apply sum_congr rfl
  intro i _
  apply sum_congr rfl
  intro j _
  field_simp [support t i]
  ring

private def current (_ : Fin 3) : ℝ := 1/3
private def behavior (i : Fin 3) : ℝ := if i=0 then 1/2 else 1/4
private def advantage (i j : Fin 3) : ℝ := if i>j then 1 else if i<j then -1 else 0
private def coordinate (i : Fin 3) : ℝ := if i=1 then 2/3 else -1/3

/-- Three ordered rewards, groups of two, zero normalization epsilon, and the
    middle softmax logit. Current, uncorrected, row-corrected and clipped
    expected ascent coordinates are all different. No ratio is at a clip kink. -/
theorem middle_logit :
    let f := fun i j => coordinate i * advantage i j
    average current current f = 0 ∧
    average behavior behavior f = 1/16 ∧
    average current behavior f = 1/36 ∧
    average current behavior (fun i j =>
      if (advantage i j > 0 ∧ current i / behavior i > 6/5) ∨
         (advantage i j < 0 ∧ current i / behavior i < 4/5) then 0 else f i j) = -1/18 := by
  dsimp only
  simp only [average, Fin.sum_univ_three]
  simp (config := { decide := true }) only [current, behavior, advantage, coordinate, ↓reduceIte]
  norm_num

/-- The same coordinate for arbitrary normalized advantage magnitudes on
    reward gaps one and two, including a gap suppressed by the epsilon floor.
    A zero term can follow either clipping branch without changing the sum. -/
theorem middle_by_gap (near far : ℝ) :
    let f := fun i j => coordinate i * advantage i j *
      (if (i=0 ∧ j=2) ∨ (i=2 ∧ j=0) then far else near)
    average current current f = 0 ∧
    average behavior behavior f = near/16 ∧
    average current behavior f = (2*near-far)/36 ∧
    average current behavior (fun i j =>
      if (i>j ∧ current i / behavior i > 6/5) ∨
         (i<j ∧ current i / behavior i < 4/5) then 0 else f i j) = -near/18 := by
  dsimp only
  simp only [average, Fin.sum_univ_three]
  simp (config := { decide := true }) only [current, behavior, advantage, coordinate, ↓reduceIte]
  norm_num
  constructor
  · ring
  constructor
  · ring
  constructor <;> ring

end
end StateSetFormal.ReplayGradient
