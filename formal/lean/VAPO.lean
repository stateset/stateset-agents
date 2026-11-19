import RegularizedBatch
import PPOBatch

namespace StateSetFormal.VAPO
open Finset Real
open scoped BigOperators
noncomputable section

/-- Sum of the residual coefficients over a finite remaining trajectory.
At q = 1, the weights are all one. The identity also covers an empty tail. -/
theorem trace_mass (q : ℝ) (n : ℕ) :
    (1-q) * (∑ k ∈ Finset.range n, q^k) = 1-q^n := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, mul_add, ih, pow_succ]
    ring

def project (l u v : ℝ) := max l (min u v)
def valueError (l u target v : ℝ) := max ((v-target)^2) ((project l u v-target)^2)

/-- Clipping a *prediction* in the pessimistic critic loss is equivalent
to placing a fixed floor under the raw square. The floor is the squared
distance of the frozen target from the rollout trust interval. -/
theorem critic_floor {l u : ℝ} (ordered : l ≤ u) (t v : ℝ) :
    valueError l u t v = max ((v-t)^2) ((project l u t-t)^2) := by
  have sqL {a b : ℝ} (h : a ≤ b) (hb : b ≤ 0) : b^2 ≤ a^2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr h) (show 0 ≤ -a-b by linarith)]
  have sqR {a b : ℝ} (ha : 0 ≤ a) (h : a ≤ b) : a^2 ≤ b^2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr h) (show 0 ≤ b+a by linarith)]
  have below (x : ℝ) (h : x ≤ l) : project l u x = l := by
    simp [project, min_eq_right (h.trans ordered), max_eq_left h]
  have inside (x : ℝ) (h : l ≤ x) (h' : x ≤ u) : project l u x = x := by
    simp [project, min_eq_right h', max_eq_right h]
  have above (x : ℝ) (h : u ≤ x) : project l u x = u := by
    simp [project, min_eq_left h, max_eq_right ordered]
  unfold valueError
  rcases le_total v l with vl | lv
  · rw [below v vl]
    rcases le_total t l with tl | lt
    · rw [below t tl]
    rcases le_total t u with tu | ut
    · rw [inside t lt tu, sub_self, zero_pow (by decide : 2 ≠ 0), max_eq_left (sq_nonneg _),
          max_eq_left (sqL (by linarith) (by linarith))]
    · rw [above t ut, max_eq_left (sqL (by linarith) (by linarith)),
          max_eq_left (sqL (by linarith) (by linarith))]
  rcases le_total v u with vu | uv
  · rw [inside v lv vu, max_self]
    rcases le_total t l with tl | lt
    · rw [below t tl, max_eq_left (sqR (by linarith) (by linarith))]
    rcases le_total t u with tu | ut
    · simp [inside t lt tu, sq_nonneg]
    · rw [above t ut, max_eq_left (sqL (by linarith) (by linarith))]
  · rw [above v uv]
    rcases le_total t l with tl | lt
    · rw [below t tl, max_eq_left (sqR (by linarith) (by linarith)),
          max_eq_left (sqR (by linarith) (by linarith))]
    rcases le_total t u with tu | ut
    · rw [inside t lt tu, sub_self, zero_pow (by decide : 2 ≠ 0), max_eq_left (sq_nonneg _),
          max_eq_left (sqR (by linarith) (by linarith))]
    · rw [above t ut]

theorem critic_minimizers {l u : ℝ} (ordered : l ≤ u) (t v : ℝ) :
    valueError l u t v = valueError l u t t ↔ |v-t| ≤ |project l u t-t| := by
  rw [critic_floor ordered, critic_floor ordered t t, sub_self, zero_pow (by decide : 2 ≠ 0),
    max_eq_right (sq_nonneg _), max_eq_right_iff, sq_le_sq]

/-- Identify the traced PPO/VAPO production critic kernel with the floor. -/
theorem kernel_floor (v old t r : ℝ) (hr : 0 ≤ r) :
    PPOBatch.valueError v old t r = max ((v-t)^2) ((project (old-r) (old+r) t-t)^2) := by
  rw [PPOBatch.valueError, PPOBatch.projected_value]
  exact critic_floor (by linarith) t v

/-- All convex choices of slope at a clipping tie are valid subgradients.
The plateau has zero slope and the strict quadratic region has the usual
squared-error derivative. -/
theorem critic_subgradient {l u : ℝ} (ordered : l ≤ u) (t v w theta : ℝ)
    (low : 0 ≤ theta) (high : theta ≤ 1)
    (flat : (v-t)^2 < (project l u t-t)^2 → theta=0)
    (raw : (project l u t-t)^2 < (v-t)^2 → theta=1) :
    valueError l u t v + (theta*(2*(v-t)))*(w-v) ≤ valueError l u t w := by
  rw [critic_floor ordered, critic_floor ordered]
  let f := (project l u t-t)^2
  change max ((v-t)^2) f + _ ≤ max ((w-t)^2) f
  have quad : (v-t)^2 + 2*(v-t)*(w-v) ≤ (w-t)^2 := by nlinarith [sq_nonneg (w-v)]
  rcases lt_trichotomy ((v-t)^2) f with hf | he | hr
  · rw [flat hf, max_eq_right hf.le]
    simp
  · have a := mul_le_mul_of_nonneg_left (quad.trans (le_max_left ((w-t)^2) f)) low
    have b := mul_le_mul_of_nonneg_left (le_max_right ((w-t)^2) f) (sub_nonneg.mpr high)
    rw [max_eq_right he.le]
    nlinarith [a, b, congrArg (fun x => theta*x) he]
  · rw [raw hr, max_eq_left hr.le, one_mul]
    exact quad.trans (le_max_left _ _)

/-- PyTorch splits a tie in maximum equally. Clamp passes its input
derivative at both endpoints. -/
def maxWeight (a b : ℝ) : ℝ := if a < b then 0 else if a=b then 1/2 else 1
def criticSlope (l u t v : ℝ) :=
  let z := project l u v
  let a := (v-t)^2
  let b := (z-t)^2
  (if l ≤ v ∧ v ≤ u then maxWeight b a * (2*(z-t)) else 0) +
    maxWeight a b * (2*(v-t))

theorem clip_inside {l u t v : ℝ} (lo : l ≤ v) (hi : v ≤ u) :
    criticSlope l u t v = 2*(v-t) := by
  simp [criticSlope, project, min_eq_right hi, max_eq_right lo, lo, hi, maxWeight]
  ring

/-- The slope selected by the implemented clamp and maximum gives a global
supporting line, including coincident clip bounds and nondifferentiable ties. -/
theorem selected_subgradient {l u : ℝ} (ordered : l ≤ u) (t v w : ℝ) :
    valueError l u t v + criticSlope l u t v * (w-v) ≤ valueError l u t w := by
  let z := project l u v
  let a := (v-t)^2
  let b := (z-t)^2
  let f := (project l u t-t)^2
  have raw (h : b ≤ a) (hs : criticSlope l u t v = 2*(v-t)) :
      valueError l u t v + criticSlope l u t v * (w-v) ≤ valueError l u t w := by
    rw [hs, valueError, max_eq_left h]
    apply le_trans _ (le_max_left ((w-t)^2) _)
    dsimp [a]
    nlinarith [sq_nonneg (w-v)]
  by_cases hin : l ≤ v ∧ v ≤ u
  · apply raw _ (clip_inside hin.1 hin.2)
    simp [a, b, z, project, min_eq_right hin.2, max_eq_right hin.1]
  have hs : criticSlope l u t v = maxWeight a b * (2*(v-t)) := by
    simp [criticSlope, hin, a, b, z]
  have hcase : b = f ∨ b < a := by
    by_cases hl : v < l
    · have hz : z = l := by simp [z, project, min_eq_right (hl.le.trans ordered), max_eq_left hl.le]
      by_cases ht : t ≤ l
      · left
        simp [b, f, hz, project, min_eq_right (ht.trans ordered), max_eq_left ht]
      · right
        dsimp [b, a]
        rw [hz]
        nlinarith [mul_pos (sub_pos.mpr hl) (show 0 < 2*t-l-v by linarith)]
    · have hu : u < v := by
        by_contra! huv
        exact hin ⟨le_of_not_gt hl, huv⟩
      have hz : z = u := by simp [z, project, min_eq_left hu.le, max_eq_right ordered]
      by_cases ht : u ≤ t
      · left
        simp [b, f, hz, project, min_eq_left ht, max_eq_right ordered]
      · right
        dsimp [b, a]
        rw [hz]
        nlinarith [mul_pos (sub_pos.mpr hu) (show 0 < v+u-2*t by linarith)]
  rcases hcase with hf | hb
  · rw [hs]
    apply critic_subgradient ordered t v w (maxWeight a b)
    · unfold maxWeight
      split_ifs <;> norm_num
    · unfold maxWeight
      split_ifs <;> norm_num
    · intro h
      change a < f at h
      simp [maxWeight, hf, h]
    · intro h
      change f < a at h
      simp [maxWeight, hf, not_lt_of_ge h.le, ne_of_gt h]
  · apply raw hb.le
    simp [hs, maxWeight, not_lt_of_ge hb.le, ne_of_gt hb]

/-- Holding the rollout targets fixed, the shared-backbone derivative
contains the regularized actor term, the token-mean critic term and the
mean NLL on verified-positive tokens. At a critic-only parameter the two
actor derivatives vanish. An empty positive population has zero numerator. -/
theorem joint_gradient {n : ℕ} (tokens pos : Fin n → ℝ)
    (actor value nll : Fin n → ℝ → ℝ) (da dv dl : Fin n → ℝ)
    (x c w : ℝ) (hv0 : ∀ i, tokens i = 0 → dv i = 0)
    (hl0 : ∀ i, pos i = 0 → dl i = 0)
    (ha : ∀ i, HasDerivAt (actor i) (da i) x)
    (hv : ∀ i, HasDerivAt (value i) (dv i) x) (hl : ∀ i, HasDerivAt (nll i) (dl i) x) :
    HasDerivAt (fun y =>
      (∑ i, actor i y) + c * (∑ i, (tokens i / ∑ j, tokens j) * (value i y / tokens i)) +
      w * (∑ i, (pos i / ∑ j, pos j) * (nll i y / pos i)))
      ((∑ i, da i) + c * ((∑ i, dv i) / ∑ i, tokens i) + w * ((∑ i, dl i) / ∑ i, pos i)) x := by
  rw [← RegularizedBatch.population_mean tokens dv hv0, ← RegularizedBatch.population_mean pos dl hl0]
  exact ((HasDerivAt.sum fun i _ => ha i).add
    ((HasDerivAt.sum fun i _ => ((hv i).div_const _).const_mul _).const_mul _)).add
    ((HasDerivAt.sum fun i _ => ((hl i).div_const _).const_mul _).const_mul _)

end
end StateSetFormal.VAPO
