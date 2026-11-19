import Returns
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace StateSetFormal.Schedules
noncomputable section

/-- Both the linear and cosine KL schedules interpolate between two
    coefficients. The default half-cycle visits each endpoint once. -/
def cosineCoef (low high phase : ℝ) :=
  low + (high-low) * ((1 + Real.cos phase)/2)

theorem cosine_bounds {lo hi : ℝ} (h : lo ≤ hi) (phase : ℝ) :
    lo ≤ cosineCoef lo hi phase ∧ cosineCoef lo hi phase ≤ hi := by
  let w := (1 + Real.cos phase)/2
  have h0 : 0 ≤ w := by dsimp [w]; linarith [Real.neg_one_le_cos phase]
  have h1 : w ≤ 1 := by dsimp [w]; linarith [Real.cos_le_one phase]
  have := Returns.convex_bounds (w := 1-w) (by linarith) (by linarith) h
  dsimp [cosineCoef, w] at *
  constructor <;> linarith

theorem cosine_endpoints (lo hi : ℝ) :
    cosineCoef lo hi 0 = hi ∧ cosineCoef lo hi Real.pi = lo := by
  simp [cosineCoef]

/-- Projected proportional updates preserve the controller interval and
    move in the error's direction as long as the current coefficient is valid. -/
def adaptive (lo hi coef err progress : ℝ) :=
  max lo (min hi (coef * (1 + err*progress)))

theorem adaptive_direction {lo hi c progress : ℝ} (e : ℝ)
    (hc : 0 ≤ c) (hl : lo ≤ c) (hh : c ≤ hi) (hp : 0 ≤ progress) :
    lo ≤ adaptive lo hi c e progress ∧ adaptive lo hi c e progress ≤ hi ∧
    (0 ≤ e → c ≤ adaptive lo hi c e progress) ∧
    (e ≤ 0 → adaptive lo hi c e progress ≤ c) := by
  unfold adaptive
  refine ⟨le_max_left .., max_le (hl.trans hh) (min_le_left ..), ?_, ?_⟩
  · intro he
    exact le_max_of_le_right (le_min hh (by nlinarith [mul_nonneg hc (mul_nonneg he hp)]))
  · intro he
    have h := mul_nonpos_of_nonneg_of_nonpos hc (mul_nonpos_of_nonpos_of_nonneg he hp)
    exact max_le hl (min_le_of_right_le (by nlinarith))

/-- Curiosity exploration uses the visit count of the conditioning state. -/
def novelty (visits : ℝ) := (1 / (1 + visits)) / 2 + 1 / 10

theorem novelty_bounds {visits : ℝ} (h : 0 ≤ visits) :
    1/10 ≤ novelty visits ∧ novelty visits ≤ 3/5 := by
  have hpos : 0 < 1+visits := by linarith
  have hle : 1/(1+visits) ≤ (1:ℝ) := (div_le_iff₀ hpos).mpr (by linarith)
  unfold novelty
  constructor <;> linarith [one_div_pos.mpr hpos]

theorem novelty_decreases {visits : ℝ} (h : 0 ≤ visits) :
    novelty (visits+1) < novelty visits := by
  have hpos : 0 < 1+visits := by linarith
  have hlt := one_div_lt_one_div_of_lt hpos (show 1+visits < 1+(visits+1) by linarith)
  unfold novelty
  linarith

/-- At fixed total pulls the UCB confidence bonus decreases with action count. -/
theorem ucb_bonus_antitone {total n m : ℝ} (ht : 1 ≤ total) (hn : 0 < n) (hnm : n ≤ m) :
    Real.sqrt (2*Real.log total/m) ≤ Real.sqrt (2*Real.log total/n) := by
  apply Real.sqrt_le_sqrt
  exact div_le_div_of_nonneg_left (mul_nonneg (by norm_num) (Real.log_nonneg ht)) hn hnm


end
end StateSetFormal.Schedules
