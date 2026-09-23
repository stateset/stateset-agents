import Std

namespace StateSetFormal

/-- Integer abstraction of clipping a log ratio before exponentiation. -/
def clipLogRatio (bound value : Int) : Int :=
  max (-bound) (min bound value)

theorem clipLogRatio_lower (bound value : Int) :
    -bound ≤ clipLogRatio bound value := by
  unfold clipLogRatio
  omega

theorem clipLogRatio_upper (bound value : Int) (h : 0 ≤ bound) :
    clipLogRatio bound value ≤ bound := by
  unfold clipLogRatio
  omega

theorem clipLogRatio_idempotent (bound value : Int) (h : 0 ≤ bound) :
    clipLogRatio bound (clipLogRatio bound value) = clipLogRatio bound value := by
  unfold clipLogRatio
  omega

/-- Integer abstraction of the asymmetric PPO/GSPO ratio clip. -/
def clipRatio (low high ratio : Int) : Int :=
  max low (min high ratio)

theorem clipRatio_identity (low high ratio : Int)
    (above : low ≤ ratio) (below : ratio ≤ high) :
    clipRatio low high ratio = ratio := by
  unfold clipRatio
  omega

def clippedSurrogate (low high ratio advantage : Int) : Int :=
  -(min (ratio * advantage)
      ((clipRatio low high ratio) * advantage))

theorem clippedSurrogate_at_least_unclipped
    (low high ratio advantage : Int) :
    -(ratio * advantage) ≤ clippedSurrogate low high ratio advantage := by
  unfold clippedSurrogate
  omega

theorem clippedSurrogate_in_range (low high ratio advantage : Int)
    (above : low ≤ ratio) (below : ratio ≤ high) :
    clippedSurrogate low high ratio advantage = -(ratio * advantage) := by
  simp only [clippedSurrogate, clipRatio_identity low high ratio above below]
  omega

theorem clippedSurrogate_zero_advantage (low high ratio : Int) :
    clippedSurrogate low high ratio 0 = 0 := by
  simp [clippedSurrogate]
  omega

/-- A zero mask contributes no token score to the objective. -/
def maskedTerm (enabled : Bool) (score : Int) : Int :=
  if enabled then score else 0

theorem maskedTerm_false (score : Int) : maskedTerm false score = 0 := by
  rfl

theorem maskedTerm_true (score : Int) : maskedTerm true score = score := by
  rfl

/-- Cross-multiplied, two-response group-mean advantages. -/
def pairAdvantages (first second : Int) : Int × Int :=
  (first - second, second - first)

theorem pairAdvantages_centered (first second : Int) :
    (pairAdvantages first second).1 + (pairAdvantages first second).2 = 0 := by
  simp only [pairAdvantages]
  omega

/-- Leave-one-out and group-mean centering agree for a group of size two,
    up to the expected factor of two. -/
theorem pairAdvantages_from_mean (first second : Int) :
    (pairAdvantages first second).1 = 2 * first - (first + second) := by
  simp [pairAdvantages]
  omega

/-- Sum of cross-multiplied group-mean advantages. This formulation avoids
    division and remains exact for every finite group, including size one. -/
def centeredSum (rewards : List Int) : Int :=
  (rewards.map fun reward => (rewards.length : Int) * reward - rewards.sum).sum

private theorem centeredSum_aux (rewards : List Int) (n total : Int) :
    (rewards.map fun reward => n * reward - total).sum =
      n * rewards.sum - (rewards.length : Int) * total := by
  induction rewards with
  | nil => simp
  | cons reward rest ih =>
      simp only [List.map_cons, List.sum_cons, List.length_cons, List.map_nil,
        Int.ofNat_add, Int.ofNat_one]
      rw [ih]
      simp only [Int.mul_add, Int.add_mul]
      omega

theorem centeredSum_zero (rewards : List Int) : centeredSum rewards = 0 := by
  unfold centeredSum
  rw [centeredSum_aux]
  omega

end StateSetFormal
