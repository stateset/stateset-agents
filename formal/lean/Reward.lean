import Std

namespace StateSetFormal

/-- Model scores in integer hundredths. The Python reward path clamps its
    final score to the interval [0, 1]. -/
def clampScore (score : Int) : Int :=
  max 0 (min 100 score)

theorem clampScore_nonnegative (score : Int) : 0 ≤ clampScore score := by
  unfold clampScore
  omega

theorem clampScore_at_most_100 (score : Int) : clampScore score ≤ 100 := by
  unfold clampScore
  omega

theorem clampScore_identity (score : Int) (lo : 0 ≤ score)
    (hi : score ≤ 100) : clampScore score = score := by
  unfold clampScore
  omega

theorem clampScore_idempotent (score : Int) :
    clampScore (clampScore score) = clampScore score := by
  unfold clampScore
  omega

/-- Failed components contribute zero to the weighted sum. -/
def componentContribution (succeeded : Bool) (weight score : Int) : Int :=
  if succeeded then weight * score else 0

theorem failed_component_zero (weight score : Int) :
    componentContribution false weight score = 0 := by
  rfl

/-- A zero-weight component cannot change the weighted sum. -/
theorem zero_weight_component (succeeded : Bool) (score : Int) :
    componentContribution succeeded 0 score = 0 := by
  cases succeeded <;> simp [componentContribution]

theorem failed_component_does_not_change_sum (weight score rest : Int) :
    componentContribution false weight score + rest = rest := by
  simp [componentContribution]

/-- Nonnegative integer weights and component scores, before converting back
    to a floating-point reward. -/
def weightedSum : List (Nat × Nat) → Nat
  | [] => 0
  | (weight, score) :: rest => weight * score + weightedSum rest

def totalWeight : List (Nat × Nat) → Nat
  | [] => 0
  | (weight, _) :: rest => weight + totalWeight rest

theorem weightedSum_bound (components : List (Nat × Nat))
    (bounded : ∀ pair ∈ components, pair.2 ≤ 100) :
    weightedSum components ≤ 100 * totalWeight components := by
  induction components with
  | nil => simp [weightedSum, totalWeight]
  | cons pair rest ih =>
      have score_bound : pair.2 ≤ 100 := bounded pair (by simp)
      have rest_bound : ∀ p ∈ rest, p.2 ≤ 100 := by
        intro p hp
        exact bounded p (by simp [hp])
      have component_bound := Nat.mul_le_mul_left pair.1 score_bound
      have tail_bound := ih rest_bound
      simp only [weightedSum, totalWeight]
      omega

theorem normalized_weighted_sum_bound (components : List (Nat × Nat))
    (bounded : ∀ pair ∈ components, pair.2 ≤ 100)
    (normalized : totalWeight components = 100) :
    weightedSum components ≤ 10000 := by
  have bound := weightedSum_bound components bounded
  rw [normalized] at bound
  omega

/-- Two group-mean advantages sum to zero, matching TrajectoryGroup's
    centered reward computation before floating-point conversion. -/
theorem two_rewards_center (a b : Int) : (a - b) + (b - a) = 0 := by
  omega

end StateSetFormal
