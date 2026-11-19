import Returns
import Mathlib.Probability.Independence.Basic

namespace StateSetFormal.TargetPolicy
open StateSetFormal.Returns Finset
open scoped BigOperators MeasureTheory
noncomputable section

/-- Newest completed learner step first. Only scheduled steps publish the
    policy and critic coordinates to the target used by the next backup. -/
def snapshot (τ initial : ℝ) : List (Bool × ℝ) → ℝ
  | [] => initial
  | (false, _) :: past => snapshot τ initial past
  | (true, live) :: past => τ * live + (1-τ) * snapshot τ initial past

theorem unscheduled (τ initial live : ℝ) (past : List (Bool × ℝ)) :
    snapshot τ initial ((false, live) :: past) = snapshot τ initial past := rfl

/-- The effect of a different initial checkpoint decays once per publication,
    independently of how many learner updates occurred between publications. -/
theorem initial_difference (τ x y : ℝ) (past : List (Bool × ℝ)) :
    snapshot τ x past - snapshot τ y past =
      (1-τ) ^ (past.filter (fun p => p.1)).length * (x-y) := by
  induction past with
  | nil => simp [snapshot]
  | cons step past ih =>
    rcases step with ⟨flag, live⟩
    cases flag
    · simpa [snapshot, List.filter] using ih
    · simp only [snapshot, List.filter, if_true, List.length_cons, pow_succ]
      linear_combination (1-τ) * ih

theorem history_bounds {τ lo hi x : ℝ} (ht : 0 ≤ τ) (ht' : τ ≤ 1)
    (hx : lo ≤ x ∧ x ≤ hi) (past : List (Bool × ℝ))
    (h : ∀ v, (true, v) ∈ past → lo ≤ v ∧ v ≤ hi) :
    lo ≤ snapshot τ x past ∧ snapshot τ x past ≤ hi := by
  induction past with
  | nil => exact hx
  | cons step past ih =>
    have hp := ih (fun v hv => h v (List.mem_cons_of_mem _ hv))
    rcases step with ⟨flag,v⟩
    cases flag
    · exact hp
    · have hv := h v (List.mem_cons_self _ _)
      simp only [snapshot]
      constructor
      · nlinarith [mul_nonneg ht (sub_nonneg.mpr hv.1),
          mul_nonneg (sub_nonneg.mpr ht') (sub_nonneg.mpr hp.1)]
      · nlinarith [mul_nonneg ht (sub_nonneg.mpr hv.2),
          mul_nonneg (sub_nonneg.mpr ht') (sub_nonneg.mpr hp.2)]

def twin (w a b : ℝ) := w * min a b + (1-w) * max a b

private theorem twin_le {w a b c d e : ℝ} (hl : 0 ≤ w) (hu : w ≤ 1)
    (ha : a ≤ c + e) (hb : b ≤ d + e) : twin w a b ≤ twin w c d + e := by
  have hm : min a b ≤ min c d + e := by
    simpa only [min_add_add_right] using min_le_min ha hb
  have hM : max a b ≤ max c d + e := by
    simpa only [max_add_add_right] using max_le_max ha hb
  unfold twin
  nlinarith [mul_le_mul_of_nonneg_left hm hl,
    mul_le_mul_of_nonneg_left hM (sub_nonneg.mpr hu)]

/-- Conditional on paired proposal scores within ε in each head, changing
    the winning proposal or the ordering of the twins costs at most γ ε. -/
theorem sampled_backup_bound {n : ℕ} {w γ ε r : ℝ} (hl : 0 ≤ w) (hu : w ≤ 1)
    (hg : 0 ≤ γ) (done : Bool) (a b c d : Fin n → ℝ) (i j : Fin n)
    (hi : ∀ k, twin w (a k) (b k) ≤ twin w (a i) (b i))
    (hj : ∀ k, twin w (c k) (d k) ≤ twin w (c j) (d j))
    (ha : ∀ k, |a k - c k| ≤ ε) (hb : ∀ k, |b k - d k| ≤ ε) :
    |backup γ r (twin w (a i) (b i)) done - backup γ r (twin w (c j) (d j)) done| ≤ γ*ε := by
  apply le_trans (backup_lipschitz _ _ _ _ hg done)
  apply mul_le_mul_of_nonneg_left _ hg
  apply greedy_lipschitz _ _ i j ε hi hj
  intro k
  have a' := abs_le.mp (ha k)
  have b' := abs_le.mp (hb k)
  have h₁ := twin_le hl hu (show c k ≤ a k + ε by linarith) (show d k ≤ b k + ε by linarith)
  have h₂ := twin_le hl hu (show a k ≤ c k + ε by linarith) (show b k ≤ d k + ε by linarith)
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem zero_discount (r v : ℝ) (done : Bool) : backup 0 r v done = r := by
  cases done <;> simp [backup]


/-- Condition on a fixed successor and proposal policy. Each draw uses the
    same finite action distribution independently, with replacement. -/
def expectedShortfall {n : ℕ} (p q : Fin n → ℝ) (top : ℝ) : ℕ → ℝ → ℝ
  | 0, best => top - best
  | k+1, best => ∑ i, p i * expectedShortfall p q top k (max best (q i))

def missMass {n : ℕ} (p q : Fin n → ℝ) (top : ℝ) :=
  ∑ i, if q i < top then p i else 0

theorem miss_probability {n : ℕ} {p q : Fin n → ℝ} {top : ℝ}
    (hp : ∀ i, 0 ≤ p i) (total : ∑ i, p i = 1) :
    0 ≤ missMass p q top ∧ missMass p q top ≤ 1 := by
  unfold missMass
  constructor
  · exact sum_nonneg fun i _ => by split <;> simp [hp i]
  · rw [← total]
    exact sum_le_sum fun i _ => by split <;> simp [hp i]

/-- If every proposal score is at most top, after k independent draws the
    expected gap is bounded by the initial gap times the k-th power of the
    mass below top. The floor can be any lower bound on proposal scores. -/
theorem independent_proposal_bound {n : ℕ} (p q : Fin n → ℝ) (top : ℝ)
    (hp : ∀ i, 0 ≤ p i) (total : ∑ i, p i = 1) (hq : ∀ i, q i ≤ top)
    (k : ℕ) (best : ℝ) (hbest : best ≤ top) :
    0 ≤ expectedShortfall p q top k best ∧
      expectedShortfall p q top k best ≤ (top-best) * missMass p q top ^ k := by
  have hm := (miss_probability (q := q) (top := top) hp total).1
  induction k generalizing best with
  | zero => simp [expectedShortfall, hbest]
  | succ k ih =>
    have hnext (i : Fin n) := ih (max best (q i)) (max_le hbest (hq i))
    constructor
    · exact sum_nonneg fun i _ => mul_nonneg (hp i) (hnext i).1
    · calc
        expectedShortfall p q top (k+1) best ≤
            ∑ i, (if q i < top then p i else 0) * ((top-best)*missMass p q top ^ k) := by
          apply sum_le_sum
          intro i _
          by_cases ht : q i < top
          · rw [if_pos ht]
            apply mul_le_mul_of_nonneg_left _ (hp i)
            apply le_trans (hnext i).2
            exact mul_le_mul_of_nonneg_right (sub_le_sub_left (le_max_left _ _) _) (pow_nonneg hm _)
          · rw [if_neg ht, zero_mul]
            have eq := le_antisymm (hq i) (le_of_not_gt ht)
            have hzero := (hnext i).2
            rw [eq, max_eq_right hbest, sub_self, zero_mul] at hzero
            rw [eq, max_eq_right hbest]
            exact mul_nonpos_of_nonneg_of_nonpos (hp i) hzero
        _ = (top-best) * missMass p q top ^ (k+1) := by
          rw [← Finset.sum_mul, ← missMass, pow_succ]
          ring

/-- On an arbitrary probability space (including continuous proposals), the
    sampled maximum misses a score threshold exactly when every draw misses.
    Atoms and ties are allowed. Independence is conditional on the fixed
    policy, successor and critics represented by this law. -/
theorem sampled_max_miss {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω}
    {k : ℕ} (score : Fin k → Ω → ℝ) (threshold : ℝ)
    (indep : ProbabilityTheory.iIndepFun (fun _ => inferInstance) score μ)
    (winner : Ω → Fin k) (wins : ∀ ω i, score i ω ≤ score (winner ω) ω)
    (miss : ENNReal) (sameLaw : ∀ i, μ {ω | score i ω < threshold} = miss) :
    μ {ω | score (winner ω) ω < threshold} = miss ^ k := by
  have events : {ω | score (winner ω) ω < threshold} =
      ⋂ i, {ω | score i ω < threshold} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iInter]
    exact ⟨fun h i => lt_of_le_of_lt (wins ω i) h, fun h => h (winner ω)⟩
  rw [events]
  have measurableMiss (i : Fin k) :
      MeasurableSet[(inferInstance : MeasurableSpace ℝ).comap (score i)] {ω | score i ω < threshold} :=
    ⟨Set.Iio threshold, measurableSet_Iio, rfl⟩
  rw [indep.meas_iInter measurableMiss]
  simp [sameLaw]

end
end StateSetFormal.TargetPolicy
