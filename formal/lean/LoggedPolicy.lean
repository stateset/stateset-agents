import RewardWeights

namespace StateSetFormal.LoggedPolicy
open Finset Real
open scoped BigOperators
noncomputable section

/-- Aggregate the per-action weight per retained token at one (prefix, next token). -/
def tokenMass {b t : ℕ} {K : Type*} [DecidableEq K]
    (mask : Fin b → Finset (Fin t)) (w : Fin b → ℝ) (key : Fin b → Fin t → K) (k : K) :=
  ∑ i, ∑ j ∈ mask i, if key i j = k then w i / (mask i).card else 0

/-- The empirical logged-action objective can be grouped by causal prefix
and token even when the selected action lengths and weights differ. -/
theorem regroup {b t : ℕ} {K : Type*} [Fintype K] [DecidableEq K]
    (mask : Fin b → Finset (Fin t)) (w : Fin b → ℝ) (key : Fin b → Fin t → K) (f : K → ℝ) :
    (∑ k, tokenMass mask w key k * f k) =
    ∑ i, (w i / (mask i).card)*(∑ j ∈ mask i, f (key i j)) := by
  simp only [tokenMass, sum_mul, mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i _
  rw [sum_comm]
  apply sum_congr rfl
  intro j _
  simp [ite_mul]

theorem extraction_gradient {b t : ℕ} (mask : Fin b → Finset (Fin t)) (w : Fin b → ℝ)
    (logp : Fin b → Fin t → ℝ → ℝ) (slope : Fin b → Fin t → ℝ) (x : ℝ)
    (hd : ∀ i, ∀ j ∈ mask i, HasDerivAt (logp i j) (slope i j) x) :
    HasDerivAt (fun y => (∑ i, -(w i/(mask i).card)*(∑ j ∈ mask i, logp i j y))/b)
      ((∑ i, ∑ j ∈ mask i, -(w i/(mask i).card)*slope i j)/b) x :=
  RewardWeights.row_loss_gradient mask _ logp slope x _ hd

/-- At one context the categorical gradient is predicted mass minus
empirical mass for each next-token logit. -/
theorem score_gradient {n : ℕ} (w p d : Fin n → ℝ) :
    (∑ i, -w i*(d i - ∑ j, p j*d j)) =
    ∑ i, ((∑ j, w j)*p i-w i)*d i := by
  simp_rw [sub_mul, mul_assoc, mul_sub, neg_mul, neg_sub_neg]
  rw [sum_sub_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum]

/-- The proposed conditional distribution has total mass one at each observed
prefix and shares precisely the empirical next-token support. -/
theorem empirical_policy {n : ℕ} (w : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i)
    (hz : 0 < ∑ i, w i) :
    (∑ i, w i/(∑ j, w j)) = 1 ∧
      ∀ i, 0 ≤ w i/(∑ j, w j) ∧ (0 < w i/(∑ j, w j) ↔ 0 < w i) := by
  refine ⟨by rw [← sum_div, div_self hz.ne'], fun i => ⟨div_nonneg (hw i) hz.le, ?_⟩⟩
  exact div_pos_iff_of_pos_right hz

/-- The unique categorical fit at an observed prefix normalizes its empirical
token masses, allowing zero mass on tokens absent from every logged action.
The candidate must support each observed token to have finite loss. -/
theorem unique_prefix_optimum {n : ℕ} (w q : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hz : 0 < ∑ i, w i)
    (hq : ∀ i, 0 ≤ q i) (support : ∀ i, 0 < w i → 0 < q i) (qm : ∑ i, q i = 1)
    (different : ∃ i, q i ≠ w i / ∑ j, w j) :
    ∑ i, w i*log (q i) < ∑ i, w i*log (w i / ∑ j, w j) := by
  let z := ∑ i, w i
  let p := fun i => w i/z
  have coord (i) (pos : 0 < w i) :
      w i*(log (q i)-log (p i)) ≤ z*q i-w i ∧
      (q i ≠ p i → w i*(log (q i)-log (p i)) < z*q i-w i) := by
    have pp := div_pos pos hz
    have qp := support i pos
    have ratio := div_pos qp pp
    have identity : w i*(q i / p i-1) = z*q i-w i := by
      dsimp [p]; field_simp; ring
    rw [← log_div qp.ne' pp.ne', ← identity]
    refine ⟨mul_le_mul_of_nonneg_left (log_le_sub_one_of_pos ratio) pos.le, ?_⟩
    intro h
    apply mul_lt_mul_of_pos_left _ pos
    exact log_lt_sub_one_of_pos ratio (fun he => h (by simpa using (div_eq_iff pp.ne').mp he))
  have sumlt : (∑ i, w i*(log (q i)-log (p i))) < ∑ i, (z*q i-w i) := by
    apply sum_lt_sum
    · intro i _
      by_cases pos : 0 < w i
      · exact (coord i pos).1
      · simp [le_antisymm (le_of_not_gt pos) (hw i), mul_nonneg hz.le (hq i)]
    · obtain ⟨i, diff⟩ := different
      refine ⟨i, mem_univ _, ?_⟩
      by_cases pos : 0 < w i
      · exact (coord i pos).2 diff
      · have zero := le_antisymm (le_of_not_gt pos) (hw i)
        rw [zero, zero_div] at diff
        simp [zero, mul_pos hz (lt_of_le_of_ne (hq i) diff.symm)]
  simp_rw [mul_sub, sum_sub_distrib, ← mul_sum, qm, mul_one] at sumlt
  change _ < ∑ i, w i*log (p i)
  dsimp only [z] at sumlt
  linarith

/-- A disagreement at any observed prefix strictly decreases dataset log
likelihood. Unvisited prefixes contribute zero under every policy. -/
theorem dataset_optimum {c a : ℕ} (w q : Fin c → Fin a → ℝ)
    (hw : ∀ k i, 0 ≤ w k i) (hq : ∀ k i, 0 ≤ q k i)
    (support : ∀ k i, 0 < w k i → 0 < q k i) (qm : ∀ k, ∑ i, q k i = 1)
    (witness : ∃ k, 0 < ∑ j, w k j ∧ ∃ i, q k i ≠ w k i / ∑ j, w k j) :
    (∑ k, ∑ i, w k i*log (q k i)) <
      ∑ k, ∑ i, w k i*log (w k i/∑ j, w k j) := by
  apply sum_lt_sum
  · intro k _
    by_cases visited : 0 < ∑ j, w k j
    · by_cases diff : ∃ i, q k i ≠ w k i/∑ j, w k j
      · exact (unique_prefix_optimum _ _ (hw k) visited (hq k) (support k) (qm k) diff).le
      · push_neg at diff
        exact le_of_eq (sum_congr rfl (fun i _ => by rw [diff i]))
    · have zero : ∑ j, w k j = 0 :=
        le_antisymm (le_of_not_gt visited) (sum_nonneg fun j _ => hw k j)
      have each := (sum_eq_zero_iff_of_nonneg (fun j (_ : j ∈ univ) => hw k j)).mp zero
      simp [fun i => each i (mem_univ i)]
  · obtain ⟨k, visited, diff⟩ := witness
    exact ⟨k, mem_univ _, unique_prefix_optimum _ _ (hw k) visited (hq k) (support k) (qm k) diff⟩

end
end StateSetFormal.LoggedPolicy
