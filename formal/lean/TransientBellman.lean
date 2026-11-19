import LoggedBellman

namespace StateSetFormal.TransientBellman
open Finset Function LoggedBellman
open scoped BigOperators NNReal
noncomputable section

/-- Transport a continuation along live empirical rows. Starting from ones
gives discounted survival mass after each further decision. -/
def carry {n k : ℕ} (g : ℝ) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (u : Fin n → ℝ) (s : Fin n) : ℝ :=
  ∑ i, if (rows s i).terminal then 0 else w s i*g*u (rows s i).next

theorem envelope {n k : ℕ} {g : ℝ} (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (u v a : Fin n → ℝ) (h : ∀ s, |u s-v s| ≤ a s) (s : Fin n) :
    |operator g rows w u s-operator g rows w v s| ≤ carry g rows w a s := by
  unfold operator carry
  rw [← sum_sub_distrib]
  apply (abs_sum_le_sum_abs _ _).trans
  apply sum_le_sum
  intro i _
  cases ht : (rows s i).terminal
  · simp only [ht, Returns.backup, Bool.false_eq_true, if_false, ← mul_sub,
      add_sub_add_left_eq_sub, abs_mul, abs_of_nonneg (hw s i), abs_of_nonneg hg]
    rw [mul_assoc]
    exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (h _) hg) (hw s i)
  · simp [ht, Returns.backup]

theorem iterate_envelope {n k : ℕ} {g : ℝ} (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (u v : Fin n → ℝ) (h : ℕ) (s : Fin n) :
    let T := operator g rows w
    let P := carry g rows w
    |T^[h] u s-T^[h] v s| ≤ P^[h] (fun _ => 1) s * dist u v := by
  dsimp only
  induction h generalizing s with
  | zero => simpa [Real.dist_eq] using (dist_le_pi_dist u v s)
  | succ h ih =>
    simp only [iterate_succ_apply']
    apply (envelope hg rows w hw _ _ _ ih s).trans_eq
    unfold carry
    rw [sum_mul]
    apply sum_congr rfl
    intro i _
    cases ht : (rows s i).terminal <;> simp [ht, mul_assoc]

/-- If surviving h transitions has uniformly bounded discounted mass below
one, the one-step operator has a unique value even at discount one. Its
exact h-step residual supplies a value-error certificate. -/
theorem block_certificate {n k : ℕ} {g : ℝ} (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i) (h : ℕ)
    (β : ℝ≥0) (hb : β < 1) (hs : ∀ s, (carry g rows w)^[h] (fun _ => 1) s ≤ β) :
    let T := operator g rows w
    ∃ V, T V = V ∧ (∀ U, T U = U → U = V) ∧
      ∀ u, dist u V ≤ dist u (T^[h] u)/(1-β) := by
  let T := operator g rows w
  have c : ContractingWith β T^[h] := by
    refine ⟨hb, LipschitzWith.of_dist_le_mul fun u v => ?_⟩
    apply (dist_pi_le_iff (mul_nonneg β.coe_nonneg dist_nonneg)).mpr
    intro s
    rw [Real.dist_eq]
    exact (iterate_envelope hg rows w hw u v h s).trans (mul_le_mul_of_nonneg_right (hs s) dist_nonneg)
  exact ⟨c.fixedPoint T^[h], c.isFixedPt_fixedPoint_iterate,
    fun U hU => c.fixedPoint_unique ((show IsFixedPt T U from hU).iterate h),
    fun u => c.dist_fixedPoint_le u⟩

end
end StateSetFormal.TransientBellman
