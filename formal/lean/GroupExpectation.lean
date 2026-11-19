import PolicyMath

namespace StateSetFormal.GroupExpectation
open Finset Real
open scoped BigOperators
noncomputable section

def coefficient {n : ℕ} (q : Fin n → ℝ) (p : ℝ) :=
  p*(∑ i, q i)/(∑ i, (q i)^2)

private theorem log_partition_shift {n : ℕ} [NeZero n] (q : Fin n → ℝ) (a : ℝ) :
    log (∑ i, exp (q i-a)) = log (∑ i, exp (q i))-a := by
  simp_rw [exp_sub, ← sum_div]
  rw [log_div (by positivity) (exp_ne_zero _), log_exp]

/-- Subtract a common anchor before evaluating both log partitions and the
    learner gap. This holds for any positive empirical group size. -/
theorem anchored_gap {n : ℕ} [NeZero n] (q : Fin n → ℝ) (p a : ℝ) :
    p-(log (∑ i, exp (2*q i))-log (∑ i, exp (q i))) =
    (p-a)-(log (∑ i, exp (2*(q i-a)))-log (∑ i, exp (q i-a))) := by
  simp_rw [mul_sub, log_partition_shift]
  ring

/-- Exponentiating the anchored log-score formula gives the empirical weight. -/
theorem anchored_coefficient {n : ℕ} [NeZero n] (q : Fin n → ℝ) (p a : ℝ) :
    exp ((p-a)-(log (∑ i, exp (2*(q i-a)))-log (∑ i, exp (q i-a)))) =
      coefficient (fun i => exp (q i)) (exp p) := by
  rw [← anchored_gap, exp_sub, StateSetFormal.Policy.gepo_denominator]
  unfold coefficient
  rw [div_div_eq_mul_div]

/-- On identical policy weights the q-weighted coefficient average is one. -/
theorem weighted_mass {n : ℕ} [NeZero n] (q : Fin n → ℝ) (hq : ∀ i, 0 < q i) :
    ∑ i, q i * coefficient q (q i) = ∑ i, q i := by
  have h : 0 < ∑ i, (q i)^2 := sum_pos (fun i _ => sq_pos_of_pos (hq i)) univ_nonempty
  unfold coefficient
  simp_rw [← mul_div_assoc, ← mul_assoc, ← sq, ← sum_div, ← sum_mul]
  field_simp

/-- Uniformly averaging the same coefficients leaves an exact deficit equal to
    the centered squared mass divided by the uncentered squared mass. -/
theorem mean_deficit {n : ℕ} [NeZero n] (q : Fin n → ℝ) (hq : ∀ i, 0 < q i) :
    1-(∑ i, coefficient q (q i))/n =
      (∑ i, (q i-(∑ j, q j)/n)^2)/(∑ i, (q i)^2) := by
  have hn : (n:ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  have h : 0 < ∑ i, (q i)^2 := sum_pos (fun i _ => sq_pos_of_pos (hq i)) univ_nonempty
  unfold coefficient
  simp_rw [sub_sq, sum_add_distrib, sum_sub_distrib, ← sum_div, ← sum_mul, ← mul_sum]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- The uniform on-policy mean never exceeds one before clipping. -/
theorem uniform_mean_le_one {n : ℕ} [NeZero n] (q : Fin n → ℝ) (hq : ∀ i, 0 < q i) :
    (∑ i, coefficient q (q i))/n ≤ 1 := by
  apply sub_nonneg.mp
  rw [mean_deficit q hq]
  positivity

/-- At a fixed frozen denominator each retained action contributes its log-score
    slope to its response weight. Token count does not divide this derivative. -/
theorem row_derivative {n t : ℕ} (q : Fin n → ℝ) (keep : Finset (Fin t))
    (s : Fin t → ℝ → ℝ) (ds : Fin t → ℝ) (x : ℝ)
    (hd : ∀ j ∈ keep, HasDerivAt (s j) (ds j) x) :
    HasDerivAt (fun y => coefficient q (exp (∑ j ∈ keep, s j y)))
      (coefficient q (exp (∑ j ∈ keep, s j x)) * ∑ j ∈ keep, ds j) x := by
  convert ((((HasDerivAt.sum hd).exp).mul_const (∑ i, q i)).div_const (∑ i, (q i)^2)) using 1
  unfold coefficient
  ring

end
end StateSetFormal.GroupExpectation
