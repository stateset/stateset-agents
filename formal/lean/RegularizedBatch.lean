import ActorCritic

namespace StateSetFormal.RegularizedBatch
open Finset Real
open scoped BigOperators
noncomputable section

/-- Multiplying each local mean by its own population share recovers
the global mean. A group with no observations must contribute zero. -/
theorem population_mean {n : ℕ} (counts sums : Fin n → ℝ)
    (empty : ∀ i, counts i = 0 → sums i = 0) :
    (∑ i, (counts i / (∑ j, counts j)) * (sums i / counts i)) =
    (∑ i, sums i) / (∑ i, counts i) := by
  rw [sum_div]
  apply sum_congr rfl
  intro i _
  by_cases h : counts i = 0
  · simp [h, empty i h]
  · by_cases total : (∑ j, counts j) = 0
    · simp [total]
    · field_simp [h, total]
      ring

/-- The policy, sequence KL, and token entropy each have their own finite
population. The derivative of the streamed objective recovers the three
global derivatives, including groups with no response tokens. -/
theorem batch_gradient {n : ℕ} (pol rows tokens : Fin n → ℝ)
    (p q h : Fin n → ℝ → ℝ) (dp dq dh : Fin n → ℝ) (x beta eta : ℝ)
    (ep : ∀ i, pol i = 0 → dp i = 0) (eq : ∀ i, rows i = 0 → dq i = 0)
    (eh : ∀ i, tokens i = 0 → dh i = 0)
    (hp : ∀ i, HasDerivAt (p i) (dp i) x) (hq : ∀ i, HasDerivAt (q i) (dq i) x)
    (hh : ∀ i, HasDerivAt (h i) (dh i) x) :
    HasDerivAt (fun y =>
      (∑ i, (pol i / ∑ j, pol j) * (p i y / pol i)) +
      beta * (∑ i, (rows i / ∑ j, rows j) * (q i y / rows i)) -
      eta * (∑ i, (tokens i / ∑ j, tokens j) * (h i y / tokens i)))
      ((∑ i, dp i) / (∑ i, pol i) + beta * ((∑ i, dq i) / (∑ i, rows i)) -
       eta * ((∑ i, dh i) / (∑ i, tokens i))) x := by
  rw [← population_mean pol dp ep, ← population_mean rows dq eq, ← population_mean tokens dh eh]
  apply HasDerivAt.sub
  · apply HasDerivAt.add
    · exact HasDerivAt.sum fun i _ => ((hp i).div_const _).const_mul _
    · exact (HasDerivAt.sum fun i _ => ((hq i).div_const _).const_mul _).const_mul _
  · exact (HasDerivAt.sum fun i _ => ((hh i).div_const _).const_mul _).const_mul _

/-- Full forward KL plus negative entropy, differentiated in any shared
policy parameter. The reference probabilities and coefficients are fixed. -/
theorem categorical_gradient {n : ℕ} (p : Fin n → ℝ → ℝ) (dp q : Fin n → ℝ)
    (x beta eta : ℝ) (hp : ∀ i, 0 < p i x) (_hq : ∀ i, 0 < q i)
    (hd : ∀ i, HasDerivAt (p i) (dp i) x) (hm : ∀ y, ∑ i, p i y = 1) :
    HasDerivAt (fun y => beta * (∑ i, p i y * (log (p i y) - log (q i))) +
        eta * (∑ i, p i y * log (p i y)))
      (∑ i, dp i * ((beta+eta)*log (p i x) - beta*log (q i))) x := by
  have kl := ScoreFunction.forward_kl_gradient q hp hd hm
  have ent := ActorCritic.neg_entropy_gradient p dp x hp hd hm
  convert (kl.const_mul beta).add (ent.const_mul eta) using 1
  simp only [mul_sum, ← sum_add_distrib]
  apply sum_congr rfl
  intro i _
  ring

def penalty {n : ℕ} (p q : Fin n → ℝ) (beta eta : ℝ) :=
  beta * (∑ i, p i * (log (p i) - log (q i))) + eta * (∑ i, p i * log (p i))

/-- At a fixed context, positive reference probabilities raised to
beta/(beta+eta) and normalized minimize forward KL plus negative entropy.
The excess penalty is exactly the scaled KL to that tilted distribution. -/
theorem tilted_gap {n : ℕ} [NeZero n] (p q : Fin n → ℝ) (beta eta : ℝ)
    (temp : 0 < beta+eta) (pm : ∑ i, p i = 1) :
    let g := ScoreFunction.temperatureProb (fun i => beta*log (q i)) (beta+eta)
    penalty p q beta eta - penalty g q beta eta =
       (beta+eta) * ∑ i, p i * (log (p i) - log (g i)) := by
  let z := fun i => beta*log (q i)
  let g := ScoreFunction.temperatureProb z (beta+eta)
  let b := log (∑ i, exp (z i/(beta+eta)))
  have score (i) : log (g i) = z i/(beta+eta) - b :=
    ScoreFunction.temperature_logprob z temp i
  have gap (t : Fin n → ℝ) (tm : ∑ i, t i = 1) :
      penalty t q beta eta = (beta+eta) * (∑ i, t i * (log (t i)-log (g i))) - (beta+eta)*b := by
    have term (i : Fin n) :
        beta*(t i*(log (t i)-log (q i))) + eta*(t i*log (t i)) =
        (beta+eta)*(t i*(log (t i)-log (g i))) - t i*((beta+eta)*b) := by
      rw [score]
      dsimp only [z]
      field_simp [temp.ne']
      ring
    unfold penalty
    simp only [mul_sum, ← sum_add_distrib, term, sum_sub_distrib, ← sum_mul, tm, one_mul]
  change penalty p q beta eta - penalty g q beta eta = _
  rw [gap p pm, gap g (ScoreFunction.temperature_mass z temp)]
  simp [g]

theorem tilted_optimal {n : ℕ} [NeZero n] (p q : Fin n → ℝ) (beta eta : ℝ)
    (temp : 0 < beta+eta) (hp : ∀ i, 0 < p i) (pm : ∑ i, p i = 1) :
    let g := ScoreFunction.temperatureProb (fun i => beta*log (q i)) (beta+eta)
    penalty g q beta eta ≤ penalty p q beta eta := by
  let z := fun i => beta*log (q i)
  let g := ScoreFunction.temperatureProb z (beta+eta)
  have hg (i) : 0 < g i := div_pos (exp_pos _) (sum_pos (fun _ _ => exp_pos _) univ_nonempty)
  have gap := tilted_gap p q beta eta temp pm
  change penalty p q beta eta - penalty g q beta eta =
    (beta+eta) * ∑ i, p i * (log (p i)-log (g i)) at gap
  rw [← Policy.k3_on_policy p g hp hg pm (ScoreFunction.temperature_mass z temp)] at gap
  exact sub_nonneg.mp (gap.symm.trans_ge (mul_nonneg temp.le
    (sum_nonneg (fun i _ => mul_nonneg (hp i).le (Policy.k3_nonnegative _)))))

/-- The full-KL implementation adds q on actions outside the learner's
support. This is necessary when using the nonnegative per-action formula. -/
theorem sparse_kl {n : ℕ} (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i)
    (support : ∀ i, p i ≠ 0 → 0 < q i) (pm : ∑ i, p i = 1) (qm : ∑ i, q i = 1) :
    (∑ i, if p i = 0 then q i else p i * Policy.k3 (log (q i) - log (p i))) =
      ∑ i, p i * (log (p i) - log (q i)) := by
  have term (i : Fin n) :
      (if p i = 0 then q i else p i * Policy.k3 (log (q i) - log (p i))) =
      p i * (log (p i) - log (q i)) + (q i - p i) := by
    by_cases h : p i = 0
    · simp [h]
    · rw [if_neg h, Policy.k3, exp_sub, exp_log (support i h), exp_log (lt_of_le_of_ne (hp i) (Ne.symm h))]
      field_simp [h]
      ring
  simp_rw [term, sum_add_distrib, sum_sub_distrib, pm, qm, sub_self, add_zero]

def smallGap (x : ℝ) :=
  x^2 * (1/2 + x*(1/6 + x*(1/24 + x*(1/120 + x/720))))

def smallSlope (x : ℝ) := x + x^2/2 + x^3/6 + x^4/24 + x^5/120

/-- The degree-six branch avoids cancellation in exp(d)-1-d. These
absolute truncation error bounds apply before floating-point rounding. -/
theorem small_gap_error {x : ℝ} (hx : |x| ≤ 1) :
    |Policy.k3 x - smallGap x| ≤ |x|^7/4410 := by
  have h := Real.exp_bound hx (n := 7) (by decide)
  norm_num [sum_range_succ, Nat.factorial] at h
  convert h using 1 <;> first | (congr 1; unfold Policy.k3 smallGap; ring) | ring

theorem small_slope_error {x : ℝ} (hx : |x| ≤ 1) :
    |(exp x - 1) - smallSlope x| ≤ |x|^6 * (7/4320) := by
  have h := Real.exp_bound hx (n := 6) (by decide)
  norm_num [sum_range_succ, Nat.factorial] at h
  convert h using 1
  congr 1
  unfold smallSlope
  ring

theorem small_derivative (x : ℝ) : HasDerivAt smallGap (smallSlope x) x := by
  have h := (((((hasDerivAt_pow 2 x).div_const 2).add ((hasDerivAt_pow 3 x).div_const 6)).add
      ((hasDerivAt_pow 4 x).div_const 24)).add ((hasDerivAt_pow 5 x).div_const 120)).add
      ((hasDerivAt_pow 6 x).div_const 720)
  convert h using 1
  · funext y
    unfold smallGap
    ring
  · unfold smallSlope
    norm_num
    ring

end
end StateSetFormal.RegularizedBatch
