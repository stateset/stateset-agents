import PolicyMath

/-! Finite-action policy gradients at a fixed state. A coordinate of the policy
parameter varies. Rewards, the reference distribution and the sampling policy
are held fixed. No independence or support assumptions are implicit. -/
namespace StateSetFormal.ScoreFunction
open Finset Real Policy
open scoped BigOperators
noncomputable section

theorem mass_derivative {n : ℕ} {p : Fin n → ℝ → ℝ} {dp : Fin n → ℝ} {x : ℝ}
    (hd : ∀ i, HasDerivAt (p i) (dp i) x) (hm : ∀ y, ∑ i, p i y = 1) :
    ∑ i, dp i = 0 := by
  have h : HasDerivAt (fun y => ∑ i, p i y) (∑ i, dp i) x :=
    HasDerivAt.sum fun i _ => hd i
  simpa only [hm, deriv_const] using h.deriv.symm

/-- Expected score times an action-independent advantage equals the derivative
of expected reward. The baseline may be chosen separately at each state. -/
theorem reward_gradient {n : ℕ} {p : Fin n → ℝ → ℝ} {dp : Fin n → ℝ} {x : ℝ}
    (r : Fin n → ℝ) (b : ℝ) (hp : ∀ i, 0 < p i x)
    (hd : ∀ i, HasDerivAt (p i) (dp i) x) (hm : ∀ y, ∑ i, p i y = 1) :
    HasDerivAt (fun y => ∑ i, p i y * r i)
      (∑ i, p i x * deriv (fun y => log (p i y)) x * (r i - b)) x := by
  have term (i : Fin n) : p i x * deriv (fun y => log (p i y)) x * (r i - b) =
      dp i * r i - dp i * b := by
    rw [((hd i).log (hp i).ne').deriv]
    field_simp [(hp i).ne']
    ring
  simp_rw [term]
  rw [sum_sub_distrib, ← sum_mul, mass_derivative hd hm, zero_mul, sub_zero]
  exact HasDerivAt.sum fun i _ => (hd i).mul_const (r i)

/-- Under a product law an independent response's reward has zero
    cross moment with the current response's zero-mean score. -/
theorem independent_cross_score {n m : ℕ} (p s : Fin n → ℝ) (q r : Fin m → ℝ)
    (h : ∑ i, p i*s i = 0) :
    ∑ i, ∑ j, p i*q j*s i*r j = 0 := by
  have term (i : Fin n) (j : Fin m) : p i*q j*s i*r j = (p i*s i)*(q j*r j) := by ring
  simp_rw [term, ← mul_sum]
  rw [← sum_mul, h, zero_mul]

/-- On any finite joint law having these same-response and cross-response
    moments, the self-including group mean attenuates the expected score
    by (n-1)/n. Leaving out the scored response recovers the full gradient. -/
theorem group_score_expectation {n m : ℕ} (hn : 1 < n) (p : Fin m → ℝ)
    (s r : Fin n → Fin m → ℝ) (g : ℝ)
    (moments : ∀ i j, ∑ k, p k*s i k*r j k = if i=j then g else 0)
    (i : Fin n) :
    let avg := fun k => (∑ j, r j k)/(n:ℝ)
    (∑ k, p k*s i k*(r i k-avg k)) = ((n:ℝ)-1)/n*g ∧
    (∑ k, p k*s i k*(r i k-((∑ j, r j k)-r i k)/((n:ℝ)-1))) = g := by
  have h : (1:ℝ) < n := by exact_mod_cast hn
  have own := moments i i
  simp only [if_pos rfl] at own
  have cross : ∑ k, p k*s i k*(∑ j, r j k) = g := by
    simp_rw [mul_sum]
    rw [sum_comm]
    simp_rw [moments i]
    simp
  dsimp only
  constructor
  · simp_rw [mul_sub, ← mul_div_assoc]
    rw [sum_sub_distrib, ← sum_div, own, cross]
    field_simp
    ring
  · simp_rw [mul_sub, ← mul_div_assoc, mul_sub]
    rw [sum_sub_distrib, ← sum_div, sum_sub_distrib, own, cross]
    simp

/-- An *unclipped* action likelihood ratio changes the sampling measure.
It requires nonzero behavior mass everywhere the sum is taken. -/
theorem change_measure {n : ℕ} (p mu f : Fin n → ℝ) (hm : ∀ i, 0 < mu i) :
    ∑ i, mu i * (p i / mu i) * f i = ∑ i, p i * f i := by
  apply sum_congr rfl
  intro i _
  field_simp [(hm i).ne']

/-- At a fixed state, PPO's clipped reward surrogate is pessimistic
relative to the exact importance-weighted advantage under common support.
This compares action distributions at that state. -/
theorem clipped_expectation_bound {n : ℕ} (p mu a : Fin n → ℝ) (lo hi : ℝ)
    (hm : ∀ i, 0 < mu i) :
    ∑ i, mu i * (-surrogate lo hi (p i/mu i) (a i)) ≤ ∑ i, p i * a i := by
  rw [← change_measure p mu a hm]
  apply sum_le_sum
  intro i _
  rw [mul_assoc]
  apply mul_le_mul_of_nonneg_left _ (hm i).le
  simp only [surrogate, neg_neg]
  exact min_le_left ..

/-- A truncated sampler recovers the target integral over visited actions.
The absent target contributions remain even if every available ratio is exact. -/
theorem truncated_change_measure {n : ℕ} (p mu f : Fin n → ℝ) :
    ∑ i, p i * f i =
    (∑ i, mu i * (p i / mu i) * f i) +
    ∑ i, (if mu i = 0 then p i * f i else 0) := by
  rw [← sum_add_distrib]
  apply sum_congr rfl
  intro i _
  by_cases h : mu i = 0
  · simp [h]
  · simp only [if_neg h, add_zero]
    field_simp

theorem log_softmax_shift {n : ℕ} [NeZero n] (logit : Fin n → ℝ) (i : Fin n) (c : ℝ) :
    (logit i + c) - log (∑ j, exp (logit j + c)) =
      logit i - log (∑ j, exp (logit j)) := by
  simp_rw [exp_add]
  rw [← sum_mul, log_mul (by positivity) (exp_pos c).ne', log_exp]
  ring

/-- An independent baseline (such as the other samples' mean in RLOO)
can be averaged over its own distribution without changing the score
gradient. The product weights express the independence assumption. -/
theorem independent_baseline {n m : ℕ}
    (p score r : Fin n → ℝ) (weight baseline : Fin m → ℝ)
    (prob : ∑ j, weight j = 1) (zero : ∑ i, p i * score i = 0) :
    ∑ j, weight j * (∑ i, p i * score i * (r i - baseline j)) =
    ∑ i, p i * score i * r i := by
  simp_rw [mul_sub, sum_sub_distrib, ← sum_mul, zero, zero_mul, sub_zero]
  rw [← sum_mul, prob, one_mul]

/-- The standalone RLAIF update averages reward-weighted log likelihood
over the actually scored action tokens, holding their empirical weights fixed. -/
theorem weighted_loglikelihood_gradient {n : ℕ} (weight : Fin n → ℝ)
    {logp : Fin n → ℝ → ℝ} {slope : Fin n → ℝ} {x : ℝ}
    (hd : ∀ i, HasDerivAt (logp i) (slope i) x) :
    HasDerivAt (fun z => (∑ i, -weight i * logp i z) / (n:ℝ))
      ((∑ i, -weight i * slope i) / (n:ℝ)) x := by
  exact (HasDerivAt.sum fun i _ => (hd i).const_mul (-weight i)).div_const _

/-- Including the sampled action in its own group mean contracts its
advantage by (G−1)/G relative to leaving that action out. -/
theorem self_including_baseline (reward others : ℝ) {size : ℝ} (h : 1 < size) :
    reward - (reward+others)/size =
      ((size-1)/size)*(reward-others/(size-1)) := by
  field_simp [show size ≠ 0 by linarith, show size-1 ≠ 0 by linarith]
  ring

def temperatureProb {n : ℕ} (logit : Fin n → ℝ) (tau : ℝ) (i : Fin n) :=
  exp (logit i / tau) / (∑ j, exp (logit j / tau))

theorem temperature_mass {n : ℕ} [NeZero n] (logit : Fin n → ℝ) {tau : ℝ} (_ht : 0 < tau) :
    ∑ i, temperatureProb logit tau i = 1 := by
  unfold temperatureProb
  rw [← sum_div, div_self (by positivity)]

theorem temperature_logprob {n : ℕ} [NeZero n] (logit : Fin n → ℝ) {tau : ℝ}
    (_ht : 0 < tau) (i : Fin n) :
    log (temperatureProb logit tau i) = logit i / tau - log (∑ j, exp (logit j / tau)) := by
  rw [temperatureProb, log_div (exp_pos _).ne' (by positivity), log_exp]

theorem corrected_k3 {n : ℕ} (p q mu : Fin n → ℝ) (hm : ∀ i, 0 < mu i)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (pm : ∑ i, p i = 1) (qm : ∑ i, q i = 1) :
    ∑ i, mu i * (p i / mu i) * k3 (log (q i) - log (p i)) =
    ∑ i, p i * (log (p i) - log (q i)) := by
  rw [change_measure p mu _ hm, k3_on_policy p q hp hq pm qm]

/-- At a fixed context, an unconstrained categorical policy maximizes the
empirical advantage-weighted log likelihood by normalizing the positive
weights. Model sharing across contexts imposes additional constraints. -/
theorem weighted_likelihood_optimal {n : ℕ} [NeZero n] (w q : Fin n → ℝ)
    (hw : ∀ i, 0 < w i) (hq : ∀ i, 0 < q i) (qm : ∑ i, q i = 1) :
    ∑ i, w i * log (q i) ≤ ∑ i, w i * log (w i / ∑ j, w j) := by
  let z := ∑ i, w i
  let p := fun i => w i / z
  have hz : 0 < z := sum_pos (fun i _ => hw i) univ_nonempty
  have hp (i) : 0 < p i := div_pos (hw i) hz
  have pm : ∑ i, p i = 1 := by
    simp only [p, ← sum_div, z, div_self hz.ne']
  have hkl : 0 ≤ ∑ i, p i * (log (p i) - log (q i)) := by
    rw [← k3_on_policy p q hp hq pm qm]
    exact sum_nonneg fun i _ => mul_nonneg (hp i).le (k3_nonnegative _)
  have term (i : Fin n) : z * (p i * (log (p i) - log (q i))) =
      w i * log (p i) - w i * log (q i) := by
    rw [← mul_assoc, show z * p i = w i by dsimp [p]; field_simp]
    ring
  have scaled := mul_nonneg hz.le hkl
  rw [mul_sum] at scaled
  simp_rw [term, sum_sub_distrib] at scaled
  exact sub_nonneg.mp scaled

/-- Differentiating k3 on fixed draws omits the score contribution from
the sampling measure. This identity uses current-policy draws at x. -/
theorem fixed_draw_k3_gradient {n : ℕ} {p : Fin n → ℝ → ℝ} (q : Fin n → ℝ)
    {dp : Fin n → ℝ} {x : ℝ} (hp : ∀ i, 0 < p i x) (hq : ∀ i, 0 < q i)
    (hd : ∀ i, HasDerivAt (p i) (dp i) x) :
    HasDerivAt (fun y => ∑ i, p i x * k3 (log (q i) - log (p i y)))
      (∑ i, (p i x - q i) * (dp i / p i x)) x := by
  apply HasDerivAt.sum
  intro i _
  have gap := ((hd i).log (hp i).ne').const_sub (log (q i))
  have h := (((gap.exp.sub gap).sub_const 1).const_mul (p i x))
  convert h using 1
  rw [exp_sub, exp_log (hq i), exp_log (hp i)]
  field_simp [(hp i).ne']
  ring

/-- The gradient of the forward KL includes the derivative of policy mass. -/
theorem forward_kl_gradient {n : ℕ} {p : Fin n → ℝ → ℝ} (q : Fin n → ℝ)
    {dp : Fin n → ℝ} {x : ℝ} (hp : ∀ i, 0 < p i x)
    (hd : ∀ i, HasDerivAt (p i) (dp i) x) (hm : ∀ y, ∑ i, p i y = 1) :
    HasDerivAt (fun y => ∑ i, p i y * (log (p i y) - log (q i)))
      (∑ i, dp i * (log (p i x) - log (q i))) x := by
  have h (i : Fin n) := (hd i).mul (((hd i).log (hp i).ne').sub_const (log (q i)))
  have term (i : Fin n) : HasDerivAt (fun y => p i y * (log (p i y) - log (q i)))
      (dp i * (log (p i x) - log (q i)) + dp i) x := by
    convert h i using 1
    field_simp [(hp i).ne']
  have total := HasDerivAt.sum (u := univ) fun i _ => term i
  simpa only [sum_add_distrib, mass_derivative hd hm, add_zero] using total

/-- Differentiating the current-to-behavior ratio as well as k3 recovers
the forward-KL gradient for a fixed full-support behavior policy. -/
theorem corrected_k3_gradient {n : ℕ} {p : Fin n → ℝ → ℝ} (q mu : Fin n → ℝ)
    {dp : Fin n → ℝ} {x : ℝ} (hp : ∀ i y, 0 < p i y) (hq : ∀ i, 0 < q i)
    (hmu : ∀ i, 0 < mu i) (hd : ∀ i, HasDerivAt (p i) (dp i) x)
    (hm : ∀ y, ∑ i, p i y = 1) (qm : ∑ i, q i = 1) :
    HasDerivAt (fun y => ∑ i, mu i * (p i y / mu i) * k3 (log (q i) - log (p i y)))
      (∑ i, dp i * (log (p i x) - log (q i))) x := by
  have eqn (y : ℝ) := corrected_k3 (fun i => p i y) q mu hmu (fun i => hp i y) hq (hm y) qm
  simp_rw [eqn]
  exact forward_kl_gradient q (fun i => hp i x) hd hm

/-- Entropy bounds include unvisited actions with p=0. This is the available
action distribution used by the exploration controller and applies to
finite-vocabulary policy entropy as well. -/
theorem entropy_bounds {n : ℕ} [NeZero n] (p : Fin n → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hm : ∑ i, p i = 1) :
    0 ≤ ∑ i, -(p i * log (p i)) ∧ ∑ i, -(p i * log (p i)) ≤ log n := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  constructor
  · apply sum_nonneg
    intro i _
    have h1 : p i ≤ 1 := hm ▸ single_le_sum (fun j _ => hp j) (mem_univ i)
    exact neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos (hp i) (log_nonpos (hp i) h1))
  · have term (i : Fin n) :
        -(p i * log (p i)) ≤ 1/n - p i + p i * log n := by
      by_cases hz : p i = 0
      · simp [hz, hn.le]
      have h := log_le_sub_one_of_pos (one_div_pos.mpr (mul_pos hn (lt_of_le_of_ne (hp i) (Ne.symm hz))))
      rw [log_div one_ne_zero (mul_ne_zero hn.ne' hz), log_one, log_mul hn.ne' hz] at h
      have bound := mul_le_mul_of_nonneg_left h (hp i)
      have cancel : p i * (1 / ((n : ℝ) * p i)) = 1/n := by field_simp; ring
      simp only [mul_sub, cancel] at bound
      linarith
    have h := sum_le_sum (s := (univ : Finset (Fin n))) (fun i _ => term i)
    simpa [sum_add_distrib, sum_sub_distrib, ← sum_mul, hm, nsmul_eq_mul, hn.ne'] using h

end
end StateSetFormal.ScoreFunction
