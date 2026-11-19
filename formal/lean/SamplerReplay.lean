import ScoreFunction

namespace StateSetFormal.SamplerReplay
open Finset Real StateSetFormal.Policy
open scoped BigOperators
noncomputable section

/-- The action law after a processor assigns positive weights on its retained support. -/
def sampled {n : ℕ} (keep : Finset (Fin n)) (w : Fin n → ℝ) (i : Fin n) :=
  if i ∈ keep then w i / (∑ j ∈ keep, w j) else 0

theorem normalized {n : ℕ} (keep : Finset (Fin n)) (w : Fin n → ℝ)
    (hz : 0 < ∑ j ∈ keep, w j) : ∑ i, sampled keep w i = 1 := by
  simp [sampled, ← sum_div, hz.ne']

theorem recorded_log {n : ℕ} {keep : Finset (Fin n)} {w : Fin n → ℝ} {i : Fin n}
    (hi : i ∈ keep) (hw : 0 < w i) (hz : 0 < ∑ j ∈ keep, w j) :
    log (sampled keep w i) = log (w i) - log (∑ j ∈ keep, w j) := by
  rw [sampled, if_pos hi, Real.log_div hw.ne' hz.ne']

/-- A dominant finite sampler weight can use the complementary mass after
    rounding, preserving small nonzero log probabilities near mass one. -/
theorem complement_log {chosen rest : ℝ} (hc : 0 < chosen) (hr : 0 ≤ rest) :
    log (chosen / (chosen + rest)) = -log (1 + rest / chosen) := by
  have h : chosen / (chosen + rest) = (1 + rest / chosen)⁻¹ := by
    rw [← one_div, one_add_div hc.ne']
    field_simp
  rw [h, log_inv]

def tailMass {n : ℕ} (z : Fin n → ℝ) (temp : ℝ) (pivot : Fin n) :=
  ∑ j ∈ univ.erase pivot, exp ((z j-z pivot)/temp)

/-- The complementary normalizer used by PPO gives the exact action log
probability for every pivot, including a tie between largest logits. -/
theorem pivot_score {n : ℕ} [NeZero n] (z : Fin n → ℝ) {temp : ℝ} (ht : 0 < temp)
    (pivot action : Fin n) :
    log (ScoreFunction.temperatureProb z temp action) =
      (z action-z pivot)/temp - log (1+tailMass z temp pivot) := by
  have mass : 1 + tailMass z temp pivot = (∑ j, exp (z j/temp)) / exp (z pivot/temp) := by
    calc
      _ = ∑ j, exp ((z j-z pivot)/temp) := by
        have erase := add_sum_erase univ (fun j => exp ((z j-z pivot)/temp)) (mem_univ pivot)
        simp only [sub_self, zero_div, exp_zero] at erase
        exact erase
      _ = _ := by simp_rw [sub_div, exp_sub, ← sum_div]
  rw [ScoreFunction.temperature_logprob z ht, mass, log_div (by positivity) (exp_pos _).ne', log_exp]
  ring

/-- The pivot's directional score is a sum over the other actions. Each
term retains its tail probability even when the pivot mass rounds to one. -/
theorem pivot_gradient {n : ℕ} [NeZero n] (z : Fin n → ℝ → ℝ) (v : Fin n → ℝ)
    {temp x : ℝ} (ht : 0 < temp) (pivot : Fin n) (hd : ∀ i, HasDerivAt (z i) (v i) x) :
    HasDerivAt (fun u => log (ScoreFunction.temperatureProb (fun j => z j u) temp pivot))
      (-(∑ j ∈ univ.erase pivot, exp ((z j x-z pivot x)/temp) * ((v j-v pivot)/temp)) /
        (1+tailMass (fun j => z j x) temp pivot)) x := by
  have h (u : ℝ) : log (ScoreFunction.temperatureProb (fun j => z j u) temp pivot) =
      -log (1+tailMass (fun j => z j u) temp pivot) := by simpa using pivot_score (fun j => z j u) ht pivot pivot
  simp_rw [h, neg_div]
  have tail : HasDerivAt (fun u => tailMass (fun j => z j u) temp pivot)
      (∑ j ∈ univ.erase pivot, exp ((z j x-z pivot x)/temp) * ((v j-v pivot)/temp)) x :=
    HasDerivAt.sum fun j _ => (((hd j).sub (hd pivot)).div_const temp).exp
  exact ((tail.const_add 1).log (by dsimp [tailMass]; positivity)).neg

theorem tempered_log {n : ℕ} {keep : Finset (Fin n)} (logits : Fin n → ℝ) (temp : ℝ)
    {i : Fin n} (hi : i ∈ keep) :
    log (sampled keep (fun j => exp (logits j/temp)) i) =
      logits i/temp - log (∑ j ∈ keep, exp (logits j/temp)) := by
  have mass : 0 < ∑ j ∈ keep, exp (logits j/temp) :=
    sum_pos (fun j _ => exp_pos _) ⟨i, hi⟩
  rw [recorded_log hi (exp_pos _) mass, Real.log_exp]

theorem deterministic_log {n : ℕ} (w : Fin n → ℝ) (i : Fin n) (h : 0 < w i) :
    log (sampled {i} w i) = 0 := by simp [sampled, h.ne']

/-- The raw old policy divided by its own filtered action law equals the
    retained raw mass, even before a parameter update. -/
theorem first_ratio {n : ℕ} {keep : Finset (Fin n)} {p : Fin n → ℝ} {i : Fin n}
    (hi : i ∈ keep) (hp : 0 < p i) :
    p i / sampled keep p i = ∑ j ∈ keep, p j := by
  rw [sampled, if_pos hi]
  field_simp [hp.ne']

theorem correct_on_support {n : ℕ} (keep : Finset (Fin n)) (w p a : Fin n → ℝ)
    (hw : ∀ i ∈ keep, 0 < w i) (hz : 0 < ∑ j ∈ keep, w j) :
    (∑ i ∈ keep, sampled keep w i * (p i / sampled keep w i) * a i) =
      ∑ i ∈ keep, p i * a i := by
  apply sum_congr rfl
  intro i hi
  have pos : 0 < sampled keep w i := by simp [sampled, hi, div_pos (hw i hi) hz]
  field_simp [pos.ne']

/-- The clipped estimator is pessimistic on the sampled support at a
    fixed prefix, for arbitrary signed advantages and positive weights. -/
theorem clipped_on_support {n : ℕ} (keep : Finset (Fin n)) (w p a : Fin n → ℝ)
    (lo hi : ℝ) (hw : ∀ i ∈ keep, 0 < w i) (hz : 0 < ∑ j ∈ keep, w j) :
    (∑ i ∈ keep, sampled keep w i * (-surrogate lo hi (p i / sampled keep w i) (a i))) ≤
      ∑ i ∈ keep, p i * a i := by
  rw [← correct_on_support keep w p a hw hz]
  apply sum_le_sum
  intro i hi'
  have pos : 0 < sampled keep w i := by simp [sampled, hi', div_pos (hw i hi') hz]
  rw [mul_assoc]
  apply mul_le_mul_of_nonneg_left _ pos.le
  simp only [surrogate, neg_neg]
  exact min_le_left ..

end
end StateSetFormal.SamplerReplay
