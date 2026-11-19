import ContinualReplay
import PolicyMath

/-! Selection of continual replay groups and their share of a normalized
    native GRPO gradient at fixed learner parameters and group-local advantages. -/
namespace StateSetFormal.ReplayMixture
open Finset
open scoped BigOperators
noncomputable section

/-- The difference between a priority-weighted draw and a uniform draw is the
    population covariance of priority and gradient divided by mean priority. -/
theorem priority_bias {n : ℕ} (w g : Fin n → ℝ) (hn : (n:ℝ) ≠ 0) (hw : ∑ i, w i ≠ 0) :
    (∑ i, w i*g i)/(∑ i, w i) - (∑ i, g i)/n =
      (∑ i, (w i-(∑ j, w j)/n)*(g i-(∑ j, g j)/n))/(∑ i, w i) := by
  simp only [sub_mul, mul_sub, sum_sub_distrib, ← sum_mul, ← mul_sum, sum_const,
    card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp [hn, hw]
  ring_nf
  simp

/-- Conditional on the stored groups, reciprocal draw probabilities recover
    the uniform-buffer gradient under positive priority for every entry. -/
theorem uniform_correction {n : ℕ} (w g : Fin n → ℝ) (positive : ∀ i, w i ≠ 0)
    (total : ∑ i, w i ≠ 0) (hn : (n:ℝ) ≠ 0) :
    ∑ i, (w i/(∑ j, w j)) * ((∑ j, w j)/(n*w i)) * g i = (∑ i, g i)/n := by
  rw [sum_div]
  apply sum_congr rfl
  intro i _
  field_simp [positive i]
  ring

/-- For any without-replacement selection law, each historical entry
    contributes its gradient times its marginal inclusion probability. -/
theorem inclusion_sum {n m : ℕ} (prob : Fin n → ℝ)
    (selected : Fin n → Finset (Fin m)) (g : Fin m → ℝ) :
    (∑ b, prob b * ∑ i ∈ selected b, g i) =
      ∑ i, (∑ b, if i ∈ selected b then prob b else 0) * g i := by
  have expand (s : Finset (Fin m)) : (∑ i ∈ s, g i) = ∑ i, if i ∈ s then g i else 0 := by
    rw [← sum_filter]
    simp
  simp_rw [expand, mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i _
  rw [sum_mul]
  apply sum_congr rfl
  intro b _
  by_cases h : i ∈ selected b <;> simp [h]

/-- For any uniform within-task quota, summing the gradient over every
    possible subbatch counts each task entry exactly the same number of times. -/
theorem uniform_task_total (pool : Finset ℕ) (r : ℕ) (g : ℕ → ℝ) :
    (∑ s ∈ pool.powersetCard (r+1), ∑ i ∈ s, g i) =
    (((pool.card-1).choose r : ℕ):ℝ) * ∑ i ∈ pool, g i := by
  have hits (i : ℕ) (hi : i ∈ pool) :
      ((pool.powersetCard (r+1)).filter (fun s => i ∈ s)).card = (pool.card-1).choose r := by
    simpa using ContinualReplay.subbatch_completions pool {i} (r+1) (by simpa) (by simp)
  calc
    _ = ∑ s ∈ pool.powersetCard (r+1), ∑ i ∈ pool, if i ∈ s then g i else 0 := by
      apply sum_congr rfl
      intro s hs
      rw [← sum_filter]
      congr 1
      ext i
      simp only [mem_filter]
      exact ⟨fun h => ⟨(mem_powersetCard.mp hs).1 h, h⟩, And.right⟩
    _ = _ := by
      rw [sum_comm, mul_sum]
      apply sum_congr rfl
      intro i hi
      rw [← sum_filter, sum_const, nsmul_eq_mul, hits i hi]

/-- One complete balanced round averages task means. Compared with a uniform
    entry draw its bias is minus the covariance of task size and task mean
    divided by mean task size. -/
theorem balanced_vs_uniform {t : ℕ} (n : Fin t → ℕ) (g : (i : Fin t) → Fin (n i) → ℝ)
    (tasks : (t:ℝ) ≠ 0) (populated : ∀ i, (n i:ℝ) ≠ 0)
    (total : (∑ i, (n i:ℝ)) ≠ 0) :
    let means := fun i => (∑ j, g i j)/(n i:ℝ)
    (∑ i, means i)/t-(∑ i, ∑ j, g i j)/(∑ i, (n i:ℝ)) =
      -(∑ i, ((n i:ℝ)-(∑ j, (n j:ℝ))/t)*(means i-(∑ j, means j)/t))/
        (∑ i, (n i:ℝ)) := by
  dsimp only
  have h := priority_bias (fun i => (n i:ℝ)) (fun i => (∑ j, g i j)/(n i:ℝ)) tasks total
  have cancel (i : Fin t) : (n i:ℝ)*((∑ j, g i j)/(n i:ℝ)) = ∑ j, g i j := by
    field_simp [populated i]
  simp_rw [cancel] at h
  rw [neg_div, ← h, neg_sub]

/-- One distinguished weight in a uniform reservoir, followed by one
    priority draw from that reservoir. Other arrivals all have weight u. -/
theorem distinguished_draw (others : Finset ℕ) (x r : ℕ) (fresh : x ∉ others)
    (fits : r ≤ others.card) (w u : ℝ) :
    let reservoirs := (insert x others).powersetCard (r+1)
    (∑ s ∈ reservoirs, if x ∈ s then w / (∑ j ∈ s, if j=x then w else u) else 0) /
        reservoirs.card =
      ((r:ℝ)+1)/((others.card:ℝ)+1) * (w/(w+r*u)) := by
  dsimp only
  let sets := (insert x others).powersetCard (r+1)
  let kept := sets.filter (fun s => x ∈ s)
  have den {s : Finset ℕ} (hs : s ∈ sets) (hx : x ∈ s) :
      (∑ j ∈ s, if j=x then w else u) = w+r*u := by
    have size := (mem_powersetCard.mp hs).2
    have term (j : ℕ) : (if j=x then w else u) = u + (if j=x then w-u else 0) := by
      split <;> simp_all
    simp only [term, sum_add_distrib, sum_const, nsmul_eq_mul, sum_ite_eq', hx, if_true, size]
    push_cast
    ring
  have numerator : (∑ s ∈ sets, if x ∈ s then w/(∑ j ∈ s, if j=x then w else u) else 0) =
      (kept.card:ℝ)*(w/(w+r*u)) := by
    calc
      _ = ∑ s ∈ sets, if x ∈ s then w/(w+r*u) else 0 := by
        apply sum_congr rfl
        intro s hs
        by_cases hx : x ∈ s
        · rw [if_pos hx, if_pos hx, den hs hx]
        · rw [if_neg hx, if_neg hx]
      _ = _ := by rw [← sum_filter]; simp [kept, nsmul_eq_mul]
  have count : ((others.card:ℝ)+1) * kept.card = (sets.card:ℝ)*((r:ℝ)+1) := by
    exact_mod_cast ContinualReplay.partial_round others x r fresh
  have enough : r+1 ≤ (insert x others).card := by simp [fresh, fits]
  have nonempty : 0 < sets.card := card_pos.mpr (powersetCard_nonempty.mpr enough)
  have pos : (sets.card:ℝ) ≠ 0 := by exact_mod_cast nonempty.ne'
  change _ / (sets.card:ℝ) = _
  rw [numerator]
  have prob : (kept.card:ℝ)/(sets.card:ℝ) = ((r:ℝ)+1)/((others.card:ℝ)+1) := by
    apply (div_eq_div_iff pos (by positivity)).mpr
    linarith
  rw [← prob]
  ring

/-- For one heavier entry, increasing reservoir capacity increases its
    marginal priority-draw probability by this amount. -/
theorem capacity_increment (n k w u : ℝ) (hn : 0 < n) (hk : 1 ≤ k)
    (hw : 0 < w) (hu : 0 ≤ u) :
    (k+1)/n*(w/(w+k*u))-k/n*(w/(w+(k-1)*u)) =
      w*(w-u)/(n*(w+k*u)*(w+(k-1)*u)) := by
  have old : 0 < w+(k-1)*u := add_pos_of_pos_of_nonneg hw (mul_nonneg (sub_nonneg.mpr hk) hu)
  have newMass : 0 < w+k*u := add_pos_of_pos_of_nonneg hw (mul_nonneg (by linarith) hu)
  field_simp [hn.ne', old.ne', newMass.ne']
  ring

/-- Replay mass in a reduction with positive fresh and nonnegative replay mass. -/
def share (fresh replay : ℝ) := replay/(fresh+replay)

/-- A draw can have its own token or trajectory count and its own gradient.
    This identity conditions on the realized fresh batch and includes zero replay. -/
theorem mixed_bias {n : ℕ} (p count grad : Fin n → ℝ) (fresh base : ℝ)
    (hf : 0 < fresh) (hc : ∀ i, 0 ≤ count i) (mass : ∑ i, p i = 1) :
    (∑ i, p i*((fresh*base+count i*grad i)/(fresh+count i))) - base =
      ∑ i, p i*share fresh (count i)*(grad i-base) := by
  have weighted : ∑ i, p i*base = base := by rw [← sum_mul, mass, one_mul]
  conv_lhs => rhs; rw [← weighted]
  rw [← sum_sub_distrib]
  apply sum_congr rfl
  intro i _
  unfold share
  field_simp [show fresh+count i ≠ 0 from (add_pos_of_pos_of_nonneg hf (hc i)).ne']
  ring

/-- Exact Jensen gap for any finite law of total replay mass, including a
    whole accumulation window. No independence of its microbatches is needed. -/
theorem pooled_share_gap {n : ℕ} (prob count : Fin n → ℝ) (fresh mean : ℝ)
    (hf : 0 < fresh) (hm : 0 ≤ mean) (hc : ∀ i, 0 ≤ count i)
    (norm : ∑ i, prob i = 1) (expectation : ∑ i, prob i*count i = mean) :
    share fresh mean-(∑ i, prob i*share fresh (count i)) =
      ∑ i, prob i*(fresh*(count i-mean)^2/((fresh+mean)^2*(fresh+count i))) := by
  have centered : ∑ i, prob i*(count i-mean) = 0 := by
    simp only [mul_sub, sum_sub_distrib, ← sum_mul, norm, one_mul, expectation, sub_self]
  have term (i : Fin n) :
      prob i*(fresh*(count i-mean)^2/((fresh+mean)^2*(fresh+count i))) =
        prob i*share fresh mean-prob i*share fresh (count i) +
          (fresh/(fresh+mean)^2)*(prob i*(count i-mean)) := by
    unfold share
    field_simp [show fresh+mean ≠ 0 from (add_pos_of_pos_of_nonneg hf hm).ne',
      show fresh+count i ≠ 0 from (add_pos_of_pos_of_nonneg hf (hc i)).ne']
    ring
  simp_rw [term]
  rw [sum_add_distrib, sum_sub_distrib, ← mul_sum, centered, mul_zero, add_zero,
    ← sum_mul, norm, one_mul]

/-- Stochastic rounding before averaging gives a smaller replay fraction than
    normalizing the expected counts. The rational expression is the exact gap. -/
theorem rounding_gap (fresh whole t : ℝ) (hf : 0 < fresh) (hw : 0 ≤ whole) (ht : 0 ≤ t) :
    share fresh (whole+t)-((1-t)*share fresh whole+t*share fresh (whole+1)) =
      fresh*t*(1-t)/((fresh+whole)*(fresh+whole+1)*(fresh+whole+t)) := by
  unfold share
  field_simp [show fresh+whole ≠ 0 by positivity,
    show fresh+(whole+1) ≠ 0 by positivity, show fresh+(whole+t) ≠ 0 by positivity,
    show fresh+whole+1 ≠ 0 by positivity, show fresh+whole+t ≠ 0 by positivity]
  ring

theorem rounding_strict (fresh whole t : ℝ) (hf : 0 < fresh) (hw : 0 ≤ whole)
    (ht : 0 < t) (hu : t < 1) :
    (1-t)*share fresh whole+t*share fresh (whole+1) < share fresh (whole+t) := by
  apply sub_pos.mp
  rw [rounding_gap _ _ _ hf hw ht.le]
  have : 0 < 1-t := sub_pos.mpr hu
  positivity

end
end StateSetFormal.ReplayMixture
