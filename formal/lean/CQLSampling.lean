import OfflineMath
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Probability.Variance

/-! Sampling error of CQL's normalized log partition at a fixed state and frozen critic. -/
namespace StateSetFormal.CQLSampling
open MeasureTheory ProbabilityTheory Finset Real
open scoped BigOperators
noncomputable section

/-- The argument of the logarithm in the sampled backup. -/
def sampleMean {Ω : Type*} {n : ℕ} (X : Fin n → Ω → ℝ) (ω : Ω) : ℝ :=
  (n : ℝ)⁻¹ * (∑ i, X i) ω

theorem softBackup_eq {Ω : Type*} {n : ℕ} [NeZero n] (q : Fin n → Ω → ℝ) (t : ℝ) (ω : Ω) :
    Offline.softBackup t (fun i => q i ω) =
      t * log (sampleMean (fun i z => exp (q i z/t)) ω) := by
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  simp only [Offline.softBackup, sampleMean, sum_apply]
  rw [mul_comm (n : ℝ)⁻¹, ← div_eq_mul_inv, log_div (by positivity) hn]

theorem penalty_offset {n : ℕ} (q : Fin n → ℝ) (t logged : ℝ) :
    Offline.cqlGap t q logged = Offline.softBackup t q + (t * log n - logged) := by
  unfold Offline.cqlGap Offline.softBackup
  ring

/-- Tangent error of the log is nonnegative and at most quadratic.
The lower bound on the sampled mass controls the denominator. -/
theorem log_remainder {z a m : ℝ} (hm : 0 < m) (hz : m ≤ z) (ha : 0 < a) :
    0 ≤ (z-a)/a - (log z-log a) ∧
    (z-a)/a - (log z-log a) ≤ (z-a)^2/(m*a) := by
  have hzp := hm.trans_le hz
  have upper := log_le_sub_one_of_pos (div_pos hzp ha)
  have lower := log_le_sub_one_of_pos (div_pos ha hzp)
  rw [log_div hzp.ne' ha.ne'] at upper
  rw [log_div ha.ne' hzp.ne'] at lower
  constructor
  · rw [sub_div, div_self ha.ne']
    linarith
  · calc
      _ ≤ (z-a)/a + (a/z-1) := by linarith
      _ = (z-a)^2/(z*a) := by field_simp; ring
      _ ≤ _ := div_le_div_of_nonneg_left (sq_nonneg _) (mul_pos hm ha)
        (mul_le_mul_of_nonneg_right hz ha.le)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- The mean and variance of the empirical average under pairwise independent
draws with common first and second centered moments. -/
theorem sample_moments {n : ℕ} [NeZero n] (X : Fin n → Ω → ℝ) (a v : ℝ)
    (hX : ∀ i, Memℒp (X i) 2 μ) (he : ∀ i, ∫ ω, X i ω ∂μ = a)
    (hv : ∀ i, variance (X i) μ = v)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) μ) :
    Memℒp (sampleMean X) 2 μ ∧ (∫ ω, sampleMean X ω ∂μ) = a ∧
      variance (sampleMean X) μ = v/n := by
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  refine ⟨(memℒp_finset_sum' univ (fun i _ => hX i)).const_mul _, ?_, ?_⟩
  · simp only [sampleMean, sum_apply, integral_mul_left,
      integral_finset_sum univ (fun i _ => (hX i).integrable one_le_two), he,
      sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  · unfold sampleMean
    rw [variance_mul, IndepFun.variance_sum (fun i _ => hX i)
        (fun _ _ _ _ hij => hind hij)]
    simp only [hv, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
    ring

private theorem integrable_log {Z : Ω → ℝ} (hi : Integrable Z μ)
    {m : ℝ} (hm : 0 < m) (hz : ∀ᵐ ω ∂μ, m ≤ Z ω) : Integrable (fun ω => log (Z ω)) μ := by
  apply (hi.add (integrable_const |log m|)).mono' hi.aemeasurable.log.aestronglyMeasurable
  filter_upwards [hz] with ω hω
  rw [Real.norm_eq_abs, Pi.add_apply]
  apply abs_le.mpr
  have hl := log_le_log hm hω
  have hu := log_le_self (hm.le.trans hω)
  constructor <;> linarith [abs_nonneg (log m), neg_abs_le (log m)]

/-- Integrate the log remainder. The linear term vanishes at the population mean. -/
theorem log_mean_bias {Z : Ω → ℝ} (hZ : Memℒp Z 2 μ) {m a : ℝ} (hm : 0 < m)
    (lower : ∀ᵐ ω ∂μ, m ≤ Z ω) (mean : (∫ ω, Z ω ∂μ) = a) :
    0 ≤ log a - ∫ ω, log (Z ω) ∂μ ∧
      log a - (∫ ω, log (Z ω) ∂μ) ≤ variance Z μ / (m*a) := by
  have hi := hZ.integrable one_le_two
  have hlog := integrable_log hi hm lower
  have ha : 0 < a := by
    have h := integral_mono_ae (integrable_const m) hi lower
    exact hm.trans_le (by simpa [mean] using h)
  let D := fun ω => (Z ω-a)/a - (log (Z ω)-log a)
  have hD : Integrable D μ :=
    (((hi.sub (integrable_const a)).div_const a).sub (hlog.sub (integrable_const _)))
  have hsq : Integrable (fun ω => (Z ω-a)^2/(m*a)) μ :=
    ((hZ.sub (memℒp_const a)).integrable_sq).div_const _
  have hid : (∫ ω, D ω ∂μ) = log a - ∫ ω, log (Z ω) ∂μ := by
    dsimp [D]
    rw [integral_sub, integral_div, integral_sub hi (integrable_const _),
        integral_sub hlog (integrable_const _)]
    · simp [mean]
    · exact (hi.sub (integrable_const a)).div_const a
    · exact hlog.sub (integrable_const _)
  have hvar : variance Z μ = ∫ ω, (Z ω-a)^2 ∂μ := by
    rw [hZ.variance_eq, ← mean]
    rfl
  rw [← hid, hvar, ← integral_div]
  exact ⟨integral_nonneg_of_ae (lower.mono fun _ hz => (log_remainder hm hz ha).1),
    integral_mono_ae hD hsq (lower.mono fun _ hz => (log_remainder hm hz ha).2)⟩

omit [IsProbabilityMeasure μ] in
private theorem exp_indep {n : ℕ} {q : Fin n → Ω → ℝ} (t : ℝ)
    (hind : Pairwise fun i j => IndepFun (q i) (q j) μ) :
    Pairwise fun i j => IndepFun (fun ω => exp (q i ω/t)) (fun ω => exp (q j ω/t)) μ :=
  fun _ _ hij => (hind hij).comp (φ := fun x : ℝ => exp (x/t))
    (ψ := fun x : ℝ => exp (x/t)) (by fun_prop) (by fun_prop)

omit [IsProbabilityMeasure μ] in
private theorem sample_floor {n : ℕ} [NeZero n] {X : Fin n → Ω → ℝ} {m : ℝ}
    (lower : ∀ᵐ ω ∂μ, ∀ i, m ≤ X i ω) : ∀ᵐ ω ∂μ, m ≤ sampleMean X ω := by
  have hn : (0 : ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  filter_upwards [lower] with ω h
  have sum := sum_le_sum (s := univ) fun i _ => h i
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at sum
  rw [sampleMean, sum_apply, mul_comm, ← div_eq_mul_inv, le_div_iff₀ hn, mul_comm]
  exact sum

/-- At a frozen state, the bias of the expected CQL soft backup is bounded
by temperature times the proposal variance divided by n, the floor and the mean
of the exponentiated score. The underlying proposal law may be continuous. -/
theorem expected_softBackup_bias {n : ℕ} [NeZero n] (q : Fin n → Ω → ℝ) {t m a v : ℝ}
    (ht : 0 < t) (hm : 0 < m)
    (hind : Pairwise fun i j => IndepFun (q i) (q j) μ)
    (hX : ∀ i, Memℒp (fun ω => exp (q i ω/t)) 2 μ)
    (he : ∀ i, (∫ ω, exp (q i ω/t) ∂μ) = a)
    (hv : ∀ i, variance (fun ω => exp (q i ω/t)) μ = v)
    (lower : ∀ᵐ ω ∂μ, ∀ i, m ≤ exp (q i ω/t)) :
    let gap := t*log a - ∫ ω, Offline.softBackup t (fun i => q i ω) ∂μ
    0 ≤ gap ∧ gap ≤ t*v/(n*m*a) := by
  let X := fun i ω => exp (q i ω/t)
  obtain ⟨hZ, heZ, hvZ⟩ := sample_moments X a v hX he hv (exp_indep t hind)
  have hfloor := sample_floor (X := X) lower
  obtain ⟨hl, hu⟩ := log_mean_bias hZ hm hfloor heZ
  simp_rw [softBackup_eq q t, integral_mul_left, ← mul_sub]
  refine ⟨mul_nonneg ht.le hl, (mul_le_mul_of_nonneg_left hu ht.le).trans_eq ?_⟩
  rw [hvZ]
  ring

/-- Relative control of the positive mass implies absolute control in log space. -/
private theorem log_window {a z δ : ℝ} (ha : 0 < a) (hu : δ < 1)
    (within : |z-a| < a*δ) : |log z-log a| < -log (1-δ) := by
  obtain ⟨hl, hh⟩ := abs_lt.mp within
  have hp : 0 < 1-δ := sub_pos.mpr hu
  have lo : a*(1-δ) < z := by linarith
  have hi : z < a/(1-δ) := by
    apply (lt_div_iff₀ hp).mpr
    have hz : z < a*(1+δ) := by linarith
    have mult := mul_lt_mul_of_pos_right hz hp
    nlinarith [mul_nonneg ha.le (sq_nonneg δ)]
  have lower := log_lt_log (mul_pos ha hp) lo
  have upper := log_lt_log ((mul_pos ha hp).trans lo) hi
  rw [log_mul ha.ne' hp.ne'] at lower
  rw [log_div ha.ne' hp.ne'] at upper
  exact abs_lt.mpr ⟨by linarith, by linarith⟩

/-- A single bank can overshoot or undershoot. Chebyshev gives a two-sided
probability bound even without a uniform positive floor on each proposal mass. -/
theorem softBackup_tail {n : ℕ} [NeZero n] (q : Fin n → Ω → ℝ) {t a v δ : ℝ}
    (ht : 0 < t) (ha : 0 < a) (hd : 0 < δ) (hu : δ < 1)
    (hind : Pairwise fun i j => IndepFun (q i) (q j) μ)
    (hX : ∀ i, Memℒp (fun ω => exp (q i ω/t)) 2 μ)
    (he : ∀ i, (∫ ω, exp (q i ω/t) ∂μ) = a)
    (hv : ∀ i, variance (fun ω => exp (q i ω/t)) μ = v) :
    μ {ω | -(t * log (1-δ)) ≤ |Offline.softBackup t (fun i => q i ω) - t*log a|} ≤
      ENNReal.ofReal (v/(n*(a*δ)^2)) := by
  let X := fun i ω => exp (q i ω/t)
  obtain ⟨hZ, heZ, hvZ⟩ := sample_moments X a v hX he hv (exp_indep t hind)
  have cheb := meas_ge_le_variance_div_sq hZ (mul_pos ha hd)
  rw [heZ, hvZ, div_div] at cheb
  apply le_trans (measure_mono _) cheb
  intro ω bad
  change a*δ ≤ |sampleMean X ω-a|
  by_contra! small
  have close := mul_lt_mul_of_pos_left (log_window ha hu small) ht
  change -(t * log (1-δ)) ≤ |Offline.softBackup t (fun i => q i ω)-t*log a| at bad
  rw [softBackup_eq, ← mul_sub, abs_mul, abs_of_pos ht] at bad
  change -(t * log (1-δ)) ≤ t * |log (sampleMean X ω)-log a| at bad
  linarith

/-- The unnormalized conservative penalty has the same bias after accounting
for its proposal-count offset and the logged action at the fixed state. -/
theorem expected_penalty_gap {n : ℕ} (q : Fin n → Ω → ℝ) (t a logged : ℝ)
    (hi : Integrable (fun ω => Offline.softBackup t (fun i => q i ω)) μ) :
    t*log a + (t*log n-logged) - (∫ ω, Offline.cqlGap t (fun i => q i ω) logged ∂μ) =
      t*log a - ∫ ω, Offline.softBackup t (fun i => q i ω) ∂μ := by
  simp_rw [penalty_offset]
  rw [integral_add hi (integrable_const _)]
  simp

/-- Termination removes all proposal error. On a live row the discount scales it. -/
theorem bellman_bias (S : Ω → ℝ) (hS : Integrable S μ) (γ r population : ℝ) (done : Bool) :
    Returns.backup γ r population done - (∫ ω, Returns.backup γ r (S ω) done ∂μ) =
      if done then 0 else γ*(population-∫ ω, S ω ∂μ) := by
  cases done
  · simp only [Returns.backup, Bool.false_eq_true, if_false]
    rw [integral_add (integrable_const r) (hS.const_mul γ), integral_mul_left]
    simp
    ring
  · simp [Returns.backup]

omit [MeasurableSpace Ω] in
/-- Centering the softmax-weighted slope leaves a ratio of two sample means. -/
theorem slope_center {n : ℕ} [NeZero n] (X D : Fin n → Ω → ℝ) (b : ℝ) (ω : Ω)
    (hz : (∑ i, X i ω) ≠ 0) :
    (∑ i, X i ω / (∑ j, X j ω) * D i ω) - b =
      sampleMean (fun i z => X i z*(D i z-b)) ω / sampleMean X ω := by
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  simp only [sampleMean, sum_apply, div_mul_eq_mul_div, ← sum_div, mul_sub,
    sum_sub_distrib, ← sum_mul]
  field_simp

/-- The mean-zero numerator cancels the linear part of the ratio error.
The scale c can be chosen to balance the two variance contributions. -/
private theorem ratio_bias {H Z : Ω → ℝ} {m a c : ℝ}
    (hm : 0 < m) (ha : 0 < a) (hc : 0 < c)
    (hH : Memℒp H 2 μ) (hZ : Memℒp Z 2 μ) (eH : (∫ ω, H ω ∂μ) = 0)
    (hfloor : ∀ᵐ ω ∂μ, m ≤ Z ω) :
    Integrable (fun ω => H ω/Z ω) μ ∧
    |∫ ω, H ω/Z ω ∂μ| ≤ ((∫ ω, (H ω)^2 ∂μ)/c + c*(∫ ω, (Z ω-a)^2 ∂μ))/(2*m*a) := by
  have hHi := hH.integrable one_le_two
  have hr : Integrable (fun ω => H ω/Z ω) μ := by
    apply (hHi.abs.div_const m).mono' (hH.1.aemeasurable.div hZ.1.aemeasurable).aestronglyMeasurable
    filter_upwards [hfloor] with ω hz
    rw [Real.norm_eq_abs, abs_div, abs_of_pos (hm.trans_le hz)]
    exact div_le_div_of_nonneg_left (abs_nonneg _) hm hz
  refine ⟨hr, ?_⟩
  let error := fun ω => H ω/Z ω - H ω/a
  have ei : Integrable error μ := hr.sub (hHi.div_const a)
  have hid : (∫ ω, error ω ∂μ) = ∫ ω, H ω/Z ω ∂μ := by
    simp only [error, integral_sub hr (hHi.div_const a), integral_div a, eH, zero_div, sub_zero]
  have hsq : Integrable (fun ω => (Z ω-a)^2) μ := (hZ.sub (memℒp_const a)).integrable_sq
  have ui : Integrable (fun ω => ((H ω)^2/c + c*(Z ω-a)^2)/(2*m*a)) μ :=
    ((hH.integrable_sq.div_const c).add (hsq.const_mul c)).div_const _
  have bound : ∀ᵐ ω ∂μ, ‖error ω‖ ≤ ((H ω)^2/c + c*(Z ω-a)^2)/(2*m*a) := by
    filter_upwards [hfloor] with ω hz
    have hzp := hm.trans_le hz
    have eq : error ω = H ω*(a-Z ω)/(Z ω*a) := by
      dsimp [error]
      field_simp
      ring
    rw [Real.norm_eq_abs, eq, abs_div, abs_of_pos (mul_pos hzp ha), abs_mul, abs_sub_comm a]
    calc
      _ ≤ |H ω| * |Z ω-a| / (m*a) := div_le_div_of_nonneg_left (by positivity) (mul_pos hm ha)
        (mul_le_mul_of_nonneg_right hz ha.le)
      _ = (2 * |H ω| * |Z ω-a|)/(2*m*a) := by ring
      _ ≤ _ := by
        apply div_le_div_of_nonneg_right _ (by positivity)
        have young := sq_nonneg (|H ω| - c * |Z ω-a|)
        rw [sub_sq, sq_abs, mul_pow, sq_abs] at young
        field_simp
        apply (le_div_iff₀ hc).2
        nlinarith only [young]
  rw [← hid]
  calc
    _ ≤ ∫ ω, ‖error ω‖ ∂μ := norm_integral_le_integral_norm _
    _ ≤ _ := (integral_mono_ae ei.norm ui bound).trans_eq (by
      rw [integral_div, integral_add (hH.integrable_sq.div_const c) (hsq.const_mul c),
        integral_div, integral_mul_left])

omit [IsProbabilityMeasure μ] in
/-- Under a locally integrable derivative bound the population log partition
has the exponentially weighted score derivative even for continuous proposals. -/
theorem population_derivative (q dq : ℝ → Ω → ℝ) {t x ε : ℝ} (ht : 0 < t) (hε : 0 < ε)
    (bound : Ω → ℝ) (hi : Integrable bound μ)
    (meas : ∀ᶠ z in nhds x, AEStronglyMeasurable (fun ω => exp (q z ω/t)) μ)
    (mass : Integrable (fun ω => exp (q x ω/t)) μ)
    (positive : 0 < ∫ ω, exp (q x ω/t) ∂μ)
    (dm : AEStronglyMeasurable (fun ω => exp (q x ω/t)*dq x ω/t) μ)
    (major : ∀ᵐ ω ∂μ, ∀ z ∈ Metric.ball x ε, ‖exp (q z ω/t)*dq z ω/t‖ ≤ bound ω)
    (diff : ∀ᵐ ω ∂μ, ∀ z ∈ Metric.ball x ε, HasDerivAt (fun u => q u ω) (dq z ω) z) :
    HasDerivAt (fun z => t*log (∫ ω, exp (q z ω/t) ∂μ))
      ((∫ ω, exp (q x ω/t)*dq x ω ∂μ)/(∫ ω, exp (q x ω/t) ∂μ)) x := by
  have hd : ∀ᵐ ω ∂μ, ∀ z ∈ Metric.ball x ε,
      HasDerivAt (fun u => exp (q u ω/t)) (exp (q z ω/t)*dq z ω/t) z := by
    filter_upwards [diff] with ω h
    intro z hz
    simpa only [mul_div_assoc] using ((h z hz).div_const t).exp
  have integralDeriv := (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    hε meas mass dm major hi hd).2
  have result := (integralDeriv.log positive.ne').const_mul t
  rw [integral_div] at result
  convert result using 1
  field_simp
  ring

/-- Compare the expected sampled penalty slope to b, the population
exponentially weighted slope. X is the exponentiated score and D its score's
directional derivative. The pair (X,D) is drawn independently across proposals.
Each X*(D-b) has mean zero and variance w. -/
theorem expected_slope_bias {n : ℕ} [NeZero n] (X D : Fin n → Ω → ℝ) {a b m v w c : ℝ}
    (hm : 0 < m) (ha : 0 < a) (hc : 0 < c)
    (hind : Pairwise fun i j => IndepFun (fun ω => (X i ω, D i ω)) (fun ω => (X j ω, D j ω)) μ)
    (hX : ∀ i, Memℒp (X i) 2 μ) (he : ∀ i, (∫ ω, X i ω ∂μ) = a)
    (hv : ∀ i, variance (X i) μ = v) (lower : ∀ᵐ ω ∂μ, ∀ i, m ≤ X i ω)
    (hY : ∀ i, Memℒp (fun ω => X i ω*(D i ω-b)) 2 μ)
    (center : ∀ i, (∫ ω, X i ω*(D i ω-b) ∂μ) = 0)
    (hw : ∀ i, variance (fun ω => X i ω*(D i ω-b)) μ = w) :
    |(∫ ω, ∑ i, X i ω/(∑ j, X j ω)*D i ω ∂μ) - b| ≤ (w/c+c*v)/(2*n*m*a) := by
  let Y := fun i ω => X i ω*(D i ω-b)
  let centered := fun p : ℝ × ℝ => p.1*(p.2-b)
  have ic : Pairwise fun i j => IndepFun (Y i) (Y j) μ :=
    fun _ _ hij => (hind hij).comp (φ := centered) (ψ := centered)
      (by dsimp [centered]; fun_prop) (by dsimp [centered]; fun_prop)
  obtain ⟨hZ, eZ, vZ⟩ := sample_moments X a v hX he hv
    (fun _ _ hij => (hind hij).comp measurable_fst measurable_fst)
  obtain ⟨hH, eH, vH⟩ := sample_moments Y 0 w hY center hw ic
  obtain ⟨hr, hb⟩ := ratio_bias hm ha hc hH hZ eH (sample_floor lower)
  have eq : (fun ω => ∑ i, X i ω/(∑ j, X j ω)*D i ω) =ᵐ[μ]
      fun ω => sampleMean Y ω/sampleMean X ω + b := by
    filter_upwards [lower] with ω h
    have pos := sum_pos (s := univ) (fun i _ => hm.trans_le (h i)) univ_nonempty
    exact sub_eq_iff_eq_add.mp (slope_center X D b ω pos.ne')
  rw [integral_congr_ae eq, integral_add hr (integrable_const b)]
  simp only [integral_const, measure_univ, ENNReal.one_toReal, one_smul, add_sub_cancel_right]
  have sh : (∫ ω, (sampleMean Y ω)^2 ∂μ) = w/n := by
    simpa only [hH.variance_eq, Pi.pow_apply, Pi.sub_apply, eH, sub_zero] using vH
  have sz : (∫ ω, (sampleMean X ω-a)^2 ∂μ) = v/n := by
    simpa only [hZ.variance_eq, Pi.pow_apply, Pi.sub_apply, eZ] using vZ
  rw [sh, sz] at hb
  exact hb.trans_eq (by ring)

end
end StateSetFormal.CQLSampling
