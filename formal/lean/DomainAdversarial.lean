import Preference

/-! Balanced domain classification and the signed feature gradient used by DANN. -/
namespace StateSetFormal.DomainAdversarial
open Real Finset
open scoped BigOperators
noncomputable section

def loss (realDomain : Bool) (z : ℝ) :=
  Policy.preferenceLoss 1 (if realDomain then z else -z)

def residual (realDomain : Bool) (z : ℝ) :=
  if realDomain then -1/(1+exp z) else 1/(1+exp (-z))

theorem logit_gradient (label : Bool) {z : ℝ → ℝ} {dz x : ℝ} (h : HasDerivAt z dz x) :
    HasDerivAt (fun y => loss label (z y)) (residual label (z x)*dz) x := by
  cases label
  · have he := (h.exp.const_add 1).log (by positivity : 1+exp (z x) ≠ 0)
    convert he using 1
    · ext
      simp [loss, Policy.preferenceLoss]
    · simp [residual, exp_neg]
      field_simp
      ring
      simp
  · have he := (h.neg.exp.const_add 1).log (by positivity : 1+exp (-z x) ≠ 0)
    convert he using 1
    · ext
      simp [loss, Policy.preferenceLoss]
    · simp [residual, exp_neg]
      field_simp
      ring

def batch {m n : ℕ} (s : Fin m → ℝ) (r : Fin n → ℝ) :=
  (∑ i, loss false (s i))/m + (∑ i, loss true (r i))/n

def slope {m n : ℕ} (s : Fin m → ℝ) (r : Fin n → ℝ) (ds : Fin m → ℝ) (dr : Fin n → ℝ) :=
  (∑ i, residual false (s i)*ds i)/m + (∑ i, residual true (r i)*dr i)/n

theorem batch_gradient {m n : ℕ} (s : Fin m → ℝ → ℝ) (r : Fin n → ℝ → ℝ)
    (ds : Fin m → ℝ) (dr : Fin n → ℝ) (x : ℝ)
    (hs : ∀ i, HasDerivAt (s i) (ds i) x) (hr : ∀ i, HasDerivAt (r i) (dr i) x) :
    HasDerivAt (fun y => batch (fun i => s i y) (fun i => r i y))
      (slope (fun i => s i x) (fun i => r i x) ds dr) x :=
  ((HasDerivAt.sum fun i _ => logit_gradient false (hs i)).div_const _).add
    ((HasDerivAt.sum fun i _ => logit_gradient true (hr i)).div_const _)

/-- The encoder's gradient is that of task minus strength times the domain
loss in its parameter. The critic receives the ordinary domain gradient. -/
theorem encoder_gradient {task domain : ℝ → ℝ} {dt dd x : ℝ} (strength : ℝ)
    (ht : HasDerivAt task dt x) (hd : HasDerivAt domain dd x) :
    HasDerivAt (fun y => task y - strength*domain y) (dt-strength*dd) x :=
  ht.sub (hd.const_mul strength)

def cell (s r p : ℝ) := -s*log (1-p) - r*log p
def posterior (s r : ℝ) := r/(s+r)
def best (s r : ℝ) := cell s r (posterior s r)

/-- Includes source-only and target-only feature values. A pure-domain
optimum is at the boundary and every finite logit gives strictly higher loss. -/
theorem strict_bayes {s r p : ℝ} (hs : 0 ≤ s) (hr : 0 ≤ r) (hm : 0 < s+r)
    (hp : 0 < p) (hp1 : p < 1) (hneq : p ≠ posterior s r) :
    best s r < cell s r p := by
  rcases eq_or_lt_of_le hs with hzero | hsp
  · subst s
    simp [best, cell, posterior, show r ≠ 0 by linarith]
    nlinarith [mul_neg_of_pos_of_neg (show 0 < r by linarith) (log_neg hp hp1)]
  rcases eq_or_lt_of_le hr with hzero | hrp
  · subst r
    simp [best, cell, posterior]
    nlinarith [mul_neg_of_pos_of_neg hsp (log_neg (by linarith : 0 < 1-p) (by linarith))]
  let q := posterior s r
  have hq : 0 < q := div_pos hrp hm
  have hq1 : q < 1 := (div_lt_one hm).mpr (by linarith)
  have eqr : (s+r)*q = r := by dsimp [q, posterior]; field_simp
  have eqs : (s+r)*(1-q) = s := by nlinarith
  have h := mul_lt_mul_of_pos_left (Preference.strict_binary_optimum hq hq1 hp hp1 hneq) hm
  change cell s r q < _
  unfold cell
  linear_combination h + (log q - log p)*eqr + (log (1-q) - log (1-p))*eqs

theorem bayes_le {s r p : ℝ} (hs : 0 ≤ s) (hr : 0 ≤ r)
    (hp : 0 < p) (hp1 : p < 1) : best s r ≤ cell s r p := by
  by_cases hm : 0 < s+r
  · by_cases he : p = posterior s r
    · simp [he, best]
    · exact (strict_bayes hs hr hm hp hp1 he).le
  · have hz : s=0 := by linarith
    have hz' : r=0 := by linarith
    simp [hz,hz',cell,best]

/-- With equal domain priors, an optimal discriminator has chance-level
loss exactly when each feature value has equal mass in the two domains. -/
theorem indistinguishable {n : ℕ} (s r : Fin n → ℝ)
    (hs : ∀ i, 0 ≤ s i) (hr : ∀ i, 0 ≤ r i) :
    (∑ i, best (s i) (r i)) = (∑ i, cell (s i) (r i) (1/2)) ↔ s=r := by
  have le (i) := bayes_le (hs i) (hr i) (p := 1/2) (by norm_num) (by norm_num)
  constructor
  · intro equal
    funext i
    by_contra ne
    have hm : 0 < s i+r i := by
      have a := hs i
      have b := hr i
      by_contra! hn
      exact ne (by linarith)
    have diff : 1/2 ≠ posterior (s i) (r i) := by
      unfold posterior
      intro he
      have eq := (eq_div_iff hm.ne').mp he
      exact ne (by linarith)
    exact (ne_of_lt (sum_lt_sum (fun j _ => le j)
      ⟨i, mem_univ _, strict_bayes (hs i) (hr i) hm (by norm_num) (by norm_num) diff⟩)) equal
  · rintro rfl
    apply sum_congr rfl
    intro i _
    by_cases hz : s i=0
    · simp [hz, best, cell]
    · have eq : posterior (s i) (s i)=1/2 := by
        unfold posterior
        field_simp [show s i+s i ≠ 0 by have := hs i; positivity]
        ring
      simp [best,eq]

/-- Stable logistic losses in the trainer are Bernoulli log losses at the
critic's predicted probability of a real feature. -/
theorem probability_loss (label : Bool) (z : ℝ) :
    loss label z = -log (if label then Preference.win z else 1-Preference.win z) := by
  cases label
  · change Policy.preferenceLoss 1 (-z) = -log (1-Preference.win z)
    rw [Preference.win_complement]
    simp [Policy.preferenceLoss, Preference.win]
  · simp [loss, Policy.preferenceLoss, Preference.win]

def frequency {m k : ℕ} (bins : Fin m → Fin k) (j : Fin k) :=
  (∑ i, if bins i = j then (1:ℝ) else 0)/m

theorem regroup {m k : ℕ} (bins : Fin m → Fin k) (f : Fin k → ℝ) :
    (∑ i, f (bins i))/m = ∑ j, frequency bins j * f j := by
  unfold frequency
  simp_rw [div_mul_eq_mul_div, sum_mul]
  rw [← sum_div, sum_comm]
  simp

/-- The batch uses one empirical distribution per domain, even if one
domain supplies many more examples or several rows share a feature. -/
theorem batch_histogram {m n k : ℕ} (s : Fin m → Fin k) (r : Fin n → Fin k)
    (z : Fin k → ℝ) :
    batch (fun i => z (s i)) (fun i => z (r i)) =
      ∑ j, cell (frequency s j) (frequency r j) (Preference.win (z j)) := by
  unfold batch
  rw [regroup s (fun j => loss false (z j)), regroup r (fun j => loss true (z j)),
      ← sum_add_distrib]
  apply sum_congr rfl
  intro j _
  simp [cell, probability_loss]
  ring

theorem frequency_normalized {m k : ℕ} [NeZero m] (bins : Fin m → Fin k) :
    ∑ j, frequency bins j = 1 := by
  simpa [NeZero.ne m] using (regroup bins (fun _ => (1:ℝ))).symm

private theorem weighted_log (s m : ℝ) (hm : m ≠ 0) :
    s * log (s/m) = s*(log s-log m) := by
  by_cases hs : s=0
  · simp [hs]
  · rw [log_div hs hm]

theorem optimal_cell_value {s r : ℝ} (hs : 0 ≤ s) (hr : 0 ≤ r) :
    best s r = (s+r)*log 2 -
      (s*(log s-log ((s+r)/2)) + r*(log r-log ((s+r)/2))) := by
  by_cases hm : s+r=0
  · have hz : s=0 := by linarith
    have hz' : r=0 := by linarith
    simp [hz, hz', best, cell]
  have he : 1-r/(s+r)=s/(s+r) := by field_simp
  simp only [best, cell, posterior, he, neg_mul, weighted_log _ _ hm]
  rw [log_div hm (by norm_num : (2:ℝ) ≠ 0)]
  ring

def js {n : ℕ} (s r : Fin n → ℝ) :=
  (∑ i, (s i*(log (s i)-log ((s i+r i)/2)) + r i*(log (r i)-log ((s i+r i)/2))))/2

/-- For normalized finite source and target masses, the Bayes discriminator
loss (sum of domain means) is log 4 minus twice their Jensen-Shannon divergence.
Zero-mass cells contribute zero, including features seen in only one domain. -/
theorem optimal_risk_js {n : ℕ} (s r : Fin n → ℝ) (hs : ∀ i, 0 ≤ s i) (hr : ∀ i, 0 ≤ r i)
    (sm : ∑ i, s i = 1) (rm : ∑ i, r i = 1) :
    (∑ i, best (s i) (r i)) = 2*log 2 - 2*js s r := by
  unfold js
  simp_rw [optimal_cell_value (hs _) (hr _)]
  rw [sum_sub_distrib, ← sum_mul, sum_add_distrib, sm, rm]
  ring

theorem bayes_logit {s r : ℝ} (hs : 0 < s) (hr : 0 < r) :
    Preference.win (log (r/s)) = posterior s r := by
  unfold Preference.win posterior
  rw [exp_neg, exp_log (div_pos hr hs)]
  field_simp [hs.ne', hr.ne']
  ring

theorem chance {n : ℕ} (s r : Fin n → ℝ)
    (sm : ∑ i, s i = 1) (rm : ∑ i, r i = 1) :
    (∑ i, cell (s i) (r i) (1/2)) = 2*log 2 := by
  have half : (1:ℝ)-1/2 = 1/2 := by norm_num
  calc
    _ = ∑ i, (s i + r i)*log 2 := by
          apply sum_congr rfl
          intro i _
          unfold cell
          rw [half]
          simp only [one_div, log_inv]
          ring
    _ = _ := by rw [← sum_mul, sum_add_distrib, sm, rm]; ring

theorem js_bounds {n : ℕ} (s r : Fin n → ℝ) (hs : ∀ i, 0 ≤ s i) (hr : ∀ i, 0 ≤ r i)
    (sm : ∑ i, s i = 1) (rm : ∑ i, r i = 1) : 0 ≤ js s r ∧ js s r ≤ log 2 := by
  have bound := sum_le_sum (s := univ) (fun i _ => bayes_le (hs i) (hr i)
    (p := 1/2) (by norm_num) (by norm_num))
  rw [chance s r sm rm, optimal_risk_js s r hs hr sm rm] at bound
  have nonneg : 0 ≤ ∑ i, best (s i) (r i) := by
    apply sum_nonneg
    intro i _
    have pr : 0 ≤ posterior (s i) (r i) := div_nonneg (hr i) (add_nonneg (hs i) (hr i))
    have pr1 : posterior (s i) (r i) ≤ 1 := by
      by_cases h : 0 < s i+r i
      · exact (div_le_one h).mpr (by linarith [hs i])
      · have hz : s i+r i=0 := by linarith [hs i, hr i]
        simp [posterior, hz]
    have a := mul_nonpos_of_nonneg_of_nonpos (hs i)
      (log_nonpos (sub_nonneg.mpr pr1) (by linarith : 1-posterior (s i) (r i) ≤ 1))
    have b := mul_nonpos_of_nonneg_of_nonpos (hr i) (log_nonpos pr pr1)
    unfold best cell
    linarith
  rw [optimal_risk_js s r hs hr sm rm] at nonneg
  constructor <;> linarith

theorem js_zero_iff {n : ℕ} (s r : Fin n → ℝ) (hs : ∀ i, 0 ≤ s i) (hr : ∀ i, 0 ≤ r i)
    (sm : ∑ i, s i = 1) (rm : ∑ i, r i = 1) : js s r = 0 ↔ s=r := by
  rw [← indistinguishable s r hs hr, chance s r sm rm, optimal_risk_js s r hs hr sm rm]
  constructor <;> intro h <;> linarith

/-- Per-cell rearrangement used to evaluate nearby frequencies without
subtracting two first-order contributions. atanh(u) is the half log odds. -/
theorem balanced_cell {m u : ℝ} (hm : 0 < m) (hl : -1 < u) (hu : u < 1) :
    (m*(1+u)*(log (m*(1+u))-log m) + m*(1-u)*(log (m*(1-u))-log m))/2 =
      m*(u*((log (1+u)-log (1-u))/2)+log (1-u^2)/2) := by
  have a : 1+u ≠ 0 := by linarith
  have b : 1-u ≠ 0 := by linarith
  rw [log_mul hm.ne' a, log_mul hm.ne' b, show 1-u^2=(1+u)*(1-u) by ring, log_mul a b]
  ring

theorem kl_linear_correction {n : ℕ} (p q : Fin n → ℝ) (mass : ∑ i, p i = ∑ i, q i) :
    (∑ i, (p i*(log (p i)-log (q i))-(p i-q i))) =
      ∑ i, p i*(log (p i)-log (q i)) := by
  rw [sum_sub_distrib, sum_sub_distrib, mass, sub_self, sub_zero]

def klPolynomial (u : ℝ) := u^2*(1/2+u*(-1/6+u*(1/12+u*(-1/20+u/30))))

theorem kl_polynomial_error {u : ℝ} (h : |u| ≤ 1/1000) :
    |(1+u)*log (1+u)-u-klPolynomial u| ≤ 2*|u|^7 := by
  let l := u-u^2/2+u^3/3-u^4/4+u^5/5-u^6/6
  have half : |u| < 1 := by linarith
  have logbound : |log (1+u)-l| ≤ |u|^7/(1-|u|) := by
    have hb := Real.abs_log_sub_add_sum_range_le (x := -u) (by simpa) 6
    norm_num [sum_range_succ, abs_neg] at hb
    convert hb using 1
    congr 1
    dsimp [l]
    ring
  have small : 0 < 1+u := by linarith [(abs_le.mp h).1]
  have hb := mul_le_mul_of_nonneg_left logbound small.le
  have fact : (1+u)/(1-|u|) ≤ 3/2 := by
    apply (div_le_iff₀ (by linarith : 0 < 1-|u|)).mpr
    linarith [le_abs_self u]
  have hf := mul_le_mul_of_nonneg_right fact (pow_nonneg (abs_nonneg u) 7)
  have alg : (1+u)*log (1+u)-u-klPolynomial u = (1+u)*(log (1+u)-l)-u^7/6 := by
    dsimp [klPolynomial, l]
    ring
  rw [alg]
  calc
    _ ≤ |(1+u)*(log (1+u)-l)| + |u^7/6| := abs_sub _ _
    _ ≤ 2*|u|^7 := by
      rw [abs_mul, abs_of_pos small, abs_div, abs_pow, abs_of_pos (by norm_num : (0:ℝ)<6)]
      have hc : (1+u)*(|u|^7/(1-|u|)) = ((1+u)/(1-|u|))*|u|^7 := by ring
      rw [hc] at hb
      linarith [pow_nonneg (abs_nonneg u) 7]

end
end StateSetFormal.DomainAdversarial
