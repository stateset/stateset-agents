import ScoreFunction

namespace StateSetFormal.Preference
open Finset Real Policy
open scoped BigOperators
noncomputable section

/-- On a fixed preference draw, the frozen reference affects the scalar weight
of the update. Only the two live sequence scores contribute parameter slopes. -/
theorem pair_gradient (c r : ℝ → ℝ) (dc dr cr rr beta x : ℝ)
    (hc : HasDerivAt c dc x) (hr : HasDerivAt r dr x) :
    HasDerivAt (fun y => preferenceLoss beta ((c y-r y)-(cr-rr)))
      ((-beta/(1+exp (beta*((c x-r x)-(cr-rr)))))*(dc-dr)) x := by
  have hd := ((hc.sub hr).sub_const (cr-rr)).const_mul (-beta)
  have he := hd.exp.const_add 1
  have hz : 1 + exp (-beta*((c x-r x)-(cr-rr))) ≠ 0 := by positivity
  unfold preferenceLoss
  convert he.log hz using 1
  rw [neg_mul, exp_neg]
  field_simp
  ring

/-- Stable forward formula used in the ATen log-sigmoid decomposition. -/
theorem stable_loss (x : ℝ) : -(min 0 x - log (1+exp (-|x|))) = preferenceLoss 1 x := by
  by_cases hx : 0 ≤ x
  · simp [abs_of_nonneg hx, min_eq_left hx, preferenceLoss]
  · have hx' := le_of_not_ge hx
    have h : 1+exp (-x) = (1+exp x)/exp x := by rw [exp_neg]; field_simp; ring
    simp only [preferenceLoss, neg_one_mul, abs_of_nonpos hx', neg_neg, min_eq_right hx']
    rw [h, log_div (by positivity) (exp_ne_zero _), log_exp]
    ring

/-- The two stable backward branches agree with the negative logistic weight,
including at a tie. -/
theorem stable_slope (x : ℝ) :
    -1*((if x < 0 then 1 else 0) -
      (if x < 0 then 1 else -1) * (exp (-|x|)/(exp (-|x|)+1))) =
      -1/(1+exp x) := by
  by_cases h : x < 0
  · simp [h, abs_of_neg h]
    field_simp
    ring
  · simp [h, abs_of_nonneg (le_of_not_gt h), exp_neg]
    field_simp

def win (x : ℝ) := 1/(1+exp (-x))

theorem win_complement (x : ℝ) : 1-win x = win (-x) := by
  unfold win
  rw [neg_neg, exp_neg]
  field_simp
  ring

theorem pair_risk (q beta m : ℝ) :
    q*preferenceLoss beta m + (1-q)*preferenceLoss beta (-m) =
      -q*log (win (beta*m)) - (1-q)*log (1-win (beta*m)) := by
  rw [win_complement]
  simp [preferenceLoss, win, neg_mul]

/-- For an interior population preference frequency the Bernoulli risk has
a unique minimizing predicted probability. -/
theorem strict_binary_optimum {q p : ℝ} (hq : 0 < q) (hq' : q < 1)
    (hp : 0 < p) (hp' : p < 1) (different : p ≠ q) :
    -q*log q - (1-q)*log (1-q) < -q*log p - (1-q)*log (1-p) := by
  have hq1 := sub_pos.mpr hq'
  have hp1 := sub_pos.mpr hp'
  have hc := log_lt_sub_one_of_pos (div_pos hp hq) (by
    intro h
    apply different
    simpa using (div_eq_iff hq.ne').mp h)
  have hr := log_le_sub_one_of_pos (div_pos hp1 hq1)
  rw [log_div hp.ne' hq.ne'] at hc
  rw [log_div hp1.ne' hq1.ne'] at hr
  have c := mul_lt_mul_of_pos_left hc hq
  have r := mul_le_mul_of_nonneg_left hr hq1.le
  rw [show q*(p/q-1) = p-q by field_simp] at c
  rw [show (1-q)*((1-p)/(1-q)-1) = q-p by field_simp] at r
  linarith

theorem win_injective : Function.Injective win := by
  intro x y eq
  unfold win at eq
  simp only [one_div, inv_inj, add_right_inj, exp_eq_exp, neg_inj] at eq
  exact eq

/-- A freely chosen positive-temperature margin recovers the population
log odds divided by beta, uniquely for interior preference frequencies. -/
theorem calibrated_margin {q beta m : ℝ} (hq : 0 < q) (hq' : q < 1) (hb : 0 < beta)
    (different : m ≠ log (q/(1-q))/beta) :
    let best := log (q/(1-q))/beta
    q*preferenceLoss beta best + (1-q)*preferenceLoss beta (-best) <
      q*preferenceLoss beta m + (1-q)*preferenceLoss beta (-m) := by
  have scaled : beta*(log (q/(1-q))/beta) = log (q/(1-q)) := by field_simp
  have recover : win (log (q/(1-q))) = q := by
    rw [win, exp_neg, exp_log (div_pos hq (sub_pos.mpr hq'))]
    field_simp
  have sep : win (beta*m) ≠ q := by
    intro h
    rw [← recover] at h
    exact different ((eq_div_iff hb.ne').mpr (by linarith [win_injective h]))
  have hp : 0 < win (beta*m) := by unfold win; positivity
  have hp' : win (beta*m) < 1 := by unfold win; apply (div_lt_one (by positivity)).mpr; linarith [exp_pos (-(beta*m))]
  dsimp only
  rw [pair_risk, pair_risk, scaled, recover]
  exact strict_binary_optimum hq hq' hp hp' sep

/-- At the first differing action the normalizer and the already-scored
common response prefix cancel for each model, including when slopes are taken. -/
theorem fork_margin (common norm c r ct rt : ℝ) :
    (common+(c-norm)+ct)-(common+(r-norm)+rt) = (c-r)+(ct-rt) := by ring

def mass {n : ℕ} (ref reward : Fin n → ℝ) (beta : ℝ) (i : Fin n) :=
  ref i * exp (reward i / beta)

def gibbs {n : ℕ} (ref reward : Fin n → ℝ) (beta : ℝ) (i : Fin n) :=
  mass ref reward beta i / ∑ j, mass ref reward beta j

private theorem partition_pos {n : ℕ} [NeZero n] {ref : Fin n → ℝ}
    (hr : ∀ i, 0 < ref i) (reward : Fin n → ℝ) (beta : ℝ) :
    0 < ∑ j, mass ref reward beta j :=
  sum_pos (fun j _ => mul_pos (hr j) (exp_pos _)) univ_nonempty

theorem gibbs_mass {n : ℕ} [NeZero n] {ref : Fin n → ℝ} (hr : ∀ i, 0 < ref i)
    (reward : Fin n → ℝ) (beta : ℝ) : ∑ i, gibbs ref reward beta i = 1 := by
  simp [gibbs, ← sum_div, (partition_pos hr reward beta).ne']

theorem log_ratio {n : ℕ} [NeZero n] {ref : Fin n → ℝ} (hr : ∀ i, 0 < ref i)
    (reward : Fin n → ℝ) (beta : ℝ) (i : Fin n) :
    log (gibbs ref reward beta i) - log (ref i) =
      reward i / beta - log (∑ j, mass ref reward beta j) := by
  have hi : 0 < mass ref reward beta i := mul_pos (hr i) (exp_pos _)
  rw [gibbs, log_div hi.ne' (partition_pos hr reward beta).ne',
      mass, log_mul (hr i).ne' (exp_ne_zero _), log_exp]
  ring

/-- The finite Gibbs policy recovers every pairwise reward gap from its
reference-adjusted sequence log likelihoods at a common prompt. -/
theorem reward_gap {n : ℕ} [NeZero n] {ref : Fin n → ℝ} (hr : ∀ i, 0 < ref i)
    (reward : Fin n → ℝ) {beta : ℝ} (hb : 0 < beta) (c r : Fin n) :
    beta*((log (gibbs ref reward beta c)-log (ref c))-
          (log (gibbs ref reward beta r)-log (ref r))) = reward c-reward r := by
  rw [log_ratio hr, log_ratio hr]
  field_simp

def objective {n : ℕ} (p ref reward : Fin n → ℝ) (beta : ℝ) :=
  ∑ i, p i * (reward i - beta*(log (p i)-log (ref i)))

/-- When the observed comparisons connect the finite response set, fitting
their reward gaps identifies the entire normalized policy at this prompt. -/
theorem connected_fit {n : ℕ} [NeZero n] {p ref : Fin n → ℝ}
    (hp : ∀ i, 0 < p i) (pm : ∑ i, p i = 1) (hr : ∀ i, 0 < ref i)
    (reward : Fin n → ℝ) {beta : ℝ} (hb : 0 < beta) (pairs : Fin n → Fin n → Prop)
    (connected : ∀ i j, Relation.EqvGen pairs i j)
    (fit : ∀ i j, pairs i j → beta*((log (p i)-log (ref i))-
      (log (p j)-log (ref j))) = reward i-reward j) :
    p = gibbs ref reward beta := by
  let g := gibbs ref reward beta
  let d := fun i => log (p i) - log (g i)
  have edge (i j) (h : pairs i j) : d i = d j := by
    have eq := mul_left_cancel₀ hb.ne' ((fit i j h).trans (reward_gap hr reward hb i j).symm)
    dsimp [d, g]
    linarith
  have along (i j) (h : Relation.EqvGen pairs i j) : d i = d j := by
    induction h with
    | rel i j h => exact edge i j h
    | refl => rfl
    | symm _ _ _ ih => exact ih.symm
    | trans _ _ _ _ _ ih ih' => exact ih.trans ih'
  let k : Fin n := 0
  have ratio (i) : p i = exp (d k) * g i := by
    have hg : 0 < g i := div_pos (mul_pos (hr i) (exp_pos _)) (partition_pos hr reward beta)
    have eq := congrArg exp (along i k (connected i k))
    change exp (log (p i)-log (g i)) = exp (d k) at eq
    rw [exp_sub, exp_log (hp i), exp_log hg] at eq
    exact (div_eq_iff hg.ne').mp eq
  have total := Finset.sum_congr (rfl : (univ : Finset (Fin n)) = univ) (fun i _ => ratio i)
  have gm : ∑ i, g i = 1 := gibbs_mass hr reward beta
  rw [pm, ← mul_sum, gm, mul_one] at total
  funext i
  simpa only [← total, one_mul] using ratio i

def scaledMargin {n : ℕ} (p ref : Fin n → ℝ) (beta : ℝ) (e : Fin n × Fin n) :=
  beta*((log (p e.1)-log (ref e.1))-(log (p e.2)-log (ref e.2)))

/-- The expected loss on a pair with true Bradley–Terry log odds t when
the policy's reference-adjusted log odds are x. -/
def comparisonRisk (t x : ℝ) :=
  win t * preferenceLoss 1 x + (1 - win t) * preferenceLoss 1 (-x)

private theorem risk_strict (t x : ℝ) (hne : x ≠ t) :
    comparisonRisk t t < comparisonRisk t x := by
  have interior (y : ℝ) : 0 < win y ∧ win y < 1 := by
    unfold win
    constructor
    · positivity
    · apply (div_lt_one (by positivity)).mpr
      linarith [exp_pos (-y)]
  unfold comparisonRisk
  rw [pair_risk, pair_risk, one_mul, one_mul]
  exact strict_binary_optimum (interior t).1 (interior t).2
    (interior x).1 (interior x).2 (fun h => hne (win_injective h))

def populationLoss {n : ℕ} (p ref reward : Fin n → ℝ) (beta : ℝ)
    (w : Fin n × Fin n → ℝ) :=
  ∑ e, w e * comparisonRisk (reward e.1 - reward e.2) (scaledMargin p ref beta e)

/-- With reward-calibrated preferences and connected positive-weight comparisons,
the Gibbs policy is the unique population DPO optimum among positive policies. -/
theorem population_optimum {n : ℕ} [NeZero n] {p ref : Fin n → ℝ}
    (hp : ∀ i, 0 < p i) (pm : ∑ i, p i = 1) (hr : ∀ i, 0 < ref i)
    (reward : Fin n → ℝ) {beta : ℝ} (hb : 0 < beta)
    (w : Fin n × Fin n → ℝ) (hw : ∀ e, 0 ≤ w e)
    (connected : ∀ i j, Relation.EqvGen (fun i j => 0 < w (i,j)) i j)
    (different : p ≠ gibbs ref reward beta) :
    populationLoss (gibbs ref reward beta) ref reward beta w <
      populationLoss p ref reward beta w := by
  let gap := fun e : Fin n × Fin n => reward e.1 - reward e.2
  have best (e) : scaledMargin (gibbs ref reward beta) ref beta e = gap e :=
    reward_gap hr reward hb e.1 e.2
  have witness : ∃ e, 0 < w e ∧ scaledMargin p ref beta e ≠ gap e := by
    by_contra! allfit
    apply different
    exact connected_fit hp pm hr reward hb _ connected (fun i j h => allfit (i,j) h)
  unfold populationLoss
  apply Finset.sum_lt_sum
  · intro e _
    rw [best]
    apply mul_le_mul_of_nonneg_left _ (hw e)
    by_cases same : scaledMargin p ref beta e = gap e
    · rw [same]
    · exact (risk_strict _ _ same).le
  · obtain ⟨e, pos, miss⟩ := witness
    exact ⟨e, mem_univ _, best e ▸ mul_lt_mul_of_pos_left (risk_strict _ _ miss) pos⟩

/-- The loss in KL-regularized return from any positive normalized policy
is exactly beta times its forward KL to the Gibbs policy. -/
theorem optimality_gap {n : ℕ} [NeZero n] {p ref : Fin n → ℝ} (hr : ∀ i, 0 < ref i)
    (pm : ∑ i, p i = 1) (reward : Fin n → ℝ) {beta : ℝ} (hb : 0 < beta) :
    beta*log (∑ j, mass ref reward beta j) - objective p ref reward beta =
      beta*∑ i, p i * (log (p i) - log (gibbs ref reward beta i)) := by
  let z := log (∑ j, mass ref reward beta j)
  have term (i : Fin n) : p i * (reward i-beta*(log (p i)-log (ref i))) +
      beta*(p i*(log (p i)-log (gibbs ref reward beta i))) = p i*(beta*z) := by
    have l := log_ratio hr reward beta i
    change _ = _ - z at l
    rw [show log (gibbs ref reward beta i) = reward i/beta-z+log (ref i) by linarith]
    field_simp
    ring
  have h := Finset.sum_congr (rfl : (univ : Finset (Fin n)) = univ) (fun i _ => term i)
  rw [sum_add_distrib, ← mul_sum, ← sum_mul, pm, one_mul] at h
  change beta*z - _ = _
  unfold objective
  linarith

theorem gibbs_optimal {n : ℕ} [NeZero n] {p ref : Fin n → ℝ} (hr : ∀ i, 0 < ref i)
    (hp : ∀ i, 0 < p i) (pm : ∑ i, p i = 1) (reward : Fin n → ℝ)
    {beta : ℝ} (hb : 0 < beta) :
    objective p ref reward beta ≤ objective (gibbs ref reward beta) ref reward beta := by
  have hg (i) : 0 < gibbs ref reward beta i :=
    div_pos (mul_pos (hr i) (exp_pos _)) (partition_pos hr reward beta)
  have gm := gibbs_mass hr reward beta
  have optimum := optimality_gap hr gm reward hb
  simp at optimum
  have other := optimality_gap hr pm reward hb
  rw [← k3_on_policy p _ hp hg pm gm] at other
  have nonneg := mul_nonneg hb.le (sum_nonneg (fun i (_ : i ∈ (univ : Finset (Fin n))) =>
    mul_nonneg (hp i).le (k3_nonnegative (log (gibbs ref reward beta i) - log (p i)))))
  linarith

end
end StateSetFormal.Preference
