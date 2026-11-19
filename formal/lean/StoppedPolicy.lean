import PPOBatch
import RegularizedBatch

namespace StateSetFormal.StoppedPolicy
open Finset Real StateSetFormal.Policy
open scoped BigOperators
noncomputable section

/-- A leaf terminates generation at EOS or horizon. Actions at an internal
prefix have separate behavior and current probabilities and their own subtrees. -/
inductive Tree (k : ℕ) where
  | leaf
  | node (p q : Fin k → ℝ) (next : Fin k → Tree k)

def Valid {k : ℕ} : Tree k → Prop
  | .leaf => True
  | .node p q next => (∑ i, p i) = 1 ∧ (∑ i, q i) = 1 ∧
      ∀ i, 0 ≤ p i ∧ 0 ≤ q i ∧ (0 < p i → 0 < q i ∧ Valid (next i))

/-- Sum of behavior path mass times the log probability ratio at every leaf. -/
def leafKL {k : ℕ} : Tree k → ℝ → ℝ → ℝ
  | .leaf, a, b => a * (log a - log b)
  | .node p q next, a, b => ∑ i, leafKL (next i) (a * p i) (b * q i)

/-- Expected sum of the per-token k3 values accumulated by a stopped rollout. -/
def sampledSum {k : ℕ} : Tree k → ℝ
  | .leaf => 0
  | .node p q next => ∑ i, p i * (k3 (log (q i) - log (p i)) + sampledSum (next i))

/-- Target mass on actions absent from behavior, accumulated only at visited prefixes. -/
def missing {k : ℕ} : Tree k → ℝ
  | .leaf => 0
  | .node p q next => (∑ i, if p i = 0 then q i else 0) + ∑ i, p i * missing (next i)

theorem zero_mass {k : ℕ} (tree : Tree k) (b : ℝ) : leafKL tree 0 b = 0 := by
  induction tree generalizing b with
  | leaf => simp [leafKL]
  | node p q next ih => simp [leafKL, ih]

theorem chain {k : ℕ} (tree : Tree k) (valid : Valid tree) (a b : ℝ)
    (ha : 0 < a) (hb : 0 < b) :
    leafKL tree a b = a * ((log a - log b) + sampledSum tree + missing tree) := by
  induction tree generalizing a b with
  | leaf => simp [leafKL, sampledSum, missing]
  | node p q next ih =>
    obtain ⟨pm, qm, children⟩ := valid
    have hp := fun i => (children i).1
    have hq (i : Fin _) (h : p i ≠ 0) := ((children i).2.2 (lt_of_le_of_ne (hp i) h.symm)).1
    have localKL : (∑ i, p i*(log (p i)-log (q i))) =
        (∑ i, p i*k3 (log (q i)-log (p i))) + ∑ i, if p i=0 then q i else 0 := by
      rw [← RegularizedBatch.sparse_kl p q hp hq pm qm, ← sum_add_distrib]
      apply sum_congr rfl
      intro i _
      by_cases h : p i=0 <;> simp [h]
    rw [leafKL, sampledSum, missing]
    have term (i : Fin k) : leafKL (next i) (a*p i) (b*q i) =
        a*(p i*(log a-log b) + p i*(log (p i)-log (q i)) +
           p i*sampledSum (next i) + p i*missing (next i)) := by
      by_cases h : p i = 0
      · simp [h, zero_mass]
      have pos := lt_of_le_of_ne (hp i) (Ne.symm h)
      rw [ih i ((children i).2.2 pos).2 _ _ (mul_pos ha pos) (mul_pos hb (hq i h)),
        log_mul ha.ne' h, log_mul hb.ne' (hq i h).ne']
      ring
    simp_rw [term, mul_add]
    simp only [sum_add_distrib, ← mul_sum, ← sum_mul, pm, one_mul, localKL]
    ring

theorem stopped_path_kl {k : ℕ} (tree : Tree k) (h : Valid tree) :
    leafKL tree 1 1 = sampledSum tree + missing tree := by
  simpa using chain tree h 1 1 (by norm_num) (by norm_num)

def SharedSupport {k : ℕ} : Tree k → Prop
  | .leaf => True
  | .node p q next => ∀ i, (p i = 0 → q i = 0) ∧ (p i ≠ 0 → SharedSupport (next i))

theorem common_support {k : ℕ} (tree : Tree k) (valid : Valid tree)
    (shared : SharedSupport tree) : leafKL tree 1 1 = sampledSum tree := by
  suffices h : missing tree = 0 by simp [stopped_path_kl tree valid, h]
  clear valid
  induction tree with
  | leaf => rfl
  | node p q next ih =>
    have localMass (i : Fin _) : (if p i=0 then q i else 0) = 0 := by
      split_ifs with h
      · exact (shared i).1 h
      · rfl
    have child (i : Fin _) : p i * missing (next i) = 0 := by
      by_cases h : p i=0
      · simp [h]
      · simp [ih i ((shared i).2 h)]
    simp [missing, localMass, child]

/-- The native near-equality branch has a certified uniform scalar error.
The cutoff can be any nonnegative value at most one. -/
def measured (cut d : ℝ) := if |d| < cut then RegularizedBatch.smallGap d else k3 d

theorem measurement_error (cut d : ℝ) (hc : 0 ≤ cut) (hc' : cut ≤ 1) :
    |measured cut d - k3 d| ≤ cut^7/4410 := by
  unfold measured
  split_ifs with h
  · rw [abs_sub_comm]
    exact (RegularizedBatch.small_gap_error (h.le.trans hc')).trans
      (div_le_div_of_nonneg_right (pow_le_pow_left₀ (abs_nonneg d) h.le _) (by norm_num))
  · simp [show 0 ≤ cut^7 / 4410 by positivity]

theorem measured_nonnegative {cut d : ℝ} (hc : cut ≤ 1) : 0 ≤ measured cut d := by
  unfold measured
  split_ifs with h
  · have lo : -1 ≤ d := (abs_le.mp (h.le.trans hc)).1
    have h₁ : 0 ≤ 360+120*d := by linarith
    have h₂ : 0 ≤ 30+6*d := by linarith
    have id : RegularizedBatch.smallGap d = d^2*(360+120*d+d^2*(30+6*d)+d^4)/720 := by
      unfold RegularizedBatch.smallGap
      ring
    rw [id]
    positivity
  · exact k3_nonnegative _

/-- For any realized nonempty sample, including action-dependent lengths,
the real-arithmetic error of the per-token mean does not grow with length. -/
theorem measured_mean {n : ℕ} (hn : n ≠ 0) (d : Fin n → ℝ) (cut : ℝ)
    (hc : 0 ≤ cut) (hc' : cut ≤ 1) :
    |(∑ i, measured cut (d i))/(n:ℝ) - (∑ i, k3 (d i))/(n:ℝ)| ≤ cut^7/4410 := by
  rw [← sub_div, ← sum_sub_distrib, abs_div, abs_of_nonneg (Nat.cast_nonneg n : (0:ℝ) ≤ n)]
  apply (div_le_iff₀ (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn) : (0:ℝ)<n)).mpr
  calc
    _ ≤ ∑ i, |measured cut (d i)-k3 (d i)| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _ : Fin n, (cut^7/4410) := sum_le_sum fun i _ => measurement_error cut (d i) hc hc'
    _ = _ := by simp [nsmul_eq_mul, mul_comm]

/-- For B positive-length rows with K continuations, averaging over all child
draws divides the expected summed drift K*D by the actual total token count. -/
theorem conditioned_mean {B K n : ℕ} (hB : B ≠ 0) (w : Fin n → ℝ)
    (draw : Fin K → Fin n → ℝ) (D : ℝ) (h : ∀ i, ∑ s, w s*draw i s = D) :
    (∑ s, w s*((∑ i, draw i s)/((B:ℝ)+K))) = D*(K/(B:ℝ))/(1+K/(B:ℝ)) := by
  simp_rw [← mul_div_assoc, mul_sum, ← sum_div]
  rw [sum_comm]
  simp only [h, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have pos : 0 < (B:ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hB)
  field_simp [pos.ne', (by positivity : (B:ℝ)+K ≠ 0)]
  ring

/-- If the root law is unchanged and only the continuation has drift D, a
batch with continuation fraction x has conditional mean token drift Dx/(1+x).
The gap from substituting a fixed mean length has an exact positive remainder. -/
theorem token_mean_gap {n : ℕ} (w x : Fin n → ℝ) (D q : ℝ)
    (hw : ∑ i, w i = 1) (hm : ∑ i, w i*x i = q)
    (hx : ∀ i, 0 ≤ x i) (hq : 0 ≤ q) :
    D*q/(1+q) - (∑ i, w i*(D*x i/(1+x i))) =
      ∑ i, w i*(D*(x i-q)^2/((1+x i)*(1+q)^2)) := by
  have point (i : Fin n) :
      w i*(D*x i/(1+x i)) + w i*(D*(x i-q)^2/((1+x i)*(1+q)^2)) =
      w i*(D*q/(1+q) - D*q/(1+q)^2) + w i*x i*(D/(1+q)^2) := by
    field_simp [(by linarith [hx i] : (1+x i) ≠ 0), (by linarith : (1+q) ≠ 0)]
    ring
  have h := sum_congr (rfl : (univ : Finset (Fin n)) = univ) (fun i _ => point i)
  simp only [sum_add_distrib, ← sum_mul, hw, hm, one_mul] at h
  linear_combination -h

theorem token_mean_bound {n : ℕ} (w x : Fin n → ℝ) (D q : ℝ)
    (hw : ∑ i, w i = 1) (hm : ∑ i, w i*x i = q)
    (hx : ∀ i, 0 ≤ x i) (hq : 0 ≤ q) (hp : ∀ i, 0 ≤ w i) (hd : 0 ≤ D) :
    (∑ i, w i*(D*x i/(1+x i))) ≤ D*q/(1+q) := by
  have h := token_mean_gap w x D q hw hm hx hq
  apply sub_nonneg.mp
  rw [h]
  exact sum_nonneg fun i _ => by have := hx i; have := hp i; positivity

/-- Conditional KL of a fair second action against a policy that assigns
probability exp(-t)/2 to its first outcome. -/
def binaryDrift (t : ℝ) :=
  (log (1/2) - log (exp (-t)/2) + (log (1/2) - log (1-exp (-t)/2)))/2

private def pair (a b : ℝ) (i : Fin 2) : ℝ := if i = 0 then a else b

def hiddenTree (q t : ℝ) : Tree 2 :=
  .node (pair (1-q) q) (pair (1-q) q) (fun i => if i = 0 then .leaf else
    .node (fun _ => 1/2) (pair (exp (-t)/2) (1-exp (-t)/2)) (fun _ => .leaf))

theorem hidden_valid {q t : ℝ} (hq : 0 < q) (hq' : q < 1) (ht : 0 ≤ t) :
    Valid (hiddenTree q t) := by
  have le : exp (-t) ≤ 1 := exp_le_one_iff.mpr (by linarith)
  have tail : 0 < 1-exp (-t)/2 := by linarith
  have head : 0 < exp (-t)/2 := by positivity
  have stop : 0 < 1-q := by linarith
  simp [hiddenTree, Valid, pair, Fin.sum_univ_two, Fin.forall_fin_two, hq.le, hq,
    stop.le, stop, head.le, head, tail.le, tail]

theorem hidden_path_kl {q t : ℝ} (hq : 0 < q) (ht : 0 ≤ t) :
    leafKL (hiddenTree q t) 1 1 = q * binaryDrift t := by
  have tail : 0 < 1-exp (-t)/2 := by
    have h : exp (-t) ≤ 1 := exp_le_one_iff.mpr (by linarith)
    linarith
  simp (disch := positivity) [hiddenTree, leafKL, pair, Fin.sum_univ_two, binaryDrift,
    Real.log_mul, Real.log_div, hq.ne', tail.ne']
  ring

/-- Expected terminal reward under either policy, with its complete generated
actions available to the reward function. -/
def payoff {k : ℕ} (current : Bool) : Tree k → (List (Fin k) → ℝ) → ℝ
  | .leaf, reward => reward []
  | .node p q next, reward =>
      ∑ i, (if current then q i else p i) * payoff current (next i) (fun path => reward (i::path))

def success (path : List (Fin 2)) : ℝ := if path = [1,0] then 1 else 0

theorem hidden_return_gap (q t : ℝ) :
    payoff false (hiddenTree q t) success - payoff true (hiddenTree q t) success =
      q*(1-exp (-t))/2 := by
  simp [payoff, hiddenTree, success, pair, Fin.sum_univ_two]
  ring

theorem rare_drift (t : ℝ) (ht : 0 ≤ t) : (t-log 2)/2 ≤ binaryDrift t := by
  have he : exp (-t) ≤ 1 := exp_le_one_iff.mpr (by linarith)
  have tail : log (1-exp (-t)/2) ≤ 0 :=
    log_nonpos (by linarith) (by linarith [exp_pos (-t)])
  unfold binaryDrift
  rw [log_div (exp_pos _).ne' (by norm_num), log_exp,
    log_div (by norm_num : (1:ℝ) ≠ 0) (by norm_num), log_one]
  linarith

/-- Every finite independent batch can see only EOS while the true path
KL exceeds any chosen cap. q is the shared continuation probability. -/
theorem missed_drift (batch : ℕ) (cap q : ℝ) (hc : 0 ≤ cap) (hq : 0 < q) (hq' : q < 1) :
    0 < (1-q)^batch ∧ (∑ _ : Fin batch, k3 (log (1-q)-log (1-q))) / batch ≤ cap ∧
      ∃ t : ℝ, Valid (hiddenTree q t) ∧ cap < leafKL (hiddenTree q t) 1 1 ∧
        q/4 ≤ payoff false (hiddenTree q t) success - payoff true (hiddenTree q t) success := by
  constructor
  · exact pow_pos (by linarith) _
  constructor
  · simpa [k3] using hc
  · let t := max (log 2) (2*(cap+1)/q + log 2)
    have hl : log 2 ≤ t := le_max_left ..
    have ht := (log_nonneg (by norm_num : (1:ℝ) ≤ 2)).trans hl
    have ht' : 2*(cap+1)/q + log 2 ≤ t := le_max_right ..
    refine ⟨t, hidden_valid hq hq' ht, ?_, ?_⟩
    · rw [hidden_path_kl hq ht]
      apply lt_of_lt_of_le _ (mul_le_mul_of_nonneg_left (rare_drift t ht) hq.le)
      have h := (div_le_iff₀ hq).mp (show 2*(cap+1)/q ≤ t-log 2 by linarith)
      nlinarith
    · rw [hidden_return_gap]
      have e := exp_le_exp.mpr (neg_le_neg hl)
      rw [exp_neg (log 2), exp_log (by norm_num)] at e
      nlinarith [mul_le_mul_of_nonneg_left e hq.le]

end
end StateSetFormal.StoppedPolicy
