import BayesianReward
import Returns
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Topology.MetricSpace.Pseudo.Pi
import Mathlib.Tactic.Positivity

/-! Empirical tabular policy evaluation at assistant decision boundaries. Each
    atom is a logged reward, a successor key and an environment terminal flag. -/
namespace StateSetFormal.LoggedBellman
open Finset Function Filter
open scoped BigOperators Topology NNReal
noncomputable section

structure Row (n : ℕ) where
  reward : ℝ
  next : Fin n
  terminal : Bool

def operator {n k : ℕ} (g : ℝ) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (v : Fin n → ℝ) (s : Fin n) : ℝ :=
  ∑ i, w s i * Returns.backup g (rows s i).reward (v (rows s i).next) (rows s i).terminal

def liveMass {n k : ℕ} (rows : Fin n → Fin k → Row n) (w : Fin n → Fin k → ℝ)
    (s : Fin n) := ∑ i, if (rows s i).terminal then 0 else w s i

/-- Only continuing probability mass can carry successor error. This also
    treats a time cutoff with a supplied continuation as a live row. -/
theorem sensitivity {n k : ℕ} (g : ℝ) (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (v u : Fin n → ℝ) (s : Fin n) :
    |operator g rows w v s-operator g rows w u s| ≤ g*liveMass rows w s*dist v u := by
  have row (i : Fin k) :
      |w s i * Returns.backup g (rows s i).reward (v (rows s i).next) (rows s i).terminal -
       w s i * Returns.backup g (rows s i).reward (u (rows s i).next) (rows s i).terminal| ≤
      (if (rows s i).terminal then 0 else w s i)*(g*dist v u) := by
    have h := dist_le_pi_dist v u (rows s i).next
    rw [Real.dist_eq] at h
    cases done : (rows s i).terminal
    · simp only [Returns.backup, done, Bool.false_eq_true, if_false, ← mul_sub, add_sub_add_left_eq_sub,
        abs_mul, abs_of_nonneg (hw s i), abs_of_nonneg hg]
      exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left h hg) (hw s i)
    · simp [Returns.backup, done]
  unfold operator
  rw [← sum_sub_distrib]
  refine (abs_sum_le_sum_abs _ _).trans ((sum_le_sum fun i _ => row i).trans_eq ?_)
  rw [← sum_mul, liveMass]
  ring

/-- A bound below one on gamma times each state's continuing mass suffices,
    including gamma=1 when every state has a positive probability of stopping. -/
theorem contraction {n k : ℕ} {g : ℝ} (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (b : ℝ≥0) (hb : b < 1) (hm : ∀ s, g*liveMass rows w s ≤ b) :
    ContractingWith b (operator g rows w) := by
  refine ⟨hb, LipschitzWith.of_dist_le_mul fun v u => ?_⟩
  apply (dist_pi_le_iff (mul_nonneg b.coe_nonneg dist_nonneg)).mpr
  intro s
  rw [Real.dist_eq]
  exact (sensitivity g hg rows w hw v u s).trans (mul_le_mul_of_nonneg_right (hm s) dist_nonneg)

/-- The fitted tabular policy has one value table. Iterating the exact backup
    converges to it; a measured sup-norm residual bounds the current error. -/
theorem value_certificate {n k : ℕ} {g : ℝ} (hg : 0 ≤ g) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (b : ℝ≥0) (hb : b < 1) (hm : ∀ s, g*liveMass rows w s ≤ b) :
    let T := operator g rows w
    ∃ V, T V = V ∧ (∀ U, T U = U → U = V) ∧
      ∀ u, Tendsto (fun t : ℕ => T^[t] u) atTop (𝓝 V) ∧
        dist u V ≤ dist u (T u)/(1-b) := by
  let T := operator g rows w
  have h := contraction hg rows w hw b hb hm
  exact ⟨h.fixedPoint T, h.fixedPoint_isFixedPt, fun _ hu => h.fixedPoint_unique hu,
    fun u => ⟨h.tendsto_iterate_fixedPoint u, h.dist_fixedPoint_le u⟩⟩

/-- The empirical mean squared TD loss bounds the squared residual at each
    state. The difference is precisely the conditional variance of its targets. -/
theorem td_risk {n k : ℕ} (g : ℝ) (rows : Fin n → Fin k → Row n)
    (w : Fin n → Fin k → ℝ) (hw : ∀ s, ∑ i, w s i = 1) (v u : Fin n → ℝ) (s : Fin n) :
    let y := fun i => Returns.backup g (rows s i).reward (v (rows s i).next) (rows s i).terminal
    ∑ i, w s i * (y i-u s)^2 =
      (∑ i, w s i * (y i-operator g rows w v s)^2) + (operator g rows w v s-u s)^2 :=
  BayesianReward.squared_risk (w s) _ (hw s) (u s)

/-- A per-state RMS residual bounds the distance to the mean of the actual
labels used in the squared-error regression. -/
theorem fitted_labels {n k : ℕ} {e : ℝ} (he : 0 ≤ e)
    (w y : Fin n → Fin k → ℝ) (hw : ∀ s i, 0 ≤ w s i)
    (mass : ∀ s, ∑ i, w s i = 1) (u : Fin n → ℝ)
    (loss : ∀ s, (∑ i, w s i * (y s i-u s)^2) ≤ e^2) :
    dist u (fun s => ∑ i, w s i * y s i) ≤ e := by
  apply (dist_pi_le_iff he).mpr
  intro s
  have h := loss s
  have decomp := BayesianReward.squared_risk (w s) (y s) (mass s) (u s)
  simp only [BayesianReward.average] at decomp
  rw [decomp] at h
  have var := sum_nonneg (s := (univ : Finset (Fin k))) fun i _ =>
    mul_nonneg (hw s i) (sq_nonneg (y s i - ∑ j, w s j * y s j))
  rw [Real.dist_eq]
  apply abs_le.mpr
  constructor <;> nlinarith

/-- RMS fitting error against a frozen target table bounds the sup-norm
    distance of the learned table to the exact backup of that target. -/
theorem fitted_backup {n k : ℕ} (g : ℝ) {e : ℝ} (he : 0 ≤ e)
    (rows : Fin n → Fin k → Row n) (w : Fin n → Fin k → ℝ)
    (hw : ∀ s i, 0 ≤ w s i) (mass : ∀ s, ∑ i, w s i = 1)
    (v u : Fin n → ℝ)
    (loss : ∀ s, (∑ i, w s i * (Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal-u s)^2) ≤ e^2) :
    dist u (operator g rows w v) ≤ e :=
  fitted_labels he w _ hw mass u loss

/-- A critic fitted to a lagged target pays the per-state RMS fitting error
    plus b times the maximum disagreement between the live and target tables.
    With synchronized tables the lag term vanishes. -/
theorem critic_certificate {n k : ℕ} {g e : ℝ} (hg : 0 ≤ g) (he : 0 ≤ e)
    (rows : Fin n → Fin k → Row n) (w : Fin n → Fin k → ℝ)
    (hw : ∀ s i, 0 ≤ w s i) (mass : ∀ s, ∑ i, w s i = 1)
    (b : ℝ≥0) (hb : b < 1) (hm : ∀ s, g*liveMass rows w s ≤ b)
    (v u V : Fin n → ℝ) (hV : operator g rows w V = V)
    (loss : ∀ s, (∑ i, w s i * (Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal-u s)^2) ≤ e^2) :
    dist u V ≤ (e+b*dist v u)/(1-b) := by
  let T := operator g rows w
  have c := contraction hg rows w hw b hb hm
  refine (c.dist_le_of_fixedPoint u hV).trans ?_
  apply (div_le_div_iff_of_pos_right c.one_sub_K_pos).mpr
  exact (dist_triangle u (T v) (T u)).trans
    (add_le_add (fitted_backup g he rows w hw mass v u loss) (c.dist_le_mul v u))

/-- A pooled TD loss on published labels certifies the empirical value when
every state has mass at least p. A measured lower bound on within-state target
variance removes irreducible loss. Only the mean signed target error matters. -/
theorem pooled_critic_certificate {n k : ℕ} {g p L noise δ : ℝ} (hg : 0 ≤ g)
    (hp : 0 < p) (hδ : 0 ≤ δ)
    (rows : Fin n → Fin k → Row n) (w y : Fin n → Fin k → ℝ)
    (hw : ∀ s i, 0 ≤ w s i) (mass : ∀ s, ∑ i, w s i = 1)
    (μ : Fin n → ℝ) (covered : ∀ s, p ≤ μ s)
    (b : ℝ≥0) (hb : b < 1) (hm : ∀ s, g*liveMass rows w s ≤ b)
    (v u V : Fin n → ℝ) (hV : operator g rows w V = V)
    (loss : (∑ s, μ s * ∑ i, w s i*(y s i-u s)^2) ≤ L)
    (hnoise : noise ≤ ∑ s, μ s * ∑ i, w s i*(y s i - ∑ j, w s j*y s j)^2)
    (bias : ∀ s, |∑ i, w s i*(y s i - Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal)| ≤ δ) :
    dist u V ≤ (Real.sqrt ((L-noise)/p) + δ + b*dist v u)/(1-b) := by
  let T := operator g rows w
  let published := fun s => ∑ i, w s i*y s i
  let err := fun s => (published s-u s)^2
  let variance := fun s => ∑ i, w s i*(y s i-published s)^2
  have decomp : (∑ s, μ s * ∑ i, w s i*(y s i-u s)^2) =
      (∑ s, μ s*variance s) + ∑ s, μ s*err s := by
    rw [← sum_add_distrib]
    apply sum_congr rfl
    intro s _
    rw [← mul_add]
    exact congrArg (μ s * ·) (BayesianReward.squared_risk (w s) (y s) (mass s) (u s))
  have pool : (∑ s, μ s*err s) ≤ L-noise := by
    change noise ≤ ∑ s, μ s*variance s at hnoise
    linarith
  have term (s) : 0 ≤ μ s*err s := mul_nonneg (hp.le.trans (covered s)) (sq_nonneg _)
  have fit : dist u published ≤ Real.sqrt ((L-noise)/p) := by
    apply (dist_pi_le_iff (Real.sqrt_nonneg _)).mpr
    intro s
    have single : err s ≤ (L-noise)/p := by
      apply (le_div_iff₀ hp).mpr
      rw [mul_comm]
      exact (mul_le_mul_of_nonneg_right (covered s) (sq_nonneg _)).trans
        ((single_le_sum (fun j _ => term j) (mem_univ s)).trans pool)
    have root := Real.sqrt_le_sqrt single
    simpa only [err, Real.sqrt_sq_eq_abs, Real.dist_eq, abs_sub_comm (u s)] using root
  have quant : dist published (T v) ≤ δ := by
    apply (dist_pi_le_iff hδ).mpr
    intro s
    rw [Real.dist_eq]
    change |(∑ i, w s i*y s i) - ∑ i, w s i*Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal| ≤ δ
    rw [← sum_sub_distrib]
    simpa only [mul_sub] using bias s
  have c := contraction hg rows w hw b hb hm
  refine (c.dist_le_of_fixedPoint u hV).trans ?_
  apply (div_le_div_iff_of_pos_right c.one_sub_K_pos).mpr
  calc
    dist u (T u) ≤ dist u (T v) + dist (T v) (T u) := dist_triangle ..
    _ ≤ (dist u published + dist published (T v)) + b*dist v u :=
      add_le_add (dist_triangle ..) (c.dist_le_mul v u)
    _ ≤ _ := add_le_add_right (add_le_add fit quant) _

/-- A state visited c times has weighted residual sum S and squared sum Q.
Centering the published labels removes exactly S²/c from Q, even when some
observations repeat or some padded observation slots have zero count. -/
theorem empirical_moments {k : ℕ} (m : Fin k → ℕ) (y : Fin k → ℝ) (u c : ℝ)
    (hc : 0 < c) (counts : ∑ i, (m i : ℝ) = c) :
    let w := fun i => (m i : ℝ)/c
    let S := ∑ i, (m i : ℝ)*(y i-u)
    let Q := ∑ i, (m i : ℝ)*(y i-u)^2
    (∑ i, w i*(y i-u)^2) = Q/c ∧
      (∑ i, w i*(y i-∑ j, w j*y j)^2) = (Q-S^2/c)/c := by
  dsimp only
  have avg (z : Fin k → ℝ) : (∑ i, (m i : ℝ)/c*z i) = (∑ i, (m i : ℝ)*z i)/c := by
    simp only [div_mul_eq_mul_div, sum_div]
  have mass : ∑ i, (m i : ℝ)/c = 1 := by rw [← sum_div, counts, div_self hc.ne']
  have mean : (∑ i, (m i : ℝ)/c*y i) - u = (∑ i, (m i : ℝ)*(y i-u))/c := by
    rw [avg]
    simp only [mul_sub, sum_sub_distrib, ← sum_mul, counts]
    field_simp
  have risk := BayesianReward.squared_risk (fun i => (m i : ℝ)/c) y mass u
  simp only [BayesianReward.average] at risk
  rw [mean, avg (fun i => (y i-u)^2)] at risk
  refine ⟨avg _, ?_⟩
  rw [show ((∑ i, (m i : ℝ)*(y i-u)^2) - (∑ i, (m i : ℝ)*(y i-u))^2/c)/c =
      (∑ i, (m i : ℝ)*(y i-u)^2)/c - ((∑ i, (m i : ℝ)*(y i-u))/c)^2 by ring]
  linarith

/-- Weighting each state's average by its observed visit frequency is the
same as adding all of the unnormalized state totals and dividing once. -/
theorem pool_counts {n : ℕ} (c z : Fin n → ℝ) (hc : ∀ s, 0 < c s) (total : ℝ) :
    (∑ s, c s/total*(z s/c s)) = (∑ s, z s)/total := by
  rw [sum_div]
  apply sum_congr rfl
  intro s _
  calc
    _ = (z s/total)*(c s/c s) := by ring
    _ = _ := by rw [div_self (hc s).ne', mul_one]

/-- Clearing the positive denominators and checking the sign before squaring
decides the real-valued certificate using only rational arithmetic on its inputs. -/
theorem tolerance_iff (L N p δ b d ε : ℝ) (hp : 0 < p) (hb : b < 1) :
    (Real.sqrt ((L-N)/p)+δ+b*d)/(1-b) ≤ ε ↔
      0 ≤ (1-b)*ε-δ-b*d ∧ L-N ≤ p*((1-b)*ε-δ-b*d)^2 := by
  rw [div_le_iff₀ (sub_pos.mpr hb), mul_comm ε]
  have rearrange : Real.sqrt ((L-N)/p)+δ+b*d ≤ (1-b)*ε ↔
      Real.sqrt ((L-N)/p) ≤ (1-b)*ε-δ-b*d := by constructor <;> intro h <;> linarith
  rw [rearrange, Real.sqrt_le_iff, div_le_iff₀ hp, mul_comm _ p]

/-- The count-based audit implies the stated tolerance on the empirical
fixed point. Its acceptance test consumes only per-state sums, visit counts,
and bounds on the continuing fraction, signed target bias and table lag. -/
theorem counted_critic_tolerance {n k : ℕ} {g p δ ε : ℝ}
    (hg : 0 ≤ g) (hp : 0 < p) (hδ : 0 ≤ δ)
    (rows : Fin n → Fin k → Row n) (y : Fin n → Fin k → ℝ)
    (m : Fin n → Fin k → ℕ) (c : Fin n → ℝ)
    (hc : ∀ s, 0 < c s) (counts : ∀ s, ∑ i, (m s i : ℝ) = c s)
    (covered : ∀ s, p ≤ c s/(∑ j, c j))
    (b : ℝ≥0) (hb : b < 1)
    (rate : ∀ s, g*((∑ i, if (rows s i).terminal then 0 else (m s i : ℝ))/c s) ≤ b)
    (v u V : Fin n → ℝ)
    (hV : operator g rows (fun s i => (m s i : ℝ)/c s) V = V)
    (bias : ∀ s, |(∑ i, (m s i : ℝ)*(y s i-Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal))/c s| ≤ δ)
    (accept : let margin := (1-b)*ε-δ-b*dist v u
      0 ≤ margin ∧ (∑ s, (∑ i, (m s i : ℝ)*(y s i-u s))^2/c s)/(∑ s, c s) ≤ p*margin^2) :
    dist u V ≤ ε := by
  let w := fun s i => (m s i : ℝ)/c s
  let μ := fun s => c s/(∑ j, c j)
  let S := fun s => ∑ i, (m s i : ℝ)*(y s i-u s)
  let Q := fun s => ∑ i, (m s i : ℝ)*(y s i-u s)^2
  let L := (∑ s, Q s)/(∑ s, c s)
  let N := (∑ s, (Q s-(S s)^2/c s))/(∑ s, c s)
  have hw (s i) : 0 ≤ w s i := div_nonneg (Nat.cast_nonneg _) (hc s).le
  have norm (s) : ∑ i, w s i = 1 := by
    simp only [w, ← sum_div, counts, div_self (hc s).ne']
  have hm (s) : g*liveMass rows w s ≤ b := by
    convert rate s using 2
    unfold liveMass
    rw [sum_div]
    apply sum_congr rfl
    intro i _
    cases h : (rows s i).terminal <;> simp [h,w]
  have herr (s) : |∑ i, w s i*(y s i-Returns.backup g (rows s i).reward
      (v (rows s i).next) (rows s i).terminal)| ≤ δ := by
    simpa only [w, div_mul_eq_mul_div, sum_div] using bias s
  have moments s := empirical_moments (m s) (y s) (u s) (c s) (hc s) (counts s)
  have loss : (∑ s, μ s*∑ i, w s i*(y s i-u s)^2) = L := by
    calc
      _ = ∑ s, μ s*(Q s/c s) :=
        sum_congr rfl fun s _ => congrArg (μ s * ·) (moments s).1
      _ = _ := pool_counts c Q hc _
  have noise : (∑ s, μ s*∑ i, w s i*(y s i-∑ j, w s j*y s j)^2) = N := by
    calc
      _ = ∑ s, μ s*((Q s-(S s)^2/c s)/c s) :=
        sum_congr rfl fun s _ => congrArg (μ s * ·) (moments s).2
      _ = _ := pool_counts c _ hc _
  have bound := pooled_critic_certificate hg hp hδ rows w y hw norm μ covered
    b hb hm v u V hV loss.le noise.ge herr
  apply bound.trans ((tolerance_iff L N p δ b (dist v u) ε hp hb).mpr ?_)
  have gap : L-N = (∑ s, (S s)^2/c s)/(∑ s, c s) := by
    dsimp [L,N]
    rw [sum_sub_distrib, sub_div]
    ring
  rw [gap]
  exact accept

/-- Successive approximate backups accumulate fitting or arithmetic errors
    geometrically. Delta may vary by iteration and includes the whole update. -/
theorem approximate_sweeps {n : ℕ} (T : (Fin n → ℝ) → (Fin n → ℝ)) (b : ℝ≥0)
    (hc : LipschitzWith b T) (V : Fin n → ℝ) (hV : T V = V)
    (u : ℕ → Fin n → ℝ) (δ : ℕ → ℝ) (h : ∀ t, dist (u (t+1)) (T (u t)) ≤ δ t)
    (t : ℕ) :
    dist (u t) V ≤ (b:ℝ)^t * dist (u 0) V + ∑ i ∈ range t, (b:ℝ)^(t-1-i)*δ i := by
  induction t with
  | zero => simp
  | succ t ih =>
    have step := dist_triangle (u (t+1)) (T (u t)) (T V)
    rw [hV] at step
    have c := hc.dist_le_mul (u t) V
    rw [hV] at c
    have sumShift : ∑ i ∈ range t, (b:ℝ)^(t+1-1-i)*δ i =
        b * ∑ i ∈ range t, (b:ℝ)^(t-1-i)*δ i := by
      rw [mul_sum]
      apply sum_congr rfl
      intro i hi
      have hi' := mem_range.mp hi
      rw [show t+1-1-i = (t-1-i)+1 by omega, pow_succ]
      ring
    rw [sum_range_succ, sumShift, pow_succ]
    simp only [Nat.add_sub_cancel, Nat.sub_self, pow_zero, one_mul]
    nlinarith [h t, mul_le_mul_of_nonneg_left ih b.coe_nonneg]

/-- A nonzero residual of size e can give exactly e/(1-b) value error
    on a one-state continuing self-loop. -/
theorem sharp (u r b : ℝ) (hb : b < 1) :
    |u-r/(1-b)| = |u-(r+b*u)|/(1-b) := by
  have hp : 0 < 1-b := sub_pos.mpr hb
  rw [show u-r/(1-b) = (u-(r+b*u))/(1-b) by field_simp; ring, abs_div, abs_of_pos hp]

/-- With all error on a self-loop of mass p and a perfect fit everywhere
else, the inverse square-root dependence on coverage is attained. -/
theorem pooled_sharp (p u r b : ℝ) (hp : 0 < p) (hb : b < 1) :
    |u-r/(1-b)| = Real.sqrt ((p*(u-(r+b*u))^2)/p)/(1-b) := by
  rw [mul_comm p, mul_div_cancel_right₀ _ hp.ne', Real.sqrt_sq_eq_abs, sharp u r b hb]

/-- An unobserved terminal state admits arbitrarily large value error at
zero pooled squared loss, even if all its reward labels are exact. -/
theorem uncovered (M : ℝ) (hM : 0 ≤ M) :
    let u : Fin 2 → ℝ := fun s => if s = 0 then 0 else M
    let μ : Fin 2 → ℝ := fun s => if s = 0 then 1 else 0
    (∑ s, μ s * (u s - Returns.backup 1 0 (u s) true)^2) = 0 ∧
      dist u (fun _ => (0:ℝ)) = M := by
  dsimp only
  constructor
  · simp [Fin.sum_univ_two, Returns.backup]
  · apply le_antisymm
    · apply (dist_pi_le_iff hM).mpr
      intro s
      by_cases h : s = 0 <;> simp [h, Real.dist_eq, abs_of_nonneg hM, hM]
    · have h := dist_le_pi_dist (fun s : Fin 2 => if s = 0 then (0:ℝ) else M)
          (fun _ => (0:ℝ)) (1:Fin 2)
      simpa [Real.dist_eq, abs_of_nonneg hM] using h

/-- One continuing machine backup has separate contributions from its
    reward, discount, successor and two arithmetic roundings. Bounds for a
    particular backend's multiply-add and output conversion enter as inputs. -/
theorem target_error (r g v storedR storedG storedV accum out : ℝ) :
    |out - (r+g*v)| ≤ |out-accum| + |accum-(storedR+storedG*storedV)| +
      |storedR-r| + |storedG| * |storedV-v| + |storedG-g| * |v| := by
  let a := out-accum
  let b := accum-(storedR+storedG*storedV)
  let c := storedR-r
  let d := storedG*(storedV-v)
  let e := (storedG-g)*v
  have eq : out-(r+g*v) = a+b+c+d+e := by dsimp [a,b,c,d,e]; ring
  rw [eq]
  calc
    _ ≤ |a+b+c+d|+|e| := abs_add_le ..
    _ ≤ |a+b+c|+|d|+|e| := by linarith [abs_add_le (a+b+c) d]
    _ ≤ |a+b|+|c|+|d|+|e| := by linarith [abs_add_le (a+b) c]
    _ ≤ |a|+|b|+|c|+|d|+|e| := by linarith [abs_add_le a b]
    _ = _ := by simp only [a,b,c,d,e,abs_mul]

/-- Using the wrong discount changes a live target by exactly this amount.
    A terminal row is invariant under either discount and successor estimate. -/
theorem discount_difference (a b r v : ℝ) (done : Bool) :
    Returns.backup a r v done - Returns.backup b r v done =
      if done then 0 else (a-b)*v := by
  cases done <;> simp [Returns.backup]
  ring

end
end StateSetFormal.LoggedBellman
