import Optimizer
import CollectiveRounding

namespace StateSetFormal.AdamHistory
open Finset Real
open scoped BigOperators
noncomputable section

/-- Bias-corrected weight at age i in a nonempty parameter-local history. -/
def weight (b : ℝ) (n i : ℕ) := (1-b)*b^i/(1-b^n)

theorem history {n : ℕ} (g : Fin n → ℝ) (b : ℝ) :
    Optimizer.moment b (List.ofFn g)/(1-b^n) =
      ∑ i : Fin n, weight b n i * g i := by
  rw [Optimizer.moment_expansion, mul_sum, sum_div]
  apply sum_congr rfl
  intro i _
  unfold weight
  ring

theorem weights_sum {n : ℕ} (hn : n ≠ 0) {b : ℝ} (hb : 0 ≤ b) (hb1 : b < 1) :
    ∑ i : Fin n, weight b n i = 1 := by
  have h := history (n := n) (fun _ => (1:ℝ)) b
  simpa [List.ofFn_const, Optimizer.corrected_constant hb hb1 _ hn] using h.symm

theorem weight_pos {n : ℕ} (hn : n ≠ 0) {b : ℝ} (hb : 0 < b) (hb1 : b < 1) (i : Fin n) :
    0 < weight b n i := by
  exact div_pos (mul_pos (sub_pos.mpr hb1) (pow_pos hb _))
    (sub_pos.mpr (pow_lt_one₀ hb.le hb1 hn))

/-- The two potentially different bias-correction kernels give a sharp
Cauchy-Schwarz coefficient. All finite signed gradient histories are allowed. -/
theorem weighted_moments {n : ℕ} (g p q : Fin n → ℝ) (hq : ∀ i, 0 < q i) :
    (∑ i, p i*g i)^2 ≤ (∑ i, (p i)^2/q i) * (∑ i, q i*(g i)^2) := by
  let a := fun i => p i / sqrt (q i)
  let b := fun i => sqrt (q i) * g i
  have prod : ∀ i, a i*b i = p i*g i := by
    intro i
    dsimp [a, b]
    field_simp [(sqrt_pos.mpr (hq i)).ne']
    ring
  have aa : ∀ i, (a i)^2 = (p i)^2/q i := by
    intro i
    simp [a, div_pow, Real.sq_sqrt (hq i).le]
  have bb : ∀ i, (b i)^2 = q i*(g i)^2 := by
    intro i
    simp [b, mul_pow, Real.sq_sqrt (hq i).le]
  simpa only [prod, aa, bb] using sum_mul_sq_le_sq_mul_sq univ a b

/-- For an arbitrary nonempty local AdamW history, the displacement from
the decayed weight is bounded even if the newest gradient is zero. -/
theorem displacement {n : ℕ} (hn : n ≠ 0) (g : Fin n → ℝ) (b1 : ℝ)
    {b2 : ℝ} (hb : 0 < b2) (hb1 : b2 < 1) (w decay : ℝ)
    {lr eps : ℝ} (hl : 0 ≤ lr) (he : 0 < eps) :
    let m := Optimizer.moment b1 (List.ofFn g)/(1-b1^n)
    let v := Optimizer.moment b2 (List.ofFn fun i => (g i)^2)/(1-b2^n)
    |Optimizer.adamW w lr decay eps m v - (1-lr*decay)*w| ≤
      lr * sqrt (∑ i : Fin n, (weight b1 n i)^2 / weight b2 n i) := by
  let q := fun i : Fin n => weight b2 n i
  let p := fun i : Fin n => weight b1 n i
  let m := ∑ i, p i*g i
  let v := ∑ i, q i*(g i)^2
  let c := ∑ i, (p i)^2/q i
  have hq : ∀ i, 0 < q i := weight_pos hn hb hb1
  have hv : 0 ≤ v := sum_nonneg fun i _ => mul_nonneg (hq i).le (sq_nonneg _)
  have hc : 0 ≤ c := sum_nonneg fun i _ => div_nonneg (sq_nonneg _) (hq i).le
  have bound : m^2 ≤ c*v := weighted_moments g p q hq
  have hmag : |m| ≤ sqrt c * sqrt v := by
    rw [← Real.sqrt_sq_eq_abs, ← Real.sqrt_mul hc]
    exact Real.sqrt_le_sqrt bound
  have den : 0 < sqrt v + eps := by positivity
  simp only [history, Optimizer.adamW, sub_sub_cancel_left, abs_neg, abs_div, abs_mul,
    abs_of_nonneg hl, abs_of_pos den]
  change lr*|m|/(sqrt v+eps) ≤ lr*sqrt c
  apply (div_le_iff₀ den).mpr
  calc
    lr*|m| ≤ lr*(sqrt c * sqrt v) := mul_le_mul_of_nonneg_left hmag hl
    _ ≤ _ := by nlinarith [mul_nonneg (mul_nonneg hl (sqrt_nonneg c)) he.le]

/-- Using the same geometric kernel for both moments gives a
per-coordinate adaptive displacement no greater than the current rate. -/
theorem same_decay {n : ℕ} (hn : n ≠ 0) (g : Fin n → ℝ)
    {b : ℝ} (hb : 0 < b) (hb1 : b < 1) (w decay : ℝ)
    {lr eps : ℝ} (hl : 0 ≤ lr) (he : 0 < eps) :
    let m := Optimizer.moment b (List.ofFn g)/(1-b^n)
    let v := Optimizer.moment b (List.ofFn fun i => (g i)^2)/(1-b^n)
    |Optimizer.adamW w lr decay eps m v - (1-lr*decay)*w| ≤ lr := by
  have h := displacement hn g b hb hb1 w decay hl he
  have ratio : ∀ i : Fin n, (weight b n i)^2 / weight b n i = weight b n i := by
    intro i
    field_simp [(weight_pos hn hb hb1 i).ne']
    ring
  simpa only [ratio, weights_sum hn hb.le hb1, Real.sqrt_one, mul_one] using h

/-- Even with both gradients in the unit interval, different moment
decays can produce an adaptive displacement greater than the learning rate. -/
theorem distinct_decay_witness :
    let m := Optimizer.moment 0 [1,0] / (1-(0:ℝ)^2)
    let v := Optimizer.moment (1/2) [1,0] / (1-(1/2:ℝ)^2)
    Optimizer.adamW 0 1 0 (1/10) m v < -1 := by
  have h : sqrt (2/3 : ℝ) < 9/10 := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
  have den : 0 < sqrt (2/3 : ℝ) + 1/10 := by positivity
  have div : 1 < 1/(sqrt (2/3 : ℝ) + 1/10) := (lt_div_iff₀ den).mpr (by linarith)
  norm_num [Optimizer.adamW, Optimizer.moment] at *
  linarith

/-- At the allowed beta2=0 endpoint, a zero current gradient erases
second-moment history. Older first-moment signal is then divided by epsilon. -/
theorem zero_second_decay (old w lr decay eps : ℝ) :
    let m := Optimizer.moment (1/2) [0,old] / (1-(1/2:ℝ)^2)
    let v := Optimizer.moment 0 [0,old^2] / (1-(0:ℝ)^2)
    Optimizer.adamW w lr decay eps m v = (1-lr*decay)*w - lr*old/(3*eps) := by
  norm_num [Optimizer.moment, Optimizer.adamW]
  ring

/-- A zero provided gradient still moves a parameter with momentum.
A missing gradient omits the entire local step, including decoupled decay. -/
theorem zero_after_positive (w decay : ℝ) {lr eps : ℝ} (hl : 0 < lr) (he : 0 < eps) :
    let m := Optimizer.moment (1/2) [0,1] / (1-(1/2:ℝ)^2)
    let v := Optimizer.moment (1/2) [0,1] / (1-(1/2:ℝ)^2)
    Optimizer.adamW w lr decay eps m v < (1-lr*decay)*w := by
  apply Optimizer.adaptive_descent hl he
  norm_num [Optimizer.moment]

/-- The default epsilon on the binary16 subnormal grid and on the
binary32 normal grid at its exponent. Nearest-even arithmetic erases it
in binary16 and gives a positive denominator at a stationary FP32 coordinate. -/
theorem epsilon_grids :
    CollectiveRounding.roundEven (2^24) 100000000 = 0 ∧
    CollectiveRounding.roundEven (2^50) 100000000 = 11258999 := by decide

/-- A gradient can be absent on one parameter while other parameters use
the current learning rate. Events are newest first. -/
abbrev Event := Option ℝ × ℝ

structure LocalState where
  w : ℝ
  m : ℝ
  v : ℝ
  t : ℕ

def run (w b1 b2 decay eps : ℝ) : List Event → LocalState
  | [] => ⟨w, 0, 0, 0⟩
  | (gradient, lr) :: past =>
    let s := run w b1 b2 decay eps past
    match gradient with
    | none => s
    | some g =>
      let m := b1*s.m + (1-b1)*g
      let v := b2*s.v + (1-b2)*g^2
      ⟨Optimizer.adamW s.w lr decay eps (m/(1-b1^(s.t+1))) (v/(1-b2^(s.t+1))),
        m, v, s.t+1⟩

def gradients (rows : List Event) := rows.filterMap Prod.fst

theorem local_moments (w b1 b2 decay eps : ℝ) (rows : List Event) :
    let s := run w b1 b2 decay eps rows
    s.t = (gradients rows).length ∧ s.m = Optimizer.moment b1 (gradients rows) ∧
      s.v = Optimizer.moment b2 ((gradients rows).map fun x => x^2) := by
  induction rows with
  | nil => simp [run, gradients, Optimizer.moment]
  | cons e past ih =>
    rcases e with ⟨g, lr⟩
    cases g <;> simp_all [run, gradients, Optimizer.moment]

/-- At each event compute its decay multiplier and adaptive displacement
independently from the nonmissing gradient history. -/
def terms (b1 b2 decay eps : ℝ) : List Event → List (ℝ × ℝ)
  | [] => []
  | (g,lr)::past =>
    let tail := terms b1 b2 decay eps past
    match g with
    | none => (1,0)::tail
    | some x =>
      let h := x::gradients past
      let m := Optimizer.moment b1 h/(1-b1^h.length)
      let v := Optimizer.moment b2 (h.map fun a => a^2)/(1-b2^h.length)
      (1-lr*decay, lr*m/(sqrt v + eps))::tail

def affine (w : ℝ) : List (ℝ × ℝ) → ℝ
  | [] => w
  | (a,u)::past => a*affine w past-u

theorem weight_from_history (w b1 b2 decay eps : ℝ) (rows : List Event) :
    (run w b1 b2 decay eps rows).w = affine w (terms b1 b2 decay eps rows) := by
  induction rows with
  | nil => rfl
  | cons e past ih =>
    obtain ⟨count, first, second⟩ := local_moments w b1 b2 decay eps past
    rcases e with ⟨g, lr⟩
    cases g <;> simp [run, terms, affine, Optimizer.adamW, Optimizer.moment, ih,
      count, first, second]

/-- Prefix products of the later decay multipliers discount each earlier
adaptive contribution. A missing gradient contributes multiplier one and zero
displacement even when other parameters take a step. -/
def discounted (factor : ℝ) : List (ℝ × ℝ) → List ℝ
  | [] => []
  | (a,u)::past => (factor*u)::discounted (factor*a) past

theorem affine_expansion (w factor : ℝ) (steps : List (ℝ × ℝ)) :
    factor * affine w steps =
      factor*(steps.map Prod.fst).prod*w - (discounted factor steps).sum := by
  induction steps generalizing factor with
  | nil => simp [affine, discounted]
  | cons step past ih =>
    rcases step with ⟨a,u⟩
    simp only [affine, discounted, List.map_cons, List.prod_cons, List.sum_cons]
    rw [mul_sub, ← mul_assoc, ih]
    ring

theorem entire_weight (w b1 b2 decay eps : ℝ) (rows : List Event) :
    let updates := terms b1 b2 decay eps rows
    (run w b1 b2 decay eps rows).w =
      (updates.map Prod.fst).prod*w - (discounted 1 updates).sum := by
  rw [weight_from_history]
  simpa using affine_expansion w 1 (terms b1 b2 decay eps rows)

/-- Repeated sub-cell moves disappear if each step is fed its last published
weight. Carrying the working weight crosses the rounding boundary after
enough equal moves. Integers count units of the finer working grid. -/
theorem hidden_motion (start step steps spacing : ℤ) (hd : 0 < spacing)
    (hs : 0 < step) (small : 2*step < spacing) (cross : spacing < 2*(steps*step)) :
    CollectiveRounding.roundEven (start*spacing-step) spacing = start ∧
    CollectiveRounding.roundEven (start*spacing-steps*step) spacing ≠ start := by
  constructor
  · apply CollectiveRounding.unique_nearest hd
    simpa [abs_of_pos hs] using small
  · intro eq
    have bound := CollectiveRounding.nearest (start*spacing-steps*step) hd
    rw [eq] at bound
    have pos : 0 < steps*step := by linarith
    simp [abs_of_pos pos] at bound
    linarith

end
end StateSetFormal.AdamHistory
