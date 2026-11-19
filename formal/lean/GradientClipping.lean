import Optimizer
import CollectiveRounding

namespace StateSetFormal.GradientClipping
open Finset Real
open scoped BigOperators
noncomputable section

/-- A widened binary32 significand fits into binary64 even after squaring.
The exponent is doubled separately, so this covers subnormal inputs too. -/
theorem wide_square {m : ℕ} (hm : m < 2^24) (e : ℤ) (s : ℝ) (hs : s^2 = 1) :
    m*m < 2^53 ∧ (s * m * (2:ℝ)^e)^2 = (m*m : ℕ) * (2:ℝ)^(e+e) := by
  constructor
  · have h := Nat.mul_self_lt_mul_self hm
    exact lt_trans h (by norm_num)
  · rw [zpow_add₀ (by norm_num : (2:ℝ) ≠ 0)]
    push_cast
    simp only [mul_pow, hs, one_mul]
    ring

/-- The total square sum of any addressable tensor of finite binary32
coordinates is inside the normal binary64 upper range. -/
theorem wide_sum_headroom {n : ℕ} (hn : n ≤ 2^63) (g : Fin n → ℝ)
    (hg : ∀ i, |g i| ≤ 2^128) : ∑ i, (g i)^2 ≤ (2:ℝ)^319 := by
  calc
    ∑ i, (g i)^2 ≤ ∑ _i : Fin n, ((2:ℝ)^128)^2 := by
      apply sum_le_sum
      intro i _
      simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg (g i)) (hg i) 2
    _ = (n:ℝ) * ((2:ℝ)^128)^2 := by simp
    _ ≤ (2:ℝ)^63 * ((2:ℝ)^128)^2 := by
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast hn) (by positivity)
    _ = _ := by norm_num

/-- Even the smallest nonzero binary32 magnitude squares above the
binary64 normal floor. -/
theorem wide_square_floor (g : ℝ) (hg : (2:ℝ)^(-149:ℤ) ≤ |g|) :
    (2:ℝ)^(-1022:ℤ) ≤ g^2 := by
  calc
    (2:ℝ)^(-1022:ℤ) ≤ ((2:ℝ)^(-149:ℤ))^2 := by norm_num
    _ ≤ _ := by simpa only [sq_abs] using pow_le_pow_left₀ (by positivity) hg 2

/-- Splitting a ragged coordinate list into blocks preserves total energy in
real arithmetic. Blocks may have arbitrary lengths, including zero. -/
theorem partition_energy (blocks : List (List ℝ)) :
    ((blocks.flatten.map (fun x => x^2)).sum) =
      (blocks.map (fun b => (b.map (fun x => x^2)).sum)).sum := by
  induction blocks with
  | nil => simp
  | cons b bs ih => simp [List.map_append, List.sum_append, ih, Function.comp_def]

/-- On a common dyadic grid, rounding a nonnegative integer sum to p
significant bits with ties to even has relative error at most 2^(-p).
For small exact inputs this also covers zero. Exponent headroom is separate. -/
theorem nearest_relative (x : ℕ) {p : ℕ} (hp : 0 < p) :
    (2:ℤ)^p * |CollectiveRounding.binaryRound p x - x| ≤ x := by
  by_cases small : x < 2^p
  · rw [CollectiveRounding.exact_significand hp (by simpa using small)]
    simp
  · have hx : x ≠ 0 := by
      intro hz
      subst x
      exact small (by positivity)
    have bits : p ≤ x.log2 := (Nat.le_log2 hx).mpr (by omega)
    have baseBound : 2^x.log2 ≤ x := (Nat.le_log2 hx).mp le_rfl
    let spacing : ℤ := 2^(x.log2+1-p)
    have near := CollectiveRounding.nearest (x:ℤ) (d := spacing) (by dsimp [spacing]; positivity)
    have shift : (2:ℤ)^p * spacing = 2*(2:ℤ)^x.log2 := by
      dsimp [spacing]
      rw [← pow_add, show p+(x.log2+1-p) = x.log2+1 by omega, pow_succ']
    have combined := mul_le_mul_of_nonneg_left near (by positivity : 0 ≤ (2:ℤ)^p)
    rw [shift] at combined
    change (2:ℤ)^p * |CollectiveRounding.roundEven (x:ℤ) spacing * spacing - x| ≤ x
    have upper : (2:ℤ)^x.log2 ≤ x := by exact_mod_cast baseBound
    linarith only [combined, upper]

/-- The local addition bounds used below hold for the modeled binary64
round-to-nearest operation on every nonnegative common grid. -/
theorem binary64_add (a b : ℕ) (unit : ℝ) (hu : 0 ≤ unit) :
    let exactSum := ((a+b : ℕ):ℝ)*unit
    let stored := (CollectiveRounding.binaryRound 53 (a+b : ℕ):ℝ)*unit
    (1 - 1/(2:ℝ)^53)*exactSum ≤ stored ∧ stored ≤ (1 + 1/(2:ℝ)^53)*exactSum := by
  have bound := nearest_relative (a+b) (p := 53) (by decide)
  let v : ℝ := CollectiveRounding.binaryRound 53 (a+b : ℕ)
  have h : (2:ℝ)^53 * |v - ((a+b : ℕ):ℝ)| ≤ ((a+b : ℕ):ℝ) := by
    dsimp [v]
    exact_mod_cast bound
  have rel : |v - ((a+b : ℕ):ℝ)| ≤ ((a+b : ℕ):ℝ)/(2:ℝ)^53 :=
    (le_div_iff₀ (by positivity)).mpr (by rw [mul_comm]; exact h)
  obtain ⟨lo, hi⟩ := abs_le.mp rel
  change (1-1/(2:ℝ)^53)*(((a+b : ℕ):ℝ)*unit) ≤ v*unit ∧
    v*unit ≤ (1+1/(2:ℝ)^53)*(((a+b : ℕ):ℝ)*unit)
  rw [← mul_assoc, ← mul_assoc]
  exact ⟨mul_le_mul_of_nonneg_right (by linarith) hu, mul_le_mul_of_nonneg_right (by linarith) hu⟩

/-- A bounded-depth floating reduction. Leaves represent exact nonnegative
squares. A leaf may terminate at any remaining depth, so uneven trees and
chunk boundaries do not require fictitious rounded additions. -/
inductive SquareReduction (u : ℝ) : ℕ → ℝ → ℝ → Prop
  | leaf (h : ℕ) (x : ℝ) (hx : 0 ≤ x) : SquareReduction u h x x
  | add {h : ℕ} {x y a b z : ℝ}
      (left : SquareReduction u h x a) (right : SquareReduction u h y b)
      (lo : (1-u)*(a+b) ≤ z) (hi : z ≤ (1+u)*(a+b)) :
      SquareReduction u (h+1) (x+y) z

/-- If each binary addition meets its relative-error contract, the full
energy lies between the two depth-dependent factors of the exact square sum. -/
theorem reduction_error {u x measured : ℝ} {h : ℕ} (hu : 0 ≤ u) (hu1 : u < 1)
    (tree : SquareReduction u h x measured) :
    (1-u)^h * x ≤ measured ∧ measured ≤ (1+u)^h * x := by
  induction tree with
  | leaf h x hx =>
    have lo : (1-u)^h ≤ 1 := pow_le_one₀ (by linarith) (by linarith)
    have hi : 1 ≤ (1+u)^h := one_le_pow₀ (by linarith)
    exact ⟨by simpa using mul_le_mul_of_nonneg_right lo hx,
      by simpa using mul_le_mul_of_nonneg_right hi hx⟩
  | add left right lo hi ihl ihr =>
    constructor
    · apply le_trans _ lo
      rw [pow_succ', mul_assoc, mul_add ((1-u)^_)]
      exact mul_le_mul_of_nonneg_left (add_le_add ihl.1 ihr.1) (by linarith : 0 ≤ 1-u)
    · apply le_trans hi
      rw [pow_succ', mul_assoc, mul_add ((1+u)^_)]
      exact mul_le_mul_of_nonneg_left (add_le_add ihl.2 ihr.2) (by linarith : 0 ≤ 1+u)

/-- Bound the effect of a norm measurement error before rounding the output
coordinates. An underestimate no larger than epsilon preserves the cap. -/
theorem estimated_clip {actual measured cap eps : ℝ} (hm : 0 ≤ measured)
    (hc : 0 ≤ cap) (he : 0 < eps) :
    let k := min 1 (cap/(measured+eps))
    k * actual ≤ cap + k * max 0 (actual - (measured+eps)) := by
  let k := min 1 (cap/(measured+eps))
  have hd : 0 < measured+eps := by linarith
  have hk : 0 ≤ k := le_min zero_le_one (div_nonneg hc hd.le)
  have hcap : k*(measured+eps) ≤ cap := (le_div_iff₀ hd).mp (min_le_right ..)
  have herr := mul_le_mul_of_nonneg_left (le_max_right 0 (actual-(measured+eps))) hk
  change k*actual ≤ cap + k*max 0 (actual-(measured+eps))
  linarith [show k*actual = k*(measured+eps) + k*(actual-(measured+eps)) by ring]

/-- The measured-norm bound composes with coordinate rounding. The error
vector includes the coefficient approximation, multiplications and final cast. -/
theorem stored_clip {n : ℕ} (g err : Fin n → ℝ) {measured cap eps : ℝ}
    (hm : 0 ≤ measured) (hc : 0 ≤ cap) (he : 0 < eps) :
    let actual := sqrt (∑ i, (g i)^2)
    let k := min 1 (cap/(measured+eps))
    sqrt (∑ i, (k*g i + err i)^2) ≤
      cap + k * max 0 (actual - (measured+eps)) + sqrt (∑ i, (err i)^2) := by
  let k := min 1 (cap/(measured+eps))
  let v := fun i => k*g i
  have hk : 0 ≤ k := (Optimizer.clip_bounds hm hc he).1
  have hv : sqrt (∑ i, (v i)^2) = k * sqrt (∑ i, (g i)^2) := by
    simp only [v, mul_pow, ← mul_sum, Real.sqrt_mul (sq_nonneg k), Real.sqrt_sq hk]
  have cross := Real.sum_mul_le_sqrt_mul_sqrt univ v err
  have triangle : sqrt (∑ i, (v i + err i)^2) ≤
      sqrt (∑ i, (v i)^2) + sqrt (∑ i, (err i)^2) := by
    apply (Real.sqrt_le_left (by positivity)).mpr
    simp only [add_sq, mul_assoc, sum_add_distrib, ← mul_sum]
    nlinarith [Real.sq_sqrt (sum_nonneg (s := univ) fun i _ => sq_nonneg (v i)),
      Real.sq_sqrt (sum_nonneg (s := univ) fun i _ => sq_nonneg (err i))]
  rw [hv] at triangle
  exact triangle.trans (add_le_add_right (estimated_clip hm hc he) _)

/-- Cap one on (1,2). Both exact coordinates are strictly inside the
binary16 rounding cells centered on (229/512,229/256), whose norm exceeds one.
The centers also belong to the coarser bfloat16 grid at these exponents. -/
theorem half_rounding_witness :
    let x := 1 / (sqrt 5 + 1/1000000)
    |x - 229/512| < 1/8192 ∧ |2*x - 229/256| < 1/4096 ∧
      (1:ℝ) < (229/512)^2 + (229/256)^2 := by
  have lo : (223606:ℝ)/100000 < sqrt 5 := Real.lt_sqrt_of_sq_lt (by norm_num)
  have hi : sqrt 5 < (223607:ℝ)/100000 := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
  have hd : 0 < sqrt 5 + 1/1000000 := by positivity
  have l : (229:ℝ)/512 - 1/8192 < 1/(sqrt 5 + 1/1000000) :=
    (lt_div_iff₀ hd).mpr (by linarith)
  have u : 1/(sqrt 5 + 1/1000000) < (229:ℝ)/512 + 1/8192 :=
    (div_lt_iff₀ hd).mpr (by linarith)
  exact ⟨abs_lt.mpr ⟨by linarith, by linarith⟩,
    abs_lt.mpr ⟨by linarith, by linarith⟩, by norm_num⟩

/-- For independently clipped heads the joint norm is bounded by the root
sum of squared head caps, including empty or zero-norm heads. -/
theorem head_caps {n : ℕ} (norm cap : Fin n → ℝ) (hn : ∀ i, 0 ≤ norm i)
    (hc : ∀ i, 0 ≤ cap i) {eps : ℝ} (he : 0 < eps) :
    sqrt (∑ i, (min 1 (cap i/(norm i+eps)) * norm i)^2) ≤
      sqrt (∑ i, (cap i)^2) := by
  apply Real.sqrt_le_sqrt
  apply sum_le_sum
  intro i _
  have h := Optimizer.clip_bounds (hn i) (hc i) he
  exact pow_le_pow_left₀ (mul_nonneg h.1 (hn i)) h.2.2 2

/-- Two independently clipped heads can strictly exceed one common cap,
for every positive cap and epsilon. -/
theorem two_heads_exceed {cap eps : ℝ} (hc : 0 < cap) (he : 0 < eps) :
    let n := 4*(cap+eps)
    let k := min 1 (cap/(n+eps))
    cap^2 < (k*n)^2 + (k*n)^2 := by
  let n := 4*(cap+eps)
  have h : 0 < n+eps := by dsimp [n]; linarith
  have small : cap/(n+eps) ≤ 1 := (div_le_one h).mpr (by dsimp [n]; linarith)
  change cap^2 < (min 1 (cap/(n+eps))*n)^2 + (min 1 (cap/(n+eps))*n)^2
  rw [min_eq_right small]
  have bound : 3*cap/4 < cap/(n+eps)*n := by
    rw [div_mul_eq_mul_div, lt_div_iff₀ h]
    dsimp [n]
    nlinarith [mul_pos hc he, sq_pos_of_pos hc]
  have sq := pow_lt_pow_left₀ bound (by positivity) (by decide : (2:ℕ) ≠ 0)
  nlinarith [sq_pos_of_pos hc]

end
end StateSetFormal.GradientClipping
