import GradientBank

namespace StateSetFormal.CollectiveRounding

/-- Nearest integer with ties to even, including negative numerators.
    The denominator is positive in all applications. -/
def roundEven (n d : ℤ) : ℤ :=
  let q := n / d
  let r := n % d
  if 2*r < d then q else if d < 2*r then q+1 else if q%2 = 0 then q else q+1

theorem nearest (n : ℤ) {d : ℤ} (hd : 0 < d) :
    2 * |roundEven n d*d - n| ≤ d := by
  have hr := Int.emod_nonneg n hd.ne'
  have hr' := Int.emod_lt_of_pos n hd
  have lo : n/d*d - n = -(n%d) := by linarith [Int.ediv_add_emod' n d]
  have hi : (n/d+1)*d - n = d - n%d := by rw [add_mul, one_mul]; linarith
  simp only [roundEven]
  split <;> rename_i h
  · rw [lo, abs_neg, abs_of_nonneg hr]; omega
  · split
    · rw [hi, abs_of_nonneg (by omega)]; omega
    · split
      · rw [lo, abs_neg, abs_of_nonneg hr]; omega
      · rw [hi, abs_of_nonneg (by omega)]; omega

theorem tie_even (n : ℤ) {d : ℤ} (h : 2*(n%d) = d) :
    roundEven n d % 2 = 0 := by
  simp only [roundEven, h, lt_self_iff_false, if_false]
  split <;> omega

theorem exact (v : ℤ) {d : ℤ} (hd : 0 < d) : roundEven (v*d) d = v := by
  simp [roundEven, Int.mul_ediv_cancel _ hd.ne', hd]

theorem unique_nearest {n d z : ℤ} (hd : 0 < d) (close : 2*|z*d-n| < d) :
    roundEven n d = z := by
  have bound := nearest n hd
  have tri := abs_sub_le (roundEven n d*d) n (z*d)
  rw [abs_sub_comm n (z*d), ← sub_mul, abs_mul, abs_of_pos hd] at tri
  by_contra h
  have one : 1 ≤ |roundEven n d-z| := by
    have := abs_pos.mpr (sub_ne_zero.mpr h)
    omega
  have h' := mul_le_mul_of_nonneg_right one hd.le
  linarith

/-- A fine nearest-even rounding before the final cast cannot change a
    rational mean when its spacing is 1/s, s is even and s exceeds the
    denominator. This covers non-dyadic small-population divisions too. -/
theorem fine_then_coarse {n d s : ℤ} (hd : 0 < d) (hs : d < s) (he : Even s) :
    roundEven (roundEven (n*s) d) s = roundEven n d := by
  let g := roundEven n d
  let w := roundEven (n*s) d
  have hg : 2*|g*d-n| ≤ d := nearest n hd
  have hw : 2*|w*d-n*s| ≤ d := nearest _ hd
  have s_pos : 0 < s := by omega
  by_cases mid : 2*(n%d) = d
  · obtain ⟨k, hk⟩ := he
    have hk0 : 0 ≤ k := by omega
    have hks : k < s := by omega
    have decomp := Int.ediv_add_emod' n d
    have repr : n*s = (k + n/d*s)*d := by
      calc
        _ = (n/d*d + n%d)*s := by rw [decomp]
        _ = (n/d*s)*d + (2*(n%d))*k := by rw [hk]; ring
        _ = _ := by rw [mid]; ring
    rw [repr, exact _ hd]
    simp only [roundEven, Int.add_mul_ediv_right _ _ s_pos.ne',
      Int.ediv_eq_zero_of_lt hk0 hks, zero_add, Int.add_mul_emod_self,
      Int.emod_eq_of_lt hk0 hks, show 2*k = s by omega, mid, lt_self_iff_false, if_false]
  · have strict : 2*|g*d-n| < d := by
      have rem := Int.emod_nonneg n hd.ne'
      have rem' := Int.emod_lt_of_pos n hd
      have lo : n/d*d-n = -(n%d) := by linarith [Int.ediv_add_emod' n d]
      have hi : (n/d+1)*d-n = d-n%d := by rw [add_mul, one_mul]; linarith
      simp only [g, roundEven]
      split <;> (try split) <;> (try split)
      all_goals first
        | rw [lo, abs_neg, abs_of_nonneg rem]; omega
        | rw [hi, abs_of_nonneg (by omega)]; omega
    apply unique_nearest s_pos
    change 2*|g*s-w| < s
    have bound : |(g*s-w)*d| ≤ |g*d-n| * s + |w*d-n*s| := by
      calc
        _ = |(g*d-n)*s - (w*d-n*s)| := by congr 1; ring
        _ ≤ |(g*d-n)*s| + |w*d-n*s| := abs_sub _ _
        _ = _ := by rw [abs_mul, abs_of_pos s_pos]
    rw [abs_mul, abs_of_pos hd] at bound
    have gap := mul_le_mul_of_nonneg_right (show 2*|g*d-n| ≤ d-1 by omega) s_pos.le
    apply (lt_of_mul_lt_mul_right (a := d) _ hd.le)
    nlinarith

/-- On a binary grid a value with p significant bits rounds to a multiple
    of 2^(bitLength - p), with the exponent range treated separately. -/
def binaryRound (p : ℕ) (x : ℤ) : ℤ :=
  let spacing : ℤ := 2 ^ (x.natAbs.log2 + 1 - p)
  roundEven x spacing * spacing

theorem exact_significand {p : ℕ} (hp : 0 < p) {x : ℤ} (hx : x.natAbs < 2^p) :
    binaryRound p x = x := by
  have h : x.natAbs.log2 < p := by
    by_cases hz : x.natAbs = 0
    · simpa [hz] using hp
    · exact (Nat.log2_lt hz).mpr hx
  have h' : x.natAbs.log2 + 1 - p = 0 := by omega
  simp [binaryRound, h', roundEven]

inductive SumTree where
  | leaf (units : ℤ)
  | join (left right : SumTree)

def SumTree.total : SumTree → ℤ
  | .leaf v => v
  | .join l r => l.total + r.total

def SumTree.mass : SumTree → ℕ
  | .leaf v => v.natAbs
  | .join l r => l.mass + r.mass

def SumTree.reduce (p : ℕ) : SumTree → ℤ
  | .leaf v => v
  | .join l r => binaryRound p (l.reduce p + r.reduce p)

theorem total_bounded (t : SumTree) : t.total.natAbs ≤ t.mass := by
  induction t with
  | leaf => exact le_rfl
  | join l r hl hr => exact (Int.natAbs_add_le ..).trans (Nat.add_le_add hl hr)

/-- For any signed rank order and any reduction tree, the widened SUM is
    exact whenever the total absolute grid-unit mass fits the significand.
    The unit itself must be representable at every required exponent. -/
theorem exact_tree {p : ℕ} (hp : 0 < p) (t : SumTree) (h : t.mass < 2^p) :
    t.reduce p = t.total := by
  induction t with
  | leaf => rfl
  | join l r hl hr =>
    change l.mass + r.mass < 2^p at h
    simp only [SumTree.reduce, hl (by omega), hr (by omega), SumTree.total]
    exact exact_significand hp ((total_bounded (.join l r)).trans_lt h)

/-- Sum of the actual absolute local rounding errors, in units of the grid. -/
def SumTree.errorBudget (p : ℕ) : SumTree → ℤ
  | .leaf _ => 0
  | .join l r =>
    let v := l.reduce p + r.reduce p
    |binaryRound p v - v| + (l.errorBudget p + r.errorBudget p)

/-- With a fixed common grid and sufficient exponent range, this absolute
    error bound also applies when intermediate additions lose precision. -/
theorem tree_error (p : ℕ) (t : SumTree) :
    |t.reduce p - t.total| ≤ t.errorBudget p := by
  induction t with
  | leaf => simp [SumTree.reduce, SumTree.total, SumTree.errorBudget]
  | join l r hl hr =>
    let v := l.reduce p + r.reduce p
    change |binaryRound p v - (l.total+r.total)| ≤
      |binaryRound p v - v| + (l.errorBudget p + r.errorBudget p)
    calc
      _ = |(binaryRound p v-v) + ((l.reduce p-l.total) + (r.reduce p-r.total))| := by
        congr 1; dsimp [v]; ring
      _ ≤ _ := (abs_add_le ..).trans (add_le_add_left
        ((abs_add_le ..).trans (add_le_add hl hr)) _)

/-- Independent bank errors propagate with their own nonnegative population
    weights. The last term also accounts for rounding during scaling, merging
    and the destination cast. All errors here are measured in real units. -/
theorem finalized_error (p e a b wp we committed : ℝ) (hp : 0 ≤ wp) (he : 0 ≤ we) :
    |committed - (wp*p+we*e)| ≤
      |committed-(wp*a+we*b)| + (wp*|a-p| + we*|b-e|) := by
  calc
    _ = |(committed-(wp*a+we*b)) + (wp*(a-p)+we*(b-e))| := by congr 1; ring
    _ ≤ _ := (abs_add_le ..).trans (add_le_add_left (by
      simpa [abs_mul, abs_of_nonneg hp, abs_of_nonneg he] using
        abs_add_le (wp*(a-p)) (we*(b-e))) _)

open Finset
open scoped BigOperators

/-- With at most h contributing ranks, scaling their coordinates into
    [-limit/h, limit/h] keeps every exact partial sum in range. -/
theorem headroom {r : ℕ} (part : Finset (Fin r)) (x : Fin r → ℝ)
    {limit h : ℝ} (hl : 0 ≤ limit) (hh : 0 < h) (hc : (part.card : ℝ) ≤ h)
    (hx : ∀ i ∈ part, |x i| ≤ limit/h) : |∑ i ∈ part, x i| ≤ limit := by
  calc
    _ ≤ ∑ i ∈ part, |x i| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ part, limit/h := sum_le_sum hx
    _ = (part.card : ℝ)*(limit/h) := by simp [nsmul_eq_mul]
    _ ≤ h*(limit/h) := mul_le_mul_of_nonneg_right hc (div_nonneg hl hh.le)
    _ = limit := by field_simp

/-- The exact weighted SUM lies inside the scaled rank extrema. -/
theorem weighted_extrema {r : ℕ} (x : Fin r → ℝ) {lo hi w : ℝ}
    (hw : 0 ≤ w) (hl : ∀ i, lo ≤ x i) (hh : ∀ i, x i ≤ hi) :
    (r*lo)*w ≤ (∑ i, x i)*w ∧ (∑ i, x i)*w ≤ (r*hi)*w := by
  have lower := sum_le_sum (s := (univ : Finset (Fin r))) (fun i _ => hl i)
  have upper := sum_le_sum (s := (univ : Finset (Fin r))) (fun i _ => hh i)
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at lower upper
  exact ⟨mul_le_mul_of_nonneg_right lower hw, mul_le_mul_of_nonneg_right upper hw⟩

/-- Projecting a floating approximation into valid extrema cannot increase
    its absolute error, provided the implemented bounds enclose the truth. -/
theorem project_error {lo hi target : ℝ} (x : ℝ) (hl : lo ≤ target) (hh : target ≤ hi) :
    |max lo (min hi x)-target| ≤ |x-target| := by
  rcases le_total x lo with below | inside
  · rw [min_eq_right (below.trans (hl.trans hh)), max_eq_left below,
        abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]
    linarith
  · rcases le_total x hi with within | above
    · rw [min_eq_right within, max_eq_right inside]
    · rw [min_eq_left above, max_eq_right (hl.trans hh),
          abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)]
      linarith

/-- A subnormal bank reduced on a wide enough grid has at most half a
    destination unit of final error, with even parity on exact ties. -/
theorem rounded_mean (t : SumTree) {p : ℕ} (hp : 0 < p)
    (h : t.mass < 2^p) {ranks : ℤ} (hr : 0 < ranks) :
    2 * |roundEven (t.reduce p) ranks*ranks - t.total| ≤ ranks := by
  rw [exact_tree hp t h]
  exact nearest _ hr

/-- The old prescale loses a replicated minimum subnormal. -/
example : roundEven 1 2 + roundEven 1 2 = 0 ∧ roundEven (1+1) 2 = 1 := by decide

/-- Rounding the policy and entropy banks separately can lose a whole
    destination unit, even when neither constituent rounding is incorrect. -/
example : roundEven 1 2 + roundEven 1 4 = 0 ∧ roundEven (2+1) 4 = 1 := by decide

end StateSetFormal.CollectiveRounding
