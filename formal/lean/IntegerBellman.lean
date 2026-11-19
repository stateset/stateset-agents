import BellmanRounding

namespace StateSetFormal.IntegerBellman
open CollectiveRounding

/-- Align exact reward and discounted successor on the larger dyadic denominator.
This equality in integers includes integer operands as the exponent-zero case. -/
theorem align (r g v : ℤ) (a b c : ℕ) :
    let m := max a (b+c)
    (r*2^(m-a) + g*v*2^(m-(b+c))) * (2^a*2^(b+c)) =
      (2:ℤ)^m * (r*2^(b+c) + g*v*2^a) := by
  dsimp only
  have hr : (2:ℤ)^(max a (b+c)-a)*2^a = 2^(max a (b+c)) := by
    rw [← pow_add, Nat.sub_add_cancel (le_max_left ..)]
  have hv : (2:ℤ)^(max a (b+c)-(b+c))*2^(b+c) = 2^(max a (b+c)) := by
    rw [← pow_add, Nat.sub_add_cancel (le_max_right ..)]
  linear_combination r*2^(b+c)*hr + g*v*2^a*hv

/-- The Python increment on a positive magnitude implements ties to even. -/
theorem quotient_increment (n : ℤ) {s : ℤ} (hs : 0 < s) :
    n/s + (if 2*(n%s) > s ∨ (2*(n%s) = s ∧ n/s%2 ≠ 0) then 1 else 0) =
      roundEven n s := by
  have := Int.emod_nonneg n hs.ne'
  simp only [roundEven]
  split_ifs <;> omega

private theorem nearest_multiple (n z : ℤ) {s : ℤ} (hs : 0 < s) :
    |roundEven n s*s-n| ≤ |z*s-n| := by
  by_contra! h
  have h' := nearest n hs
  have eq : roundEven n s = z := unique_nearest hs (by omega)
  rw [eq] at h
  exact (lt_irrefl _ h)

/-- Number of source-grid bits discarded, with subnormal spacing at least 2^floorBits. -/
def shift (precision floorBits n : ℕ) := max floorBits (n.log2+1-precision)
def representable (p floorBits y : ℕ) : Prop := 2^(shift p floorBits y) ∣ y

/-- The chosen grid point is at least as close as every representable magnitude,
including competitors in other binades where the grid spacing differs. -/
theorem grid_nearest {p : ℕ} (hp : 0 < p) (f n y : ℕ) (hy : representable p f y) :
    let s : ℤ := 2^shift p f n
    |roundEven n s*s-n| ≤ |(y:ℤ)-n| := by
  dsimp only
  have hs : (0:ℤ) < 2^shift p f n := by positivity
  by_cases le : shift p f n ≤ shift p f y
  · have hd : 2^(shift p f n) ∣ y := dvd_trans (pow_dvd_pow 2 le) hy
    obtain ⟨z, hz⟩ := hd
    have h : (y:ℤ) = z*(2:ℤ)^shift p f n := by exact_mod_cast (hz.trans (Nat.mul_comm ..))
    simpa only [← h] using nearest_multiple (n:ℤ) z hs
  · have hn : n ≠ 0 := by
      intro zero
      simp [shift, zero, Nat.sub_eq_zero_of_le hp] at le
    have hlog : y.log2 < n.log2 := by
      by_contra! h
      exact le (max_le_max_left f (Nat.sub_le_sub_right (Nat.add_le_add_right h 1) p))
    have hyb : y ≤ 2^n.log2 := by
      by_contra! h
      exact (not_le_of_gt hlog) ((Nat.le_log2 (by rintro rfl; exact (Nat.not_lt_zero _ h) : y ≠ 0)).mpr h.le)
    have hbn : 2^n.log2 ≤ n := Nat.log2_self_le hn
    have he : shift p f n ≤ n.log2 := by unfold shift at *; omega
    obtain ⟨z, hz⟩ := pow_dvd_pow 2 he
    have hb : ((2:ℕ)^n.log2:ℤ) = z*(2:ℤ)^shift p f n := by
      exact_mod_cast hz.trans (Nat.mul_comm ..)
    have h := nearest_multiple (n:ℤ) z hs
    rw [← hb] at h
    have h₁ : (y:ℤ) ≤ ((2:ℕ)^n.log2:ℤ) := by exact_mod_cast hyb
    have h₂ : ((2:ℕ)^n.log2:ℤ) ≤ (n:ℤ) := by exact_mod_cast hbn
    refine h.trans ?_
    rw [abs_of_nonpos (sub_nonpos.mpr h₂), abs_of_nonpos (sub_nonpos.mpr (h₁.trans h₂))]
    linarith

/-- A significand carry can reach the first value in the next binade. In
both cases the rounded value is on its own destination grid. -/
theorem closed_grid {p f k q : ℕ} (hp : 0 < p) (hf : f ≤ k) (hq : q ≤ 2^p) :
    representable p f (q*2^k) := by
  unfold representable
  rcases lt_or_eq_of_le hq with h | rfl
  · have bound : q*2^k < 2^(p+k) := by
      rw [pow_add]; exact Nat.mul_lt_mul_of_pos_right h (by positivity)
    have hl : (q*2^k).log2 < p+k := by
      by_cases hz : q*2^k = 0
      · simp [hz]; omega
      · exact (Nat.log2_lt hz).mpr bound
    have e : shift p f (q*2^k) ≤ k := by unfold shift; omega
    exact dvd_trans (pow_dvd_pow 2 e) (Nat.dvd_mul_left _ _)
  · rw [← pow_add]
    apply pow_dvd_pow
    simp only [shift, Nat.log2_two_pow]
    omega

theorem legal_result {p : ℕ} (hp : 0 < p) (f n : ℕ) :
    let k := shift p f n
    representable p f ((roundEven n ((2:ℤ)^k)).toNat*2^k) := by
  dsimp only
  let k := shift p f n
  have hlog : n.log2+1 ≤ p+k := by dsimp [k, shift]; omega
  have hb : n < 2^(n.log2+1) := by
    by_cases hz : n = 0
    · simp [hz]
    · exact (Nat.log2_lt hz).mp (by omega)
  have hn := hb.trans_le (Nat.pow_le_pow_right (by decide : 0 < 2) hlog)
  rw [pow_add] at hn
  have hs : (0:ℤ) < 2^k := by positivity
  have hd : (n:ℤ)/2^k < 2^p := (Int.ediv_lt_iff_lt_mul hs).mpr (by exact_mod_cast hn)
  have hround : roundEven n (2^k) ≤ (2:ℤ)^p := by
    simp only [roundEven]; split_ifs <;> omega
  apply closed_grid hp (le_max_left ..)
  exact_mod_cast Int.toNat_le_toNat hround

/-- Scaling both the numerator and the divisor preserves the integer tie decision. -/
theorem scale_quotient (n s : ℤ) {factor : ℤ} (hf : 0 < factor) :
    roundEven (factor*n) (factor*s) = roundEven n s := by
  simp only [roundEven, Int.mul_ediv_mul_of_pos _ _ hf, Int.mul_emod_mul_of_pos _ _ hf,
    show 2*(factor*(n%s)) = factor*(2*(n%s)) by ring]
  simp [hf.ne', hf]

def rounded (p f n : ℕ) : ℤ :=
  let s : ℤ := 2^shift p f n
  roundEven n s * s

private theorem scaled_log {n : ℕ} (hn : n ≠ 0) (j : ℕ) :
    (2^j*n).log2 = j+n.log2 := by
  have nz : 2^j*n ≠ 0 := Nat.mul_ne_zero (by positivity) hn
  have lower : 2^(j+n.log2) ≤ 2^j*n := by
    rw [pow_add]; exact Nat.mul_le_mul_left _ (Nat.log2_self_le hn)
  have upper : 2^j*n < 2^(j+n.log2+1) := by
    rw [Nat.add_assoc, pow_add]
    exact Nat.mul_lt_mul_of_pos_left Nat.lt_log2_self (by positivity)
  have := (Nat.le_log2 nz).mpr lower
  have := (Nat.log2_lt nz).mpr upper
  omega

/-- When the input denominator is coarser than the smallest output subnormal,
refining the grid to include all output competitors leaves the rounded value
unchanged. If the original significand already fits, both calculations are exact. -/
theorem finer_unit {p : ℕ} (hp : 0 < p) (j n : ℕ) :
    rounded p 0 (2^j*n) = (2:ℤ)^j * rounded p 0 n := by
  by_cases hn : n = 0
  · simp [hn, rounded, shift, roundEven, Nat.sub_eq_zero_of_le hp]
  have hlog := scaled_log hn j
  by_cases h : p ≤ n.log2+1
  · have hs : shift p 0 (2^j*n) = j+shift p 0 n := by unfold shift; rw [hlog]; omega
    simp only [rounded, hs, pow_add, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
    rw [scale_quotient _ _ (by positivity)]
    ring
  · let k := shift p 0 (2^j*n)
    have hk : k ≤ j := by dsimp [k, shift]; rw [hlog]; omega
    have one : shift p 0 n = 0 := by unfold shift; omega
    have multiple : (2:ℤ)^j*n = ((2:ℤ)^(j-k)*n)*2^k := by
      rw [show (2:ℤ)^j = 2^(j-k)*2^k by rw [← pow_add, Nat.sub_add_cancel hk]]
      ring
    change roundEven (↑(2^j*n):ℤ) (2^k)*2^k = (2:ℤ)^j*rounded p 0 n
    push_cast
    rw [multiple, exact _ (by positivity), ← multiple]
    simp [rounded, one, roundEven]

/-- Global nearestness also holds against values with finer denominators
than the exact numerator's initial integer lattice. -/
theorem finer_grid_nearest {p : ℕ} (hp : 0 < p) (j n y : ℕ)
    (hy : representable p 0 y) :
    |(2:ℤ)^j*(rounded p 0 n - n)| ≤ |(y:ℤ)-(2:ℤ)^j*n| := by
  have h := grid_nearest hp 0 (2^j*n) y hy
  change |rounded p 0 (2^j*n) - ↑(2^j*n)| ≤ |(y:ℤ) - ↑(2^j*n)| at h
  rw [finer_unit hp] at h
  push_cast at h
  simpa only [mul_sub] using h

/-- Clipping the encoded magnitude at the virtual overflow endpoint preserves
nearestness among all finite and virtual competitors in range. -/
theorem clipped_nearest {n cap q y : ℤ} (hy : y ≤ cap) (h : |q-n| ≤ |y-n|) :
    |min cap q - n| ≤ |y-n| := by
  by_cases hq : q ≤ cap
  · simpa [min_eq_right hq] using h
  rw [min_eq_left (by omega)]
  by_cases hn : cap ≤ n
  · rw [abs_of_nonpos (by omega : cap-n ≤ 0), abs_of_nonpos (by omega : y-n ≤ 0)]
    omega
  · rw [abs_of_nonneg (by omega : 0 ≤ cap-n)]
    rw [abs_of_nonneg (by omega : 0 ≤ q-n)] at h
    omega

end StateSetFormal.IntegerBellman
