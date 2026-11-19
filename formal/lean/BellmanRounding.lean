import CollectiveRounding
import LoggedBellman

/-! A local rounding cell for a Bellman multiply-add. The final cell can use
the virtual power of two above the largest finite value as its outer endpoint. -/
namespace StateSetFormal.BellmanRounding
noncomputable section

/-- In exact arithmetic the clipped endpoint and its reflected probe coincide. -/
theorem reflected_clip (a b x : ℝ) :
    min (max a (x + (x - min (max a x) b))) b = min (max a x) b := by
  simp only [min_def, max_def]
  split_ifs <;> linarith

/-- For each narrow critic precision, take r = 1 + 2⁻ᵖ, v = 2⁻⁽ᵖ⁺¹⁾
and g = 1 - 2⁻⁵³. In units of 2⁻⁽ᵖ⁺⁵⁴⁾ the exact target is n, the narrow
spacing is 2^54 and the binary64 spacing is 2^(p+2). Rounding twice
chooses the even upper neighbor while direct rounding chooses the lower. -/
theorem double_round_witness {p : ℕ} (hp : p = 7 ∨ p = 10 ∨ p = 23) :
    let n : ℤ := 2^(p+54) + 3*2^53 - 1
    let fine : ℤ := 2^(p+2)
    CollectiveRounding.roundEven n (2^54) = 2^p+1 ∧
      CollectiveRounding.roundEven (CollectiveRounding.roundEven n fine * fine) (2^54)
        = 2^p+2 := by
  rcases hp with rfl | rfl | rfl <;> decide

/-- The tie endpoint encodes the parity of the two adjacent significands. -/
def choose (lo hi tie x : ℝ) : ℝ :=
  if x < (lo + hi)/2 then lo else if (lo + hi)/2 < x then hi else tie

private theorem endpoints {lo hi tie : ℝ} (ht : tie = lo ∨ tie = hi) (x : ℝ) :
    choose lo hi tie x = lo ∨ choose lo hi tie x = hi := by
  unfold choose
  split_ifs <;> simp_all

/-- Use the sign at the midpoint only when a second endpoint was found.
Moving partway into the selected cell avoids signed-zero cancellation. -/
def corrected (d c b residual : ℝ) : ℝ :=
  if b ≠ c ∧ (if b > c then 0 < residual else residual < 0)
  then d - (d - b)/2 else d

/-- Wide accumulation and an optional intermediate cast may both land on a
narrow midpoint. Monotonicity and exact representation of the midpoint and
quarter-cell probes suffice to recover the one-shot rounding decision.
The residual passed to the selector need only have the correct strict signs. -/
theorem round_cell {lo hi tie x residual : ℝ} (hlt : lo < hi)
    (ht : tie = lo ∨ tie = hi) (wide cast : ℝ → ℝ)
    (hw : Monotone wide) (hc : Monotone cast)
    (wm : wide ((lo+hi)/2) = (lo+hi)/2) (cm : cast ((lo+hi)/2) = (lo+hi)/2)
    (cl : cast (((lo+hi)/2+lo)/2) = ((lo+hi)/2+lo)/2)
    (ch : cast (((lo+hi)/2+hi)/2) = ((lo+hi)/2+hi)/2)
    (neg : residual < 0 ↔ x < (lo+hi)/2) (pos : 0 < residual ↔ (lo+hi)/2 < x) :
    let d := wide x
    let c := choose lo hi tie (cast d)
    let b := choose lo hi tie (cast (d+(d-c)))
    choose lo hi tie (cast (corrected d c b residual)) = choose lo hi tie x := by
  dsimp only
  let m := (lo+hi)/2
  let d := wide x
  let c := choose lo hi tie (cast d)
  let b := choose lo hi tie (cast (d+(d-c)))
  change choose lo hi tie (cast (corrected d c b residual)) = choose lo hi tie x
  have lm : lo < m := by dsimp [m]; linarith
  have mh : m < hi := by dsimp [m]; linarith
  change cast ((m+lo)/2) = (m+lo)/2 at cl
  change cast ((m+hi)/2) = (m+hi)/2 at ch
  change wide m = m at wm
  have hm : 2*m = lo+hi := by dsimp [m]; ring
  have lower (y : ℝ) (hy : y ≤ (m+lo)/2) : cast y < m :=
    lt_of_le_of_lt ((hc hy).trans_eq cl) (by linarith)
  have upper (y : ℝ) (hy : (m+hi)/2 ≤ y) : m < cast y :=
    lt_of_lt_of_le (by linarith : m < (m+hi)/2) (ch.ge.trans (hc hy))
  have hb : b = lo ∨ b = hi := endpoints ht _
  rcases lt_trichotomy x m with hx | hx | hx
  · have hd : d ≤ m := (hw hx.le).trans_eq wm
    have hr := neg.mpr hx
    have goalLo : choose lo hi tie x = lo := if_pos hx
    rw [goalLo]
    rcases endpoints ht (cast d) with H | H
    · change c = lo at H
      have same : corrected d c b residual = d := by
        rcases hb with B | B <;> simp [corrected, H, B, hlt, hr.asymm]
      rw [same]; exact H
    · change c = hi at H
      have B : b = lo := if_pos (lower _ (by rw [H]; linarith))
      rw [corrected, B, H, if_pos (by simp [hlt.ne, hlt.asymm, hr])]
      exact if_pos (lower _ (by linarith))
  · have H : d = m := by simpa [d, hx] using wm
    have hr : residual = 0 := by
      rcases lt_trichotomy residual 0 with h | h | h
      · exact False.elim ((neg.mp h).ne hx)
      · exact h
      · exact False.elim ((pos.mp h).ne hx.symm)
    simp [corrected, hr, H, hx, m, cm]
  · have hd : m ≤ d := wm.ge.trans (hw hx.le)
    have hr := pos.mpr hx
    have goalHi : choose lo hi tie x = hi := by simp [choose, hx, hx.asymm]
    rw [goalHi]
    rcases endpoints ht (cast d) with H | H
    · change c = lo at H
      have beyond := upper (d+(d-c)) (by rw [H]; linarith)
      have B : b = hi := by simp [b, choose, beyond, beyond.asymm]
      rw [corrected, B, H, if_pos (by simp [hlt.ne', hlt, hr])]
      have inside := upper (d-(d-hi)/2) (by linarith)
      simp [choose, inside, inside.asymm]
    · change c = hi at H
      have same : corrected d c b residual = d := by
        rcases hb with B | B <;> simp [corrected, H, B, hlt.asymm, hr.asymm]
      rw [same]; exact H

/-- If the fused residual, expressed in units u, fits the working significand,
its rounding is exact. The difference tail can then restore the complete
Bellman residual, even when it alone decides the side of a midpoint. -/
theorem compensated_residual {p : ℕ} (hp : 0 < p) {z : ℤ} (hz : z.natAbs < 2^p)
    {u r g v m diff tail : ℝ} (split : diff + tail = r-m)
    (grid : (z : ℝ)*u = diff + g*v) :
    (CollectiveRounding.binaryRound p z : ℝ)*u + tail = r + g*v - m := by
  rw [CollectiveRounding.exact_significand hp hz, grid]
  linarith

/-- Direct rounding in a finite cell moves the exact target by at most
half the cell width, including either legal choice on an exact tie. -/
theorem half_spacing {lo hi tie x : ℝ} (ht : tie = lo ∨ tie = hi)
    (hl : lo ≤ x) (hh : x ≤ hi) :
    |choose lo hi tie x - x| ≤ (hi-lo)/2 := by
  unfold choose
  split_ifs with h₁ h₂
  · rw [abs_of_nonpos (sub_nonpos.mpr hl)]; linarith
  · rw [abs_of_nonneg (sub_nonneg.mpr hh)]; linarith
  · rcases ht with rfl | rfl
    · rw [abs_of_nonpos (sub_nonpos.mpr hl)]; linarith
    · rw [abs_of_nonneg (sub_nonneg.mpr hh)]; linarith

/-- A rounded self-loop can report zero squared TD error at a value that
misses the exact fixed point by the full half-spacing certificate. -/
theorem stationary_tie (u h g : ℝ) (hh : 0 < h) (hg : g < 1) :
    let r := (1-g)*u + h/2
    choose u (u+h) u (r+g*u) = u ∧ |u - r/(1-g)| = (h/2)/(1-g) := by
  dsimp only
  have mid : (1-g)*u + h/2 + g*u = (u+(u+h))/2 := by ring
  constructor
  · simp [choose, mid]
  · rw [LoggedBellman.sharp _ _ _ hg, mid,
      show u-(u+(u+h))/2 = -(h/2) by ring, abs_neg, abs_of_nonneg (by linarith)]

/-- A tail strictly smaller than the exact portion of the working grid
cannot change which side of zero a rounded fused residual lies on. Outside
that portion monotonicity suffices, even when the fused residual is inexact. -/
theorem residual_sign {z k : ℤ} {u tail : ℝ} (hu : 0 < u)
    (ht : |tail| < (k : ℝ)*u) (rnd : ℝ → ℝ) (mono : Monotone rnd)
    (grid : ∀ j : ℤ, -k ≤ j → j ≤ k → rnd ((j : ℝ)*u) = (j : ℝ)*u) :
    (rnd ((z : ℝ)*u) + tail < 0 ↔ (z : ℝ)*u + tail < 0) ∧
    (0 < rnd ((z : ℝ)*u) + tail ↔ 0 < (z : ℝ)*u + tail) := by
  have hkR : 0 < (k : ℝ) := pos_of_mul_pos_left ((abs_nonneg tail).trans_lt ht) hu.le
  have hk : 0 < k := by exact_mod_cast hkR
  have bounds := abs_lt.mp ht
  have hzlo := grid (-k) le_rfl (by omega)
  push_cast at hzlo
  have hzhi := grid k (by omega) le_rfl
  by_cases low : z < -k
  · have h : (z : ℝ) ≤ -(k : ℝ) := by exact_mod_cast low.le
    have hz := mul_le_mul_of_nonneg_right h hu.le
    have hq := (mono hz).trans_eq hzlo
    constructor <;> constructor <;> intro <;> linarith
  by_cases high : k < z
  · have h : (k : ℝ) ≤ (z : ℝ) := by exact_mod_cast high.le
    have hz := mul_le_mul_of_nonneg_right h hu.le
    have hq := hzhi.ge.trans (mono hz)
    constructor <;> constructor <;> intro <;> linarith
  simp only [grid z (by omega) (by omega), iff_self, and_self]

end
end StateSetFormal.BellmanRounding
