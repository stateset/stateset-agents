import Returns
import GradientBank

namespace StateSetFormal.DecisionReturns
noncomputable section

def value (g : ℝ) : List ℝ → ℝ
  | [] => 0
  | r :: rs => r + g * value g rs

def forwardSum (g power : ℝ) : List ℝ → ℝ
  | [] => 0
  | r :: rs => power * r + forwardSum g (power*g) rs

/-- The backward recurrence is the sum of rewards weighted by the
successive powers of the discount from this turn onward. -/
theorem forward_return (g power : ℝ) (rs : List ℝ) :
    forwardSum g power rs = power * value g rs := by
  induction rs generalizing power with
  | nil => simp [forwardSum, value]
  | cons r rs ih => rw [forwardSum, value, ih]; ring

/-- A recording cutoff omits the discounted return of its unobserved suffix.
    For a positive discount two possible continuations can therefore require
    different labels for exactly the same recorded decisions. -/
theorem cutoff_return (g : ℝ) (seen future : List ℝ) :
    value g (seen ++ future) = value g seen + g^seen.length * value g future := by
  induction seen with
  | nil => simp [value]
  | cons r rs ih =>
    change r+g*value g (rs++future) = r+g*value g rs + g^(rs.length+1)*value g future
    rw [ih, pow_succ]
    ring

theorem cutoff_ambiguity {g : ℝ} (hg : 0 < g) (seen : List ℝ) :
    value g (seen ++ [0]) < value g (seen ++ [1]) := by
  simp [cutoff_return, value, pow_pos hg]

def scan (g : ℝ) : List ℝ → List ℝ
  | [] => []
  | r :: rs => let tail := scan g rs; (r+g*tail.headD 0) :: tail

theorem head_return (g : ℝ) (rs : List ℝ) : (scan g rs).headD 0 = value g rs := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    change r + g*(scan g rs).headD 0 = r + g*value g rs
    rw [ih]

theorem turn_return (g : ℝ) (rs : List ℝ) (i : ℕ) :
    ((scan g rs)[i]?).getD 0 = value g (rs.drop i) := by
  induction rs generalizing i with
  | nil => simp [scan, value]
  | cons r rs ih =>
    cases i with
    | zero =>
      simp only [List.drop_zero, scan, List.getElem?_cons_zero, Option.getD_some, value, head_return]
    | succ i => simpa [scan] using ih i

/-- Selecting assistant turn indices after computing the returns preserves
all intervening user/tool rewards and discounts across those turns. -/
theorem project_actions (g : ℝ) (rs : List ℝ) (positions : List ℕ) :
    positions.map (fun i => ((scan g rs)[i]?).getD 0) =
      positions.map (fun i => forwardSum g 1 (rs.drop i)) := by
  simp [turn_return, forward_return]

theorem split_return (g : ℝ) (a b : List ℝ) :
    value g (a++b) = value g a + g^a.length * value g b := by
  induction a with
  | nil => simp [value]
  | cons r a ih => simp [value, ih, pow_succ]; ring

/-- Affine reward normalization changes each return by a shift that
depends on the number of remaining turns. -/
theorem affine_return (g a b : ℝ) (rs : List ℝ) :
    value g (rs.map (fun r => a*r+b)) =
      a*value g rs + b*value g (rs.map fun _ => 1) := by
  induction rs with
  | nil => simp [value]
  | cons r rs ih => simp [value, ih]; ring

def exactNum (m d r : ℤ) : List ℤ → ℤ
  | [] => r
  | a :: rs => r*d^(rs.length+1) + m*exactNum m d a rs

/-- For n+1 integral reward ticks and rational discount m/d, multiplying
the return by d^n suffices to represent it exactly on an integer grid. -/
theorem exact_horizon (m d : ℤ) (hd : 0 < d) (r : ℤ) (rs : List ℤ) :
    value ((m:ℝ)/d) ((r::rs).map (fun z : ℤ => (z:ℝ))) =
      (exactNum m d r rs : ℝ) / (d:ℝ)^rs.length := by
  have hd' : (d:ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  induction rs generalizing r with
  | nil => simp [value, exactNum]
  | cons a rs ih =>
    change (r:ℝ) + (m:ℝ)/d * value _ ((a::rs).map (fun z : ℤ => (z:ℝ))) = _
    rw [ih]
    simp [exactNum, pow_succ]
    field_simp
    ring

/-- Among free predictions at one observed conditioning history, the mean
action vector minimizes action error. Its excess error is precisely the
squared distance of the prediction to that mean, summed over coordinates. -/
theorem conditional_action_fit {n k : ℕ} (hn : n ≠ 0) (a : Fin n → Fin k → ℝ)
    (prediction : Fin k → ℝ) :
    let mean := fun j => (∑ i, a i j)/n
    (∑ i, ∑ j, (a i j-prediction j)^2)/n =
      (∑ i, ∑ j, (a i j-mean j)^2)/n + ∑ j, (prediction j-mean j)^2 := by
  dsimp only
  rw [Finset.sum_comm, Finset.sum_comm (f := fun i j => (a i j - (∑ h, a h j)/n)^2)]
  simp_rw [GradientBank.reward_mse_decomposition hn (fun i => a i _) (prediction _)]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, add_div, mul_div_cancel_left₀]
  exact_mod_cast hn

private theorem div_lower (p d : ℤ) (hd : 0 < d) :
    ((p/d : ℤ) : ℝ) ≤ (p : ℝ)/d := by
  apply (le_div_iff₀ (show (0:ℝ) < d by exact_mod_cast hd)).mpr
  exact_mod_cast Int.ediv_mul_le p hd.ne'

theorem grid_floor (p d : ℤ) (hd : 0 < d) : ⌊(p:ℝ)/d⌋ = p/d := by
  apply Int.floor_eq_iff.mpr
  refine ⟨div_lower p d hd, ?_⟩
  apply (div_lt_iff₀ (show (0:ℝ) < d by exact_mod_cast hd)).mpr
  exact_mod_cast Int.lt_ediv_add_one_mul_self p hd

/-- When an exact grid point is rounded to distinct outward endpoints,
both inequalities are strict. In particular a zero upper endpoint then
encloses a strictly negative return, preserving the sign of rounded zero. -/
theorem open_grid (p d : ℤ) (hd : 0 < d) (apart : p/d ≠ -((-p)/d)) :
    ((p/d : ℤ) : ℝ) < (p:ℝ)/d ∧ (p:ℝ)/d < (-((-p)/d) : ℤ) := by
  let x := (p:ℝ)/d
  have lo : ⌊x⌋ = p/d := grid_floor p d hd
  have hi : ⌈x⌉ = -((-p)/d) := by
    have h := grid_floor (-p) d hd
    rw [Int.cast_neg, neg_div, Int.floor_neg] at h
    exact neg_eq_iff_eq_neg.mp h
  rw [← lo, ← hi] at apart ⊢
  constructor
  · apply lt_of_le_of_ne (Int.floor_le x)
    intro eq
    apply apart
    have h := congrArg Int.ceil eq
    simpa using h
  · apply lt_of_le_of_ne (Int.le_ceil x)
    intro eq
    apply apart
    have h := congrArg Int.floor eq
    simpa using h

/-- One integer-grid recurrence encloses the exact result in grid units,
including negative discounts whose multiplication reverses the endpoints. -/
theorem outward_step (r m d l h : ℤ) (hd : 0 < d) (x : ℝ)
    (hx : (l : ℝ) ≤ x) (hx' : x ≤ (h : ℝ)) :
    ((r + min (m*l) (m*h)/d : ℤ) : ℝ) ≤ r+(m:ℝ)*x/d ∧
    (r:ℝ)+(m:ℝ)*x/d ≤ ((r - (-max (m*l) (m*h))/d : ℤ) : ℝ) := by
  have dp : (0:ℝ) < d := by exact_mod_cast hd
  have lo : ((min (m*l) (m*h) : ℤ) : ℝ) ≤ m*x := by
    push_cast
    by_cases hm : (0:ℝ) ≤ m
    · exact (min_le_left _ _).trans (mul_le_mul_of_nonneg_left hx hm)
    · exact (min_le_right _ _).trans (mul_le_mul_of_nonpos_left hx' (le_of_not_ge hm))
  have hi : (m:ℝ)*x ≤ ((max (m*l) (m*h) : ℤ) : ℝ) := by
    push_cast
    by_cases hm : (0:ℝ) ≤ m
    · exact (mul_le_mul_of_nonneg_left hx' hm).trans (le_max_right _ _)
    · exact (mul_le_mul_of_nonpos_left hx (le_of_not_ge hm)).trans (le_max_left _ _)
  have low := (div_lower (min (m*l) (m*h)) d hd).trans (div_le_div_of_nonneg_right lo dp.le)
  have high := div_lower (-max (m*l) (m*h)) d hd
  rw [Int.cast_neg, neg_div] at high
  push_cast
  constructor
  · exact add_le_add_left low _
  · change _ ≤ (r:ℝ) - _
    linarith [div_le_div_of_nonneg_right hi dp.le]

theorem rounded_enclosure {round : ℝ → ℝ} (ordered : Monotone round)
    {lo x hi : ℝ} (a : lo ≤ x) (b : x ≤ hi) (same : round lo = round hi) :
    round x = round lo :=
  le_antisymm ((ordered b).trans_eq same.symm) (ordered a)

/-- Selecting a nearest representable number is monotone, on a nonuniform
grid as well. Deterministic selection suffices when distances tie. -/
theorem nearest_monotone {round : ℝ → ℝ}
    (closest : ∀ x y, |round x-x| ≤ |round y-x|) : Monotone round := by
  intro a b hab
  rcases hab.eq_or_lt with same | lt
  · rw [same]
  by_contra! reverse
  have l := (sq_le_sq).mpr (closest a b)
  have h := (sq_le_sq).mpr (closest b a)
  nlinarith [mul_pos (sub_pos.mpr lt) (sub_pos.mpr reverse)]

def grid (m d : ℤ) : List ℤ → ℤ × ℤ
  | [] => (0,0)
  | r :: rs => let (l,h) := grid m d rs
               (r+min (m*l) (m*h)/d, r-(-max (m*l) (m*h))/d)

/-- The executable integer recurrence encloses the exact return at every
finite horizon. The input rewards and both endpoints are measured in ticks. -/
theorem grid_trace (m d : ℤ) (hd : 0 < d) (rs : List ℤ) :
    let v := value ((m:ℝ)/d) (rs.map (fun r : ℤ => (r:ℝ)))
    ((grid m d rs).1 : ℝ) ≤ v ∧ v ≤ ((grid m d rs).2 : ℝ) := by
  induction rs with
  | nil => simp [grid, value]
  | cons r rs ih =>
    dsimp only at ih ⊢
    simp only [grid, List.map_cons, value]
    rw [div_mul_eq_mul_div]
    exact outward_step r m d _ _ hd _ ih.1 ih.2

/-- A scan that publishes matching rounded endpoints gives the same
rounded return as the independent forward polynomial at any positive scale. -/
theorem certified_return (m d : ℤ) (hd : 0 < d) (rs : List ℤ) (unit : ℝ)
    (hu : 0 < unit) {round : ℝ → ℝ} (ordered : Monotone round)
    (agree : round ((grid m d rs).1/unit) = round ((grid m d rs).2/unit)) :
    round (forwardSum ((m:ℝ)/d) (1/unit) (rs.map (fun r : ℤ => (r:ℝ)))) =
      round ((grid m d rs).1/unit) := by
  rw [forward_return, one_div_mul_eq_div]
  have h := grid_trace m d hd rs
  exact rounded_enclosure ordered (div_le_div_of_nonneg_right h.1 hu.le)
    (div_le_div_of_nonneg_right h.2 hu.le) agree

end
end StateSetFormal.DecisionReturns
