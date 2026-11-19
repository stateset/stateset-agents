import PolicyMath

/-! Contracts for the client-side layout and scaling of remote token losses.
The remote service supplies the learner log probability and sums token terms. -/
namespace StateSetFormal.ProviderLoss
open Finset Real Policy
open scoped BigOperators
noncomputable section

/-- River stores one entry per input ID, including a final inactive slot.
The context window keeps the first `kept` IDs of prompt plus sampled suffix. -/
def riverWire {α : Type*} [Zero α] (prompt kept : ℕ) (values : List α) : List α :=
  List.replicate (min prompt kept - 1) 0 ++ values.take (kept-prompt) ++ [0]

theorem wire_length {α : Type*} [Zero α] {p k : ℕ} (xs : List α)
    (hp : 0 < p) (hk : 0 < k) (h : k ≤ p + xs.length) :
    (riverWire p k xs).length = k := by
  simp [riverWire]
  omega

/-- At every prediction position the value belongs to precisely the following
retained token. Prompt targets and the last input slot get zero. -/
theorem wire_at {α : Type*} [Zero α] {p k i : ℕ} (xs : List α)
    (hp : 0 < p) (hi : i < k) (h : k ≤ p + xs.length) :
    (riverWire p k xs)[i]? =
      if p ≤ i+1 ∧ i+1 < k then xs[i+1-p]? else some 0 := by
  let pad := List.replicate (min p k - 1) (0 : α)
  let vals := xs.take (k-p)
  have plen : pad.length = min p k - 1 := by simp [pad]
  have vlen : vals.length = k-p := by simp [vals]; omega
  have both : (pad ++ vals).length = k-1 := by rw [List.length_append, plen, vlen]; omega
  change ((pad ++ vals) ++ [0])[i]? = _
  by_cases last : i+1 = k
  · rw [List.getElem?_append_right (by omega : (pad ++ vals).length ≤ i),
        show i-(pad ++ vals).length = 0 by omega]
    simp [last]
  · rw [List.getElem?_append_left (by omega : i < (pad ++ vals).length)]
    by_cases before : i < pad.length
    · rw [List.getElem?_append_left before]
      have hit : i < min p k - 1 := plen ▸ before
      simp [pad, hit, show ¬p ≤ i+1 by omega]
    · rw [List.getElem?_append_right (by omega : pad.length ≤ i)]
      have active : p ≤ i+1 ∧ i+1 < k := by omega
      rw [if_pos active, show i-pad.length = i+1-p by omega]
      exact List.getElem?_take_of_lt (by omega)

/-- Tinker supplies the target stream explicitly; prediction i is paired
with retained input token i+1. For a contiguous trajectory this equals the
River server-side next-token convention at every position with a target. -/
theorem shifted_target {α : Type*} (ids : List α) (i : ℕ) :
    (ids.drop 1)[i]? = ids[i+1]? := by
  simp [List.getElem?_drop, Nat.add_comm]

/-- The native SDK can include zero-credit but trainable tokens in its
denominator. A token mean omitting them has this exact relative scale. -/
theorem inactive_population (g : ℝ) {active inactive : ℝ}
    (ha : 0 < active) (hz : 0 ≤ inactive) :
    g / (active+inactive) = (active/(active+inactive)) * (g/active) := by
  field_simp [ha.ne', (add_pos_of_pos_of_nonneg ha hz).ne']
  ring

/-- The difference between averaging two turn means and treating their tokens
as one trajectory; either unequal means or unequal lengths alone is harmless. -/
theorem turn_mean_difference (x y : ℝ) {m n : ℝ} (hm : 0 < m) (hn : 0 < n) :
    (x+y)/2 - (m*x+n*y)/(m+n) = (n-m)*(x-y)/(2*(m+n)) := by
  field_simp [(add_pos hm hn).ne']
  ring

/-- Positive scaling of precomputed advantages commutes with every
supported additive River token objective, including PPO for either sign. -/
theorem surrogate_scale {s : ℝ} (hs : 0 ≤ s) (lo hi r a : ℝ) :
    surrogate lo hi r (s*a) = s * surrogate lo hi r a := by
  unfold surrogate
  rw [show r*(s*a) = s*(r*a) by ring,
      show max lo (min hi r)*(s*a) = s*(max lo (min hi r)*a) by ring,
      ← mul_min_of_nonneg _ _ hs]
  ring

theorem is_scale (s x q a : ℝ) :
    -(exp (x-q)) * (s*a) = s * (-(exp (x-q)) * a) := by ring

theorem cispo_scale (s cap r a x : ℝ) :
    -(min cap r) * (s*a) * x = s * (-(min cap r)*a*x) := by ring

/-- The sampled IS gradient with respect to the learner log probability. -/
theorem is_derivative (old a x : ℝ) :
    HasDerivAt (fun z => -exp (z-old) * a) (-exp (x-old) * a) x := by
  simpa using (((hasDerivAt_id x).sub_const old).exp.neg.mul_const a)

/-- Decoupled PPO moves a positive capped proximal-to-behavior ratio into
the advantage, leaving the learner-to-proximal clipping interval intact. -/
theorem decoupled_ppo (lo hi cur prox old a : ℝ) {cap : ℝ} (hc : 0 ≤ cap) :
    surrogate lo hi (exp (cur-prox)) (a * min (exp (prox-old)) cap) =
    min (exp (prox-old)) cap * surrogate lo hi (exp (cur-prox)) a := by
  rw [mul_comm a]
  exact surrogate_scale (le_min (exp_pos _).le hc) ..

theorem uncapped_ratio_chain (cur prox old : ℝ) :
    exp (cur-prox)*exp (prox-old) = exp (cur-old) := by
  rw [← exp_add]
  congr 1
  ring

/-- Product of conditional masses along a rollout under an autoregressive
policy. No stationarity or history-independence is assumed. -/
def pathMass {A : Type*} (p : List A → A → ℝ) (history : List A) : List A → ℝ
  | [] => 1
  | a :: tail => p history a * pathMass p (history ++ [a]) tail

theorem path_ratio {A : Type*} (p q : List A → A → ℝ) (history actions : List A) :
    pathMass (fun h a => p h a / q h a) history actions =
    pathMass p history actions / pathMass q history actions := by
  induction actions generalizing history with
  | nil => simp [pathMass]
  | cons a tail ih => simp only [pathMass, ih, mul_div_mul_comm]

theorem path_mass_positive {A : Type*} (q : List A → A → ℝ)
    (support : ∀ h a, 0 < q h a) (history actions : List A) :
    0 < pathMass q history actions := by
  induction actions generalizing history with
  | nil => simp [pathMass]
  | cons a tail ih => exact mul_pos (support _ _) (ih _)

/-- On any finite set of paths the product corrects both prefix visitation
and action probability. A complete stopped sample space gives expectations. -/
theorem path_change_measure {A : Type*} {n : ℕ} (p q : List A → A → ℝ)
    (support : ∀ h a, 0 < q h a) (history : List A) (paths : Fin n → List A)
    (reward : Fin n → ℝ) :
    ∑ t, pathMass q history (paths t) *
      pathMass (fun h a => p h a / q h a) history (paths t) * reward t =
    ∑ t, pathMass p history (paths t) * reward t := by
  apply sum_congr rfl
  intro t _
  rw [path_ratio]
  field_simp [(path_mass_positive q support history (paths t)).ne']

/-- Exact multiplicative sensitivity of IS to encoding the old log
probability. The premise bounds the conversion error on this action. -/
theorem wire_ratio_error {old encoded eps : ℝ} (cur : ℝ) (h : |encoded-old| ≤ eps) :
    exp (-eps) * exp (cur-old) ≤ exp (cur-encoded) ∧
    exp (cur-encoded) ≤ exp eps * exp (cur-old) := by
  rw [← exp_add, ← exp_add]
  have bounds := abs_le.mp h
  constructor <;> apply exp_le_exp.mpr <;> linarith

/-- Conditional token importance weighting changes the distribution of the
current action, keeping the sampler's prefix distribution. The residual to
the full target expectation is explicit for every finite two-step policy. -/
theorem prefix_measure_gap {n m : ℕ}
    (p q : Fin n → ℝ) (next old reward : Fin n → Fin m → ℝ)
    (support : ∀ i j, 0 < old i j) :
    (∑ i, p i * ∑ j, next i j * reward i j) -
      (∑ i, q i * ∑ j, old i j * (next i j / old i j) * reward i j) =
    ∑ i, (p i - q i) * ∑ j, next i j * reward i j := by
  rw [← sum_sub_distrib]
  apply sum_congr rfl
  intro i _
  have h (j : Fin m) : old i j * (next i j / old i j) * reward i j =
      next i j * reward i j := by field_simp [(support i j).ne']
  simp_rw [h, sub_mul]

/-- Already scaled token loss or gradient contributions sum to the same
logical update across every partition into transport microbatches. -/
theorem transport_sum (chunks : List (List ℝ)) :
    (chunks.map List.sum).sum = chunks.flatten.sum := by
  induction chunks with
  | nil => simp
  | cons chunk rest ih => simp only [List.map_cons, List.sum_cons, List.flatten_cons, List.sum_append, ih]

end
end StateSetFormal.ProviderLoss
