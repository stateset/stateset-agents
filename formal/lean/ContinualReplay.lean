import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Finset.Prod
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Nat

/-! Counting laws for Algorithm R and the integer draws in continual GRPO replay. -/
namespace StateSetFormal.ContinualReplay
open Finset
open scoped BigOperators

/-- Give each retained identity one ticket, assigning the remaining tickets to a
    skip. Swapping the fresh identity with the ticket implements that transition. -/
def exchange (fresh ticket : ℕ) (s : Finset ℕ) :=
  s.map (Equiv.swap fresh ticket).toEmbedding

@[simp] theorem mem_exchange {n t i : ℕ} {s : Finset ℕ} :
    i ∈ exchange n t s ↔ Equiv.swap n t i ∈ s := by
  simp [exchange, mem_map_equiv, Equiv.symm_swap]

theorem exchange_involution (n t : ℕ) (s : Finset ℕ) :
    exchange n t (exchange n t s) = s := by
  ext i
  simp [Equiv.swap_apply_self]

theorem native_step {n t : ℕ} {s : Finset ℕ} (fresh : n ∉ s) :
    exchange n t s = if t ∈ s then insert n (s.erase t) else s := by
  ext i
  by_cases ht : t ∈ s <;> by_cases hi : i = n <;> by_cases hit : i = t <;>
    simp_all [Equiv.swap_apply_def]

private theorem predecessor {n t : ℕ} {s : Finset ℕ}
    (hs : s ⊆ range (n+1)) (ht : t ∈ range (n+1)) :
    exchange n t s ⊆ range n ↔ t ∉ s := by
  constructor
  · intro sub inside
    have : n ∈ exchange n t s := mem_exchange.mpr (by simpa using inside)
    simpa using sub this
  · intro outside i hi
    have h := mem_exchange.mp hi
    by_cases fresh : i = n
    · subst i
      simp [outside] at h
    by_cases ticket : i = t
    · subst i
      exact mem_range.mpr (by have := mem_range.mp ht; omega)
    rw [Equiv.swap_apply_of_ne_of_ne fresh ticket] at h
    exact mem_range.mpr (by have := mem_range.mp (hs h); omega)

/-- Number of equally likely draw histories ending in a given set, indexed
    backwards through the involution. During the initial fill there is one path. -/
def histories (k : ℕ) : ℕ → Finset ℕ → ℕ
  | 0, s => if s = ∅ then 1 else 0
  | n+1, s => if n < k then (if s = range (n+1) then 1 else 0)
    else ∑ t ∈ range (n+1), if exchange n t s ⊆ range n then histories k n (exchange n t s) else 0

/-- Every possible retained subset has exactly the same number of histories,
    for any capacity and any finite stream of distinct arrival identities. -/
theorem uniform_reservoir (k n : ℕ) (s : Finset ℕ)
    (hs : s ⊆ range n) (hc : s.card = min k n) :
    histories k n s = Nat.factorial (n-k) := by
  induction n generalizing s with
  | zero =>
    have : s = ∅ := by simpa using hs
    simp [this, histories]
  | succ n ih =>
    by_cases fill : n < k
    · have size : s.card = n+1 := by omega
      have full := eq_of_subset_of_card_le hs (by simp [size])
      simp [histories, fill, full, Nat.sub_eq_zero_of_le (by omega : n+1 ≤ k)]
    · have size : s.card = k := by omega
      rw [histories, if_neg fill]
      calc
        _ = ∑ t ∈ range (n+1), if t ∉ s then Nat.factorial (n-k) else 0 := by
          apply sum_congr rfl
          intro t ht
          by_cases old : exchange n t s ⊆ range n
          · rw [if_pos old, if_pos ((predecessor hs ht).mp old)]
            apply ih _ old
            simp [exchange, size, Nat.min_eq_left (by omega : k ≤ n)]
          · rw [if_neg old, if_neg (fun h => old ((predecessor hs ht).mpr h))]
        _ = ((range (n+1)) \ s).card * Nat.factorial (n-k) := by
          rw [← sum_filter]
          simp [sdiff_eq_filter]
        _ = Nat.factorial (n+1-k) := by
          rw [card_sdiff hs, card_range, size,
            show n+1-k = (n-k)+1 by omega, Nat.factorial_succ]

/-- Count the size-k reservoirs containing a specified uniform subbatch. This
    count depends only on the subbatch size, even after marginalizing storage. -/
theorem subbatch_completions (pool batch : Finset ℕ) (k : ℕ)
    (inside : batch ⊆ pool) (fits : batch.card ≤ k) :
    ((pool.powersetCard k).filter (fun s => batch ⊆ s)).card =
      (pool.card - batch.card).choose (k - batch.card) := by
  let extras := (pool \ batch).powersetCard (k-batch.card)
  have separate {a : Finset ℕ} (ha : a ∈ extras) : Disjoint batch a :=
    (subset_sdiff.mp (mem_powersetCard.mp ha).1).2.symm
  have eq : ((pool.powersetCard k).filter (fun s => batch ⊆ s)) =
      extras.image (batch ∪ ·) := by
    ext s
    constructor
    · intro h
      obtain ⟨valid, contains⟩ := mem_filter.mp h
      obtain ⟨sub, size⟩ := mem_powersetCard.mp valid
      exact mem_image.mpr ⟨s \ batch, mem_powersetCard.mpr
        ⟨sdiff_subset_sdiff sub (Subset.refl _), by rw [card_sdiff contains, size]⟩,
        union_sdiff_of_subset contains⟩
    · intro h
      obtain ⟨a, ha, rfl⟩ := mem_image.mp h
      obtain ⟨sub, size⟩ := mem_powersetCard.mp ha
      exact mem_filter.mpr ⟨mem_powersetCard.mpr ⟨union_subset inside
        (sub.trans sdiff_subset), by rw [card_union_of_disjoint (separate ha), size]; omega⟩,
        subset_union_left⟩
  have inj : Set.InjOn (batch ∪ ·) (↑extras : Set (Finset ℕ)) := by
    intro a ha b hb h
    simpa [union_sdiff_cancel_left (separate ha), union_sdiff_cancel_left (separate hb)]
      using congrArg (fun s => s \ batch) h
  rw [eq, card_image_of_injOn inj, card_powersetCard, card_sdiff inside]

/-- A cumulative integer-weight interval contains exactly its assigned mass. -/
theorem reward_tickets (start mass : ℕ) :
    (Ico start (start+mass)).card = mass := by simp

/-- A separate row ticket preserves its weight conditional on rounding up. -/
theorem joint_tickets (rem den start mass total : ℕ)
    (rate : rem ≤ den) (fits : start+mass ≤ total) :
    (((range den ×ˢ range total)).filter
      (fun p => p.1 < rem ∧ start ≤ p.2 ∧ p.2 < start+mass)).card = rem*mass := by
  have eq : (((range den ×ˢ range total)).filter
      (fun p => p.1 < rem ∧ start ≤ p.2 ∧ p.2 < start+mass)) =
      (range rem ×ˢ Ico start (start+mass)) := by
    ext p
    simp only [mem_filter, mem_product, mem_range, mem_Ico]
    omega
  rw [eq, card_product, card_range, reward_tickets]

/-- Sum of the requested replay counts across equally likely rounding tickets. -/
theorem rounded_total (whole rem den : ℕ) (h : rem ≤ den) :
    (∑ t ∈ range den, (whole + if t < rem then 1 else 0)) = den*whole+rem := by
  rw [sum_add_distrib, ← sum_filter]
  have subset : (range den).filter (fun t => t < rem) = range rem := by
    ext t
    simp only [mem_filter, mem_range]
    omega
  simp [subset]

/-- Count the equally likely partial-round subsets that include a particular task.
    With m other active tasks and r other places, the inclusion fraction is
    (r+1)/(m+1). -/
theorem partial_round (others : Finset ℕ) (task r : ℕ) (fresh : task ∉ others) :
    (others.card+1) * (((insert task others).powersetCard (r+1)).filter
      (fun selected => task ∈ selected)).card =
      ((insert task others).powersetCard (r+1)).card * (r+1) := by
  have includes : (((insert task others).powersetCard (r+1)).filter
      (fun selected => task ∈ selected)) = (others.powersetCard r).image (insert task) := by
    ext s
    rw [mem_filter, powersetCard_succ_insert fresh, mem_union]
    constructor
    · rintro ⟨h | h, hs⟩
      · exact (fresh ((mem_powersetCard.mp h).1 hs)).elim
      · exact h
    · intro h
      obtain ⟨a, _, e⟩ := mem_image.mp h
      exact ⟨Or.inr h, e ▸ mem_insert_self ..⟩
  have inj : Set.InjOn (insert task) (↑(others.powersetCard r) : Set (Finset ℕ)) := by
    intro a ha b hb he
    have absent {x : Finset ℕ} (hx : x ∈ others.powersetCard r) : task ∉ x :=
      fun h => fresh ((mem_powersetCard.mp hx).1 h)
    simpa [absent ha, absent hb] using congrArg (fun x : Finset ℕ => x.erase task) he
  rw [includes, card_image_of_injOn inj, card_powersetCard, card_powersetCard,
    card_insert_of_not_mem fresh]
  exact Nat.succ_mul_choose_eq ..

/-- Completed rounds give each task one entry until it is exhausted. Any two
    tasks still competing in the final partial round differ by at most one. -/
def quota (capacity rounds : ℕ) (extra : Bool) :=
  min capacity rounds + if rounds < capacity ∧ extra then 1 else 0

theorem quota_bound (capacity rounds : ℕ) (extra : Bool) :
    quota capacity rounds extra ≤ capacity := by
  unfold quota
  split <;> omega

theorem fair_quota {left right rounds : ℕ} (a b : Bool)
    (available : quota left rounds a < left) :
    quota right rounds b ≤ quota left rounds a + 1 := by
  unfold quota at *
  split at available <;> split <;> omega

/-- Taking at most the eligible population clips the mean to the same cap. -/
theorem capped_rounded_total (whole rem den cap : ℕ) (h : rem ≤ den) :
    (∑ t ∈ range den, min cap (whole + if t < rem then 1 else 0)) =
      min (den*cap) (den*whole+rem) := by
  by_cases small : cap ≤ whole
  · have point (t : ℕ) : min cap (whole + if t < rem then 1 else 0) = cap :=
      Nat.min_eq_left (by omega)
    simp_rw [point]
    rw [Nat.min_eq_left (by have := Nat.mul_le_mul_left den small; omega)]
    simp
  · have point (t : ℕ) : min cap (whole + if t < rem then 1 else 0) =
        (whole + if t < rem then 1 else 0) :=
      Nat.min_eq_right (by split <;> omega)
    simp_rw [point]
    have bound := Nat.mul_le_mul_left den (show whole+1 ≤ cap by omega)
    rw [Nat.mul_add, Nat.mul_one] at bound
    rw [rounded_total _ _ _ h, Nat.min_eq_right (by omega)]

end StateSetFormal.ContinualReplay
