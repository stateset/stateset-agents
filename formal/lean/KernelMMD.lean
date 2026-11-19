import OfflineMath

/-! BEAR's actor compares its actions at a state to repeated copies of one
observed action. The Dirac bounds specialize the empirical MMD² to this case. The general Gaussian result below permits two arbitrary finite samples. -/
namespace StateSetFormal.KernelMMD
open Finset Real
open scoped BigOperators
noncomputable section

def mmdDirac {n : ℕ} (k : Fin n → Fin n → ℝ) (w : Fin n → ℝ) :=
  (∑ i, ∑ j, k i j) / (n:ℝ)^2 + 1 - 2*(∑ i, w i)/n

private theorem lower {n : ℕ} [NeZero n] (k : Fin n → Fin n → ℝ) (w : Fin n → ℝ)
    (h : (∑ i, w i)^2 ≤ ∑ i, ∑ j, k i j) : 0 ≤ mmdDirac k w := by
  have hn : (0:ℝ) < n := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne n))
  have identity : mmdDirac k w =
      ((∑ i, ∑ j, k i j) + (n:ℝ)^2 - 2*(∑ i, w i)*n) / (n:ℝ)^2 := by
    unfold mmdDirac
    field_simp
    ring
  rw [identity]
  exact div_nonneg (by nlinarith [sq_nonneg ((n:ℝ) - ∑ i, w i)]) (sq_nonneg _)

theorem laplacian_nonnegative {X : Type*} [PseudoMetricSpace X] {n : ℕ} [NeZero n]
    (x : Fin n → X) (a : X) {s : ℝ} (hs : 0 < s) :
    0 ≤ mmdDirac (fun i j => exp (-dist (x i) (x j)/s))
      (fun i => exp (-dist (x i) a/s)) := by
  apply lower
  simp_rw [sq, sum_mul, mul_sum]
  apply sum_le_sum
  intro i _
  apply sum_le_sum
  intro j _
  rw [← exp_add, ← add_div]
  apply exp_le_exp.mpr
  apply div_le_div_of_nonneg_right _ hs.le
  linarith [dist_triangle (x i) a (x j), dist_comm a (x j)]

def normSq {d : ℕ} (u : Fin d → ℝ) := ∑ k, (u k)^2
def dot {d : ℕ} (u v : Fin d → ℝ) := ∑ k, u k * v k
def gaussian {d : ℕ} (u v : Fin d → ℝ) := exp (-normSq (u-v)/2)
def anchorWeight {d : ℕ} (u : Fin d → ℝ) := exp (-normSq u/2)

theorem gaussian_factor {d : ℕ} (u v : Fin d → ℝ) :
    gaussian u v = anchorWeight u * anchorWeight v * exp (dot u v) := by
  unfold gaussian anchorWeight dot normSq
  rw [← exp_add, ← exp_add]
  congr 1
  have identity (k : Fin d) : ((u-v) k)^2 = (u k)^2 + (v k)^2 - 2*(u k*v k) := by
    simp only [Pi.sub_apply]
    ring
  simp_rw [identity, sum_sub_distrib, sum_add_distrib, ← mul_sum]
  ring

private theorem gram_sum_nonnegative {n d : ℕ} (v : Fin n → Fin d → ℝ) :
    0 ≤ ∑ i, ∑ j, dot (v i) (v j) := by
  unfold dot
  have reorder (i : Fin n) : (∑ j, ∑ k, v i k * v j k) =
      ∑ k, ∑ j, v i k * v j k := sum_comm
  simp_rw [reorder]
  rw [sum_comm]
  apply sum_nonneg
  intro k _
  rw [sum_comm]
  simpa only [sq, sum_mul, mul_sum] using sq_nonneg (∑ i, v i k)

/-- Coordinates u are the policy actions minus the observed action, divided
    by bandwidth. The Gaussian cross-kernel at the observed action is w_i. -/
theorem gaussian_nonnegative {n d : ℕ} [NeZero n] (u : Fin n → Fin d → ℝ) :
    0 ≤ mmdDirac (fun i j => gaussian (u i) (u j)) (fun i => anchorWeight (u i)) := by
  let w := fun i => anchorWeight (u i)
  let v := fun i k => w i * u i k
  have term (i j : Fin n) : w i * w j + dot (v i) (v j) ≤ gaussian (u i) (u j) := by
    rw [gaussian_factor]
    have factor : dot (v i) (v j) = (w i*w j) * dot (u i) (u j) := by
      unfold dot
      rw [mul_sum]
      apply sum_congr rfl
      intro k _
      dsimp [v]
      ring
    rw [factor, ← mul_one_add]
    exact mul_le_mul_of_nonneg_left (by linarith [add_one_le_exp (dot (u i) (u j))])
      (mul_nonneg (exp_pos _).le (exp_pos _).le)
  apply lower
  have h := sum_le_sum (s := (univ : Finset (Fin n))) fun i _ =>
    sum_le_sum (s := (univ : Finset (Fin n))) fun j _ => term i j
  simp only [sum_add_distrib] at h
  have h2 := gram_sum_nonnegative v
  change (∑ i, w i)^2 ≤ _
  simp_rw [sq, sum_mul, mul_sum]
  linarith


/-- Every tensor power of the linear kernel has a positive semidefinite
Gram matrix. The induction incorporates a feature coordinate into each weight. -/
theorem dot_power_psd {I : Type*} [Fintype I] {d : ℕ} (x : I → Fin d → ℝ)
    (k : ℕ) (w : I → ℝ) : 0 ≤ ∑ i, ∑ j, w i*w j*(dot (x i) (x j))^k := by
  induction k generalizing w with
  | zero =>
    simp only [pow_zero, mul_one, ← mul_sum, ← sum_mul]
    exact mul_self_nonneg _
  | succ k ih =>
    have term (i j : I) : w i*w j*(dot (x i) (x j))^(k+1) =
        ∑ a, (w i*x i a)*(w j*x j a)*(dot (x i) (x j))^k := by
      rw [pow_succ, ← mul_assoc]
      unfold dot
      rw [mul_sum]
      apply sum_congr rfl
      intro a _
      ring
    simp_rw [term]
    have flip (i : I) : (∑ j, ∑ a, (w i*x i a)*(w j*x j a)*(dot (x i) (x j))^k) =
        ∑ a, ∑ j, (w i*x i a)*(w j*x j a)*(dot (x i) (x j))^k := sum_comm
    simp_rw [flip]
    rw [sum_comm]
    exact sum_nonneg fun a _ => ih (fun i => w i*x i a)

/-- Gaussian kernels are positive semidefinite for arbitrary finite signed
coefficients, allowing different sample sizes and duplicates in both domains. -/
theorem gaussian_psd {I : Type*} [Fintype I] {d : ℕ} (x : I → Fin d → ℝ) (c : I → ℝ) :
    0 ≤ ∑ i, ∑ j, c i*c j*gaussian (x i) (x j) := by
  let w := fun i => c i * anchorWeight (x i)
  have term (i j : I) : c i*c j*gaussian (x i) (x j) = (w i*w j)*exp (dot (x i) (x j)) := by
    rw [gaussian_factor]
    dsimp [w]
    ring
  simp_rw [term]
  have series (i j : I) : HasSum (fun k : ℕ => (w i*w j)*((dot (x i) (x j))^k/k.factorial))
      ((w i*w j)*exp (dot (x i) (x j))) := by
    rw [Real.exp_eq_exp_ℝ]
    exact (NormedSpace.expSeries_div_hasSum_exp ℝ _).mul_left _
  have h := hasSum_sum (s := (univ : Finset I)) fun i _ =>
    hasSum_sum (s := (univ : Finset I)) fun j _ => series i j
  rw [← h.tsum_eq]
  apply tsum_nonneg
  intro k
  simp only [← mul_div_assoc, ← sum_div]
  exact div_nonneg (dot_power_psd x k w) (by positivity)

/-- Derivative along any path of normalized pairwise offsets. For a source
coordinate with fixed target the offset derivative is 1/bandwidth. -/
theorem gaussian_direction {d : ℕ} (v : Fin d → ℝ → ℝ) (dv : Fin d → ℝ) (x : ℝ)
    (h : ∀ i, HasDerivAt (v i) (dv i) x) :
    HasDerivAt (fun y => exp (-(∑ i, (v i y)^2)/2))
      (exp (-(∑ i, (v i x)^2)/2) * (-(∑ i, v i x*dv i))) x := by
  have he := (((HasDerivAt.sum (u := univ) fun i _ => (h i).pow 2).neg).div_const 2).exp
  convert he using 1
  simp only [Nat.cast_ofNat, Nat.reduceSub, pow_one, mul_assoc, ← mul_sum]
  ring

theorem tail_magnitude {adj offset sigma : ℝ} (ha : 0 < adj) (ho : 0 < offset)
    (hs : 0 < sigma) (z : ℝ) :
    exp (z + log offset + log adj - log sigma) = adj*offset*exp z/sigma := by
  rw [exp_sub, exp_add, exp_add, exp_log ha, exp_log ho, exp_log hs]
  ring

theorem gaussian_symm {d : ℕ} (x y : Fin d → ℝ) : gaussian x y = gaussian y x := by
  unfold gaussian normSq
  simp only [Pi.sub_apply]
  have h (k : Fin d) : (x k-y k)^2 = (y k-x k)^2 := by ring
  simp_rw [h]

theorem gaussian_translate {d : ℕ} (x y a : Fin d → ℝ) :
    gaussian (x+a) (y+a) = gaussian x y := by simp [gaussian]

def empirical {m n d : ℕ} (s : Fin m → Fin d → ℝ) (r : Fin n → Fin d → ℝ) :=
    (∑ i, ∑ j, gaussian (s i) (s j))/(m:ℝ)^2 +
    (∑ i, ∑ j, gaussian (r i) (r j))/(n:ℝ)^2 -
    2*(∑ i, ∑ j, gaussian (s i) (r j))/((m:ℝ)*n)

theorem empirical_nonnegative {m n d : ℕ} (s : Fin m → Fin d → ℝ) (r : Fin n → Fin d → ℝ) :
    0 ≤ empirical s r := by
  let x := Sum.elim s r
  let w : Fin m ⊕ Fin n → ℝ := Sum.elim (fun _ => (1:ℝ)/m) (fun _ => -1/(n:ℝ))
  have h := gaussian_psd x w
  simp only [Fintype.sum_sum_type, x, w, Sum.elim_inl, Sum.elim_inr, sum_add_distrib,
      ← mul_sum] at h
  have cross : (∑ i, ∑ j, gaussian (r i) (s j)) = ∑ j, ∑ i, gaussian (s j) (r i) := by
    rw [sum_comm]
    apply sum_congr rfl
    intro j _
    apply sum_congr rfl
    intro i _
    exact gaussian_symm _ _
  rw [cross] at h
  unfold empirical
  convert h using 1
  ring


end
end StateSetFormal.KernelMMD

namespace StateSetFormal.CORAL
open Finset
open scoped BigOperators
noncomputable section

def covariance {n : ℕ} (x y : Fin n → ℝ) :=
  (∑ i, (x i-(∑ j, x j)/n)*(y i-(∑ j, y j)/n))/((n:ℝ)-1)

/-- Translation-free covariance formula using all ordered pairs of observations. -/
theorem covariance_pairs {n : ℕ} (hn : 1 < n) (x y : Fin n → ℝ) :
    covariance x y = (∑ i, ∑ j, (x i-x j)*(y i-y j))/(2*n*((n:ℝ)-1)) := by
  have hn' : (1:ℝ) < n := by exact_mod_cast hn
  have h0 : (n:ℝ) ≠ 0 := by linarith
  have den : (n:ℝ)-1 ≠ 0 := by linarith
  have scalar : (∑ i, (n:ℝ)*x i*y i) = (n:ℝ)*(∑ i, x i*y i) := by
    simp [mul_sum, mul_assoc]
  unfold covariance
  apply (div_eq_div_iff den (show (2:ℝ)*n*(n-1) ≠ 0 by positivity)).mpr
  simp only [sub_mul, mul_sub, sum_sub_distrib, ← sum_mul, ← mul_sum, sum_const,
    card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp only [scalar]
  field_simp
  ring

def loss {d : ℕ} (s r : Fin d → Fin d → ℝ) :=
  (∑ i, ∑ j, (s i j-r i j)^2)/(4*(d:ℝ)^2)

theorem loss_nonnegative {d : ℕ} (s r : Fin d → Fin d → ℝ) : 0 ≤ loss s r :=
  div_nonneg (sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _) (by positivity)

theorem gradient {d : ℕ} (s r : Fin d → Fin d → ℝ → ℝ) (ds dr : Fin d → Fin d → ℝ)
    (x : ℝ) (hs : ∀ i j, HasDerivAt (s i j) (ds i j) x)
    (hr : ∀ i j, HasDerivAt (r i j) (dr i j) x) :
    HasDerivAt (fun y => loss (fun i j => s i j y) (fun i j => r i j y))
      ((∑ i, ∑ j, 2*(s i j x-r i j x)*(ds i j-dr i j))/(4*(d:ℝ)^2)) x := by
  have entry (i j) := ((hs i j).sub (hr i j)).pow 2
  simpa [loss, pow_one] using ((HasDerivAt.sum fun i _ => HasDerivAt.sum fun j _ => entry i j).div_const (4*(d:ℝ)^2))

end
end StateSetFormal.CORAL
