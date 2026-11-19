------------------------- MODULE GroupExpectation -------------------------
EXTENDS Integers, Sequences, FiniteSets

Rows == 1..3
Groups == {1, 2}
Slots == 1..2
Paths == {<<>>, <<1>>, <<2>>, <<1,1>>, <<1,2>>, <<2,1>>, <<2,2>>}
Group(i) == IF i = 2 THEN 2 ELSE 1
Adv(i) == IF i = 1 THEN 1 ELSE -1

(* Heads have action probabilities (1/3, 2/3) and (2/3, 1/3).
   All path masses share denominator nine, including the empty product. *)
Atom(a, h) == IF h = 0 THEN a ELSE 3 - a
Mass(path, h) ==
    (IF Len(path) >= 1 THEN Atom(path[1], h) ELSE 3) *
    (IF Len(path) = 2 THEN Atom(path[2], h) ELSE 3)

Truth(path, h) ==
    CASE path = <<>> -> 9
      [] path = <<1>> -> IF h = 0 THEN 3 ELSE 6
      [] path = <<2>> -> IF h = 0 THEN 6 ELSE 3
      [] path = <<1,1>> -> IF h = 0 THEN 1 ELSE 4
      [] path = <<2,2>> -> IF h = 0 THEN 4 ELSE 1
      [] OTHER -> 2

VARIABLES paths, seen, sums, squares, phase, head, steps, ratios, grads
vars == <<paths, seen, sums, squares, phase, head, steps, ratios, grads>>
Zero == <<0, 1>>

Init ==
    /\ paths \in [Rows -> Paths]
    /\ seen = {}
    /\ phase = "collect"
    /\ head = 0
    /\ steps = 0
    /\ sums = [g \in Groups |-> 0]
    /\ squares = sums
    /\ ratios = [i \in Rows |-> Zero]
    /\ grads = [i \in Rows |-> [j \in Slots |-> Zero]]

Add(i) ==
    LET g == Group(i)
        q == Mass(paths[i], 0)
    IN /\ phase = "collect"
       /\ i \notin seen
       /\ seen' = seen \cup {i}
       /\ sums' = [sums EXCEPT ![g] = @ + q]
       /\ squares' = [squares EXCEPT ![g] = @ + q*q]
       /\ UNCHANGED <<paths, phase, head, steps, ratios, grads>>

Empty == \A i \in Rows : Len(paths[i]) = 0

Seal ==
    /\ phase = "collect"
    /\ seen = Rows
    /\ phase' = IF Empty THEN "done" ELSE "learn"
    /\ UNCHANGED <<paths, seen, sums, squares, head, steps, ratios, grads>>

(* Each ratio and gradient is a pair of integer numerator and denominator.
   The three rows have equal weight under the default sequence-mean loss. *)
Update(h) ==
    /\ phase = "learn"
    /\ head' = h
    /\ steps' = steps + 1
    /\ phase' = IF steps' = 2 THEN "done" ELSE "learn"
    /\ ratios' = [i \in Rows |-> <<Mass(paths[i], h) * sums[Group(i)], squares[Group(i)]>>]
    /\ grads' = [i \in Rows |-> [j \in Slots |->
           LET r == ratios'[i]
               clipped == (Adv(i) > 0 /\ 2*r[1] > 3*r[2]) \/
                          (Adv(i) < 0 /\ 2*r[1] < r[2])
           IN IF j > Len(paths[i]) \/ clipped THEN Zero ELSE <<-Adv(i)*r[1], 3*r[2]>>]]
    /\ UNCHANGED <<paths, seen, sums, squares>>

Next == (\E i \in Rows : Add(i)) \/ Seal \/ (\E h \in {0, 1} : Update(h))
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independently reconstruct both group aggregates from the probability table. *)
Q(i) == IF i \in seen THEN Truth(paths[i], 0) ELSE 0
S(g) == IF g = 1 THEN Q(1) + Q(3) ELSE Q(2)
Z(g) == IF g = 1 THEN Q(1)*Q(1) + Q(3)*Q(3) ELSE Q(2)*Q(2)
Frozen == \A g \in Groups : sums[g] = S(g) /\ squares[g] = Z(g)

Weights == steps > 0 => \A i \in Rows :
    LET r == ratios[i]
    IN r[2] > 0 /\ r[1]*Z(Group(i)) = Truth(paths[i], head)*S(Group(i))*r[2]

Gradient == steps > 0 => \A i \in Rows, j \in Slots :
    LET n == Truth(paths[i], head)*S(Group(i))
        d == Z(Group(i))
        a == Adv(i)
        slope == grads[i][j]
        active == j <= Len(paths[i]) /\ (IF a = 1 THEN 2*n <= 3*d ELSE 2*n >= d)
    IN slope[2] > 0 /\ 3*d*slope[1] = slope[2]*(IF active THEN -a*n ELSE 0)

Cadence == phase = "done" => steps = (IF Empty THEN 0 ELSE 2)
Done == <>(phase = "done")
=============================================================================
