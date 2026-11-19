----------------------- MODULE RoundedGradientBank --------------------------
EXTENDS Integers
(* One parameter coordinate on an integer grid, with enough precision for
   the two SUMs and a single destination rounding after population weighting. *)
Ranks == 1..2
Samples == {<<0,0,0,0>>} \cup
           {<<p,e,c,t>> : p \in -1..1, e \in -1..1, c \in 1..2, t \in 2..3}
VARIABLES rows, seen, policy, entropy, finished, result
vars == <<rows, seen, policy, entropy, finished, result>>
Total(k) == rows[1][k] + rows[2][k]
Abs(x) == IF x < 0 THEN -x ELSE x
RoundEven(n,d) ==
    LET a == Abs(n)
        q == a \div d
        r == a % d
        z == IF 2*r > d \/ (2*r = d /\ q % 2 = 1) THEN q+1 ELSE q
    IN IF n < 0 THEN -z ELSE z

Init == /\ rows \in [Ranks -> Samples] /\ Total(3) > 0
        /\ seen = {} /\ policy = 0 /\ entropy = 0 /\ result = 0 /\ finished = FALSE
Collect(r) == /\ r \notin seen /\ seen' = seen \cup {r}
              /\ policy' = policy + rows[r][1]
              /\ entropy' = entropy + rows[r][2]
              /\ UNCHANGED <<rows,finished,result>>
Finalize == /\ seen = Ranks /\ ~finished
            /\ result' = RoundEven(policy*Total(4) + entropy*Total(3), Total(3)*Total(4))
            /\ finished' = TRUE /\ UNCHANGED <<rows,seen,policy,entropy>>
Next == (\E r \in Ranks : Collect(r)) \/ Finalize
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* This oracle uses the distance and parity of the answer. It computes
   the reference directly from the source rows and never calls RoundEven. *)
NearestEven == finished =>
    LET d == Total(3)*Total(4)
        n == Total(1)*Total(4) + Total(2)*Total(3)
        error == 2*Abs(result*d-n)
    IN seen = Ranks /\ (error < d \/ (error = d /\ result % 2 = 0))
Completes == <>finished
=============================================================================
