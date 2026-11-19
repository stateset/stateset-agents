------------------------ MODULE RewardTokenWeights ---------------------------
EXTENDS Integers
(* Two possible completion rows, up to two scored tokens each. The third
   and fourth row entries are signed log-probability slopes. A missing
   second row also exercises the raw-reward single-token update. *)
Rows == 1..2
MaskedSentinel == 99
Samples == {<<r,1,a,b>> : r \in -1..1, a \in -1..1, b \in (-1..1 \cup {MaskedSentinel})}
           \cup {<<r,2,a,b>> : r \in -1..1, a \in -1..1, b \in -1..1}
Absent == <<0,0,MaskedSentinel,MaskedSentinel>>
VARIABLES rows, seen, n, reward, squares, slopes, product, finished, variance, numerator
vars == <<rows,seen,n,reward,squares,slopes,product,finished,variance,numerator>>
Init == /\ rows \in [Rows -> Samples \cup {Absent}] /\ rows[1] # Absent
        /\ seen = {} /\ n = 0 /\ reward = 0 /\ squares = 0 /\ slopes = 0
        /\ product = 0 /\ finished = FALSE /\ variance = 0 /\ numerator = 0
Collect(i) ==
    /\ i \notin seen /\ seen' = seen \cup {i}
    /\ LET row == rows[i]
           r == row[1]
           c == row[2]
           g == (IF c >= 1 THEN row[3] ELSE 0) + (IF c = 2 THEN row[4] ELSE 0)
       IN /\ n' = n+c /\ reward' = reward+c*r /\ squares' = squares+c*r*r
          /\ slopes' = slopes+g /\ product' = product+r*g
    /\ UNCHANGED <<rows,finished,variance,numerator>>
Finish == /\ seen = Rows /\ ~finished /\ finished' = TRUE
          /\ variance' = n*squares-reward*reward
          /\ numerator' = IF n = 1 THEN -product ELSE -(n*product-reward*slopes)
          /\ UNCHANGED <<rows,seen,n,reward,squares,slopes,product>>
Next == (\E i \in Rows : Collect(i)) \/ Finish
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent flattened-token oracle. Pairwise differences give the
   sample-variance numerator and the centered score derivative, without
   computing a batch mean or using the accumulated row statistics. *)
Row(k) == 1 + ((k-1) \div 2)
Pos(k) == 1 + ((k-1) % 2)
Present(k) == Pos(k) <= rows[Row(k)][2]
R(k) == rows[Row(k)][1]
G(k) == rows[Row(k)][2+Pos(k)]
RECURSIVE Sum(_, _)
Sum(f, k) == IF k = 0 THEN 0 ELSE f[k]+Sum(f,k-1)
Tokens(f) == Sum(f,4)
Count == Tokens([k \in 1..4 |-> IF Present(k) THEN 1 ELSE 0])
PairedVariance ==
    Tokens([k \in 1..4 |-> Tokens([j \in 1..4 |->
        IF Present(k) /\ Present(j) /\ j < k THEN (R(k)-R(j))*(R(k)-R(j)) ELSE 0])])
PairedScore ==
    Tokens([k \in 1..4 |-> Tokens([j \in 1..4 |->
        IF Present(k) /\ Present(j) THEN (R(j)-R(k))*G(k) ELSE 0])])
Correct == finished =>
    /\ n = Count /\ n > 0 /\ variance = PairedVariance /\ variance >= 0
    /\ numerator = IF Count = 1 THEN -R(1)*G(1) ELSE PairedScore
Completes == <>finished
=============================================================================
