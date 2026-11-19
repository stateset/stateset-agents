--------------------------- MODULE BayesianReward ---------------------------
EXTENDS Integers, FiniteSets

(* Each of two ensemble members contributes two equally likely dropout draws.
   Each conditional reward has mean in -1..1 and variance 1 or 4. *)
Rows == 1..4
Absent == <<0, 0>>
VARIABLES draws, n, sum, squares, noise
vars == <<draws, n, sum, squares, noise>>
Init == /\ draws = [i \in Rows |-> Absent]
        /\ n = 0 /\ sum = 0 /\ squares = 0 /\ noise = 0
Take(i, m, v) ==
    /\ draws[i] = Absent
    /\ draws' = [draws EXCEPT ![i] = <<m, v>>]
    /\ n' = n + 1
    /\ sum' = sum + m
    /\ squares' = squares + m*m
    /\ noise' = noise + v
Next == \E i \in Rows, m \in -1..1, v \in {1,4} : Take(i, m, v)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

M(i) == draws[i][1]
V(i) == draws[i][2]
Seen == {i \in Rows : draws[i] # Absent}
Sq(x) == x*x
Pair(i,j) == IF {i,j} \subseteq Seen THEN Sq(M(i)-M(j)) ELSE 0
Pairwise == Pair(1,2)+Pair(1,3)+Pair(1,4)+Pair(2,3)+Pair(2,4)+Pair(3,4)
Epistemic == n*squares - Sq(sum)
(* n squared is the shared denominator for epistemic and aleatoric terms. *)
Total == Epistemic + n*noise
Aggregates == /\ n = Cardinality(Seen)
              /\ sum = M(1)+M(2)+M(3)+M(4)
              /\ noise = V(1)+V(2)+V(3)+V(4)
              /\ Epistemic = Pairwise
              /\ n = 1 => Epistemic = 0
Within == 2*(Sq(M(1)-M(2))+Sq(M(3)-M(4)))
Between == Sq(M(1)+M(2)-M(3)-M(4))
Hierarchy == n = 4 => Total = Within + Between + 4*(V(1)+V(2)+V(3)+V(4))
Completes == <>(n = 4)
=============================================================================
