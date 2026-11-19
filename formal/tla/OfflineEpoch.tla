---------------------------- MODULE OfflineEpoch ----------------------------
EXTENDS Integers, FiniteSets
(* Visit one shuffled epoch, including the short final minibatch. Per-row
   observations can change after each optimizer update. Twelve units make
   every batch mean exact for the bounded domain. *)
VARIABLES n, capacity, signals, pending, observed, numerator, steps
vars == <<n,capacity,signals,pending,observed,numerator,steps>>
Rows == 1..n
Slots == 1..4
Min(a,b) == IF a <= b THEN a ELSE b
Sum(f) == f[1]+f[2]+f[3]+f[4]
Init == /\ n \in 1..4 /\ capacity \in 1..5
        /\ signals \in [Rows -> {-1,1}] /\ pending = Rows
        /\ observed = [i \in Slots |-> 0] /\ numerator = 0 /\ steps = 0
Take(batch) ==
    /\ batch \subseteq pending /\ batch # {}
    /\ Cardinality(batch) = Min(capacity,Cardinality(pending))
    /\ LET scores == [i \in Slots |-> IF i \in batch THEN signals[i]+steps ELSE 0]
           mean == (12*Sum(scores)) \div Cardinality(batch)
       IN /\ numerator' = numerator + Cardinality(batch)*mean
          /\ observed' = [i \in Slots |-> IF i \in batch THEN scores[i] ELSE observed[i]]
    /\ pending' = pending \ batch /\ steps' = steps+1
    /\ UNCHANGED <<n,capacity,signals>>
Next == \E batch \in SUBSET pending : Take(batch)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
WeightedMean == numerator = 12*Sum(observed)
Coverage == Cardinality(Rows \ pending) = Min(n,steps*capacity)
StepCount == pending = {} => steps = (n+capacity-1) \div capacity
Progress == <>(pending = {})
=============================================================================
