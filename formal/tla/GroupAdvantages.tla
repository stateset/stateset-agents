------------------------- MODULE GroupAdvantages -------------------------
EXTENDS Integers, FiniteSets
(* Four rewards on a common integer grid. The whole batch or one prompt
   supplies the variance; each numerator uses its own prompt's baseline. *)
Rows == 1..4
Sq(x) == x*x
Sum(f) == f[1]+f[2]+f[3]+f[4]
VARIABLES ticks, cut, offset, eps, seen, count, total, squares
vars == <<ticks,cut,offset,eps,seen,count,total,squares>>
Buckets == 0..2
Bucket(i) == IF i<=cut THEN 1 ELSE 2
In(i,b) == b=0 \/ Bucket(i)=b
X(i) == ticks[i]+offset
Init == /\ ticks \in [Rows -> -1..1] /\ cut \in {1,2,3}
        /\ offset \in {0,17} /\ eps \in {0,1}
        /\ seen={} /\ count=[b \in Buckets |-> 0] /\ total=count /\ squares=count
Add(i) == /\ i \notin seen /\ seen'=seen \cup {i}
          /\ count'=[b \in Buckets |-> count[b]+IF In(i,b) THEN 1 ELSE 0]
          /\ total'=[b \in Buckets |-> total[b]+IF In(i,b) THEN X(i) ELSE 0]
          /\ squares'=[b \in Buckets |-> squares[b]+IF In(i,b) THEN Sq(X(i)) ELSE 0]
          /\ UNCHANGED <<ticks,cut,offset,eps>>
Next == \E i \in Rows : Add(i)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Active(i,b) == i \in seen /\ In(i,b)
PairVariance(b) == Sum([i \in Rows |-> Sum([j \in Rows |->
    IF i<j /\ Active(i,b) /\ Active(j,b) THEN Sq(ticks[i]-ticks[j]) ELSE 0])])
Variance(b) == count[b]*squares[b]-Sq(total[b])
Numerator(i,b) == count[b]*X(i)-total[b]
Correct == \A b \in Buckets :
    /\ count[b]=Cardinality({i \in Rows : Active(i,b)})
    /\ Variance(b)=PairVariance(b)
    /\ (Variance(b)>Sq(count[b]*eps))=(PairVariance(b)>Sq(Cardinality({i \in Rows : Active(i,b)})*eps))
    /\ \A i \in Rows : Active(i,b) =>
         Numerator(i,b)=Sum([j \in Rows |-> IF Active(j,b) THEN ticks[i]-ticks[j] ELSE 0])
Completes == <>(seen=Rows)
=============================================================================
