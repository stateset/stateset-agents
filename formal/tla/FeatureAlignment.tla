-------------------------- MODULE FeatureAlignment --------------------------
EXTENDS Integers, FiniteSets
(* Three target rows, two source rows; any subset may be available. Each
   row is in one of two bins at a distance of sqrt(2 log 2) bandwidths.
   Gram entries in units of one half are 2 on a match and 1 otherwise. *)
Rows == 1..5
Source == 1..2
VARIABLES one, available, seen, xx, yy, xy, covS, covR, done
vars == <<one,available,seen,xx,yy,xy,covS,covR,done>>
S == seen \cap Source
R == seen \ Source
Ones(A) == Cardinality(A \cap one)
Unlikes(i,A) == IF i \in one THEN Cardinality(A \ one) ELSE Ones(A)
Gram(i,A) == 2*Cardinality(A)-Unlikes(i,A)
Init == /\ one \in SUBSET Rows /\ available \in SUBSET Rows /\ seen={}
        /\ xx=0 /\ yy=0 /\ xy=0 /\ covS=0 /\ covR=0 /\ done=FALSE
Add(i) == /\ ~done /\ i \in available \ seen /\ seen'=seen \cup {i}
          /\ IF i \in Source THEN
               /\ xx'=xx+2+2*Gram(i,S) /\ xy'=xy+Gram(i,R)
               /\ covS'=covS+Unlikes(i,S) /\ UNCHANGED <<yy,covR>>
             ELSE
               /\ yy'=yy+2+2*Gram(i,R) /\ xy'=xy+Gram(i,S)
               /\ covR'=covR+Unlikes(i,R) /\ UNCHANGED <<xx,covS>>
          /\ UNCHANGED <<one,available,done>>
Finish == /\ ~done /\ seen=available /\ done'=TRUE
          /\ UNCHANGED <<one,available,seen,xx,yy,xy,covS,covR>>
Next == Finish \/ (\E i \in Rows : Add(i))
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
n == Cardinality(S)
m == Cardinality(R)
MD == xx*m*m+yy*n*n-2*xy*n*m
(* Covariance numerator from centering all observations about their mean,
   scaled by population squared to keep TLC arithmetic integral. *)
Centered(A) == LET a == Ones(A)
                   b == Cardinality(A)-a
               IN a*b*b + b*a*a
Correct == /\ seen \subseteq available
           /\ MD=2*(Ones(S)*m-Ones(R)*n)^2
           /\ n*covS=Centered(S) /\ m*covR=Centered(R)
           /\ done => seen=available
Finished == <>done
=============================================================================
