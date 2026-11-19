--------------------------- MODULE ReplayStrata ---------------------------
EXTENDS Integers, FiniteSets, Sequences, TLC
(* Named task capacities 1, 2 and 3 plus two unlabeled entries. A uniform
   permutation restricted to eligible tasks gives a uniform partial round.
   Each pool's independent row permutation implements drawing without replacement. *)
Named == 1..3
Rows == 1..8
Seeds == Permutations(Named) \X Permutations(Named) \X Permutations(1..2)
         \X Permutations(1..3) \X Permutations(1..2)
Task(i) == CASE i=1 -> 1 [] i \in 2..3 -> 2 [] i \in 4..6 -> 3 [] OTHER -> 0
Pool(t) == {i \in Rows : Task(i)=t}
Contents(p) == {p[j] : j \in DOMAIN p}
Quota(p,t) == Cardinality(Contents(p) \cap Pool(t))
Active(p) == {t \in Named : Quota(p,t)<Cardinality(Pool(t))}
Lowest(p) == {t \in Active(p) : \A u \in Active(p) : Quota(p,t)<=Quota(p,u)}
Which(p,s) ==
  IF Active(p)={} THEN 0
  ELSE LET order == IF Len(p)<3 THEN s[1] ELSE s[2]
           next == CHOOSE j \in 1..3 : order[j] \in Lowest(p) /\
                     \A i \in 1..(j-1) : order[i] \notin Lowest(p)
       IN order[next]
Row(t,offset,s) == CASE t=1 -> 1 [] t=2 -> 1+s[3][offset]
                     [] t=3 -> 3+s[4][offset] [] OTHER -> 6+s[5][offset]
VARIABLES size, picked
vars == <<size,picked>>
Init == /\ size=0 /\ picked = [s \in Seeds |-> <<>>]
Draw == /\ size<8 /\ size'=size+1
        /\ picked'=[s \in Seeds |->
            LET t == Which(picked[s],s)
            IN Append(picked[s],Row(t,Quota(picked[s],t)+1,s))]
Shape == \A s \in Seeds : Len(picked[s])=size /\ Cardinality(Contents(picked[s]))=size
(* Reference inclusion law from completed rounds and remaining places. *)
Fraction(t) ==
  IF t=0 THEN <<IF size<=6 THEN 0 ELSE size-6,2>>
  ELSE IF size<=3 THEN <<size,3*t>>
  ELSE IF t=1 \/ size>=6 THEN <<1,1>>
  ELSE <<size-1,2*t>>
Inclusion == \A i \in Rows :
  LET p == Fraction(Task(i))
  IN p[2]*Cardinality({s \in Seeds : i \in Contents(picked[s])})=p[1]*Cardinality(Seeds)
RoundIndependence == size>=4 => \A t \in Named, u \in {2,3} :
  6*Cardinality({s \in Seeds : Task(picked[s][1])=t /\ Task(picked[s][4])=u})=Cardinality(Seeds)
Spec == Init /\ [][Draw]_vars /\ WF_vars(Draw)
Finished == <>(size=8)
=============================================================================
