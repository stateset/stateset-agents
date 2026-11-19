------------------------- MODULE GradientClipping ---------------------------
EXTENDS Integers, FiniteSets
(* Actor and critic have separate L2 caps. A reduction can read any nonempty
   chunk of the four coordinates, in any order. Epsilon is one model unit.
   Signed multiples of two different Pythagorean triples make the independent
   reference and the clipping divisions exact using integer cross-products. *)
VARIABLES a, b, cap, seen, energy, out, done
vars == <<a,b,cap,seen,energy,out,done>>
Heads == 1..2
Ix == 1..4
G == <<3*a,-4*a,5*b,12*b>>
Head(i) == IF i <= 2 THEN 1 ELSE 2
Sq(x) == x*x
Abs(x) == IF x < 0 THEN -x ELSE x
Part(i,S) == IF i \in S THEN Sq(G[i]) ELSE 0
Norm(h) == IF h = 1 THEN 5*Abs(a) ELSE 13*Abs(b)
Init == /\ a \in -2..2 /\ b \in -2..2 /\ cap \in {0,4,6,14,26}
        /\ seen = {} /\ energy = <<0,0>> /\ done = FALSE
        /\ out = [i \in Ix |-> <<0,1>>]
Collect ==
    \E chunk \in SUBSET (Ix \ seen) :
       /\ chunk # {}
       /\ seen' = seen \cup chunk
       /\ energy' = <<energy[1]+Part(1,chunk)+Part(2,chunk),
                      energy[2]+Part(3,chunk)+Part(4,chunk)>>
       /\ UNCHANGED <<a,b,cap,out,done>>
Clip ==
    /\ seen = Ix /\ ~done
    /\ LET divisor(h) == 1 + (CHOOSE n \in 0..26 : Sq(n) = energy[h])
           num(h) == IF cap < divisor(h) THEN cap ELSE divisor(h)
       IN out' = [i \in Ix |-> <<G[i]*num(Head(i)),divisor(Head(i))>>]
    /\ done' = TRUE /\ UNCHANGED <<a,b,cap,energy,seen>>
Next == Collect \/ Clip
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
PartialEnergy ==
    /\ energy[1] = Part(1,seen) + Part(2,seen)
    /\ energy[2] = Part(3,seen) + Part(4,seen)
Correct ==
    done => \A i \in Ix :
       LET d == Norm(Head(i))+1 IN
       IF d <= cap THEN out[i][1] = G[i]*out[i][2]
       ELSE out[i][1]*d = G[i]*cap*out[i][2]
Bounded ==
    done => \A h \in Heads :
       LET i == 2*h-1
       IN /\ out[i][2] = out[i+1][2]
          /\ Sq(out[i][1])+Sq(out[i+1][1]) <= Sq(cap*out[i][2])
Completes == <>done
=============================================================================
