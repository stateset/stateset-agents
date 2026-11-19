-------------------------- MODULE TransientBellman --------------------------
EXTENDS Integers, FiniteSets
(* Two equally weighted outcomes per state, zero denotes termination. The
   discount is one. All iterate numerators use the exact denominator 2^h. *)
S == 1..2
Pairs == {<<a,b>> : a \in 0..2, b \in 0..2}
VARIABLES rows,r,u,h,y,surv
vars == <<rows,r,u,h,y,surv>>
Num(s,j) == Cardinality({i \in 1..2 : rows[s][i]=j})
P(x,s) == Num(s,1)*x[1]+Num(s,2)*x[2]
Abs(x) == IF x<0 THEN -x ELSE x
Max(x,z) == IF x<z THEN z ELSE x
Init == /\ rows \in [S -> Pairs] /\ r \in [S -> {-1,1}] /\ u \in [S -> {-1,1}]
        /\ h=0 /\ y=u /\ surv= [s \in S |-> 1]
Step == /\ h<3 /\ h'=h+1 /\ y'= [s \in S |-> 2^(h+1)*r[s]+P(y,s)]
        /\ surv'= [s \in S |-> P(surv,s)] /\ UNCHANGED <<rows,r,u>>
Spec == Init /\ [][Step]_vars /\ WF_vars(Step)
Scale == 2^h
B == Max(surv[1],surv[2])
Residual == Max(Abs(y[1]-Scale*u[1]),Abs(y[2]-Scale*u[2]))
D == (2-Num(1,1))*(2-Num(2,2))-Num(1,2)*Num(2,1)
V(s) == 2*(r[s]*(2-Num(3-s,3-s))+Num(s,3-s)*r[3-s])
Bound == B < Scale => /\ D>0
    /\ \A s \in S : (Scale-B)*Abs(D*u[s]-V(s)) <= D*Residual
ReachesStop(s) == Num(s,0)>0 \/ (Num(s,3-s)>0 /\ Num(3-s,0)>0)
Coverage == h>=2 => ((B<Scale) <=> (\A s \in S: ReachesStop(s)))
Completion == <>(h=3)
=============================================================================
