--------------------------- MODULE AdamHistory ------------------------------
EXTENDS Integers, Sequences
(* One parameter in a three-update trainer. Other parameters allow a global
   update while this parameter has no gradient. beta1=beta2=1/2, lr=1/4,
   decay=1/2, epsilon=1. Working weights use units of 1/8; published weights
   are rounded to multiples of four working units. *)
Missing == 2
Inputs == {-1,0,1,Missing}
Sgn(x) == IF x < 0 THEN -1 ELSE IF x = 0 THEN 0 ELSE 1
(* Exact sign of a*sqrt(r)+b for a nonnegative integer radicand. *)
RootSign(a,r,b) ==
    IF a = 0 \/ r = 0 THEN Sgn(b)
    ELSE IF a > 0
         THEN IF b >= 0 THEN 1 ELSE Sgn(a*a*r-b*b)
         ELSE IF b <= 0 THEN -1 ELSE Sgn(b*b-a*a*r)
(* Compare the exact post-AdamW working value to j/2. Corrected moments
   are m/d and v/d. The denominator sqrt(v/d)+1 is strictly positive. *)
Side(w,m,v,d,j) ==
    LET a == 7*w-4*j IN RootSign(a,v*d,a*d-16*m)
Cell(w,m,v,d,z) ==
    LET lo == Side(w,m,v,d,2*z-1)
        hi == Side(w,m,v,d,2*z+1)
    IN /\ lo >= 0 /\ hi <= 0 /\ (lo = 0 \/ hi = 0 => z % 2 = 0)
Round(n,d) ==
    LET q == n \div d
        twice == 2*(n % d)
    IN IF twice < d \/ (twice=d /\ q%2=0) THEN q ELSE q+1
(* The finite implementation uses eight fractional bits for its root.
   Matches checks each rounded update against the exact irrational comparison
   in Cell, independently of this approximation. *)
Update(w,m,v,d) ==
    LET root == CHOOSE k \in 0..256 : k*k*d <= 65536*v /\ 65536*v < (k+1)*(k+1)*d
        den == d*(256+root)
    IN Round(7*w*den-4096*m,8*den)
Publish(w) == 4*Round(w,4)
VARIABLES inputs,i,t,m,v,w,visible,history,previous,phase,snap
vars == <<inputs,i,t,m,v,w,visible,history,previous,phase,snap>>
Init == /\ inputs \in [1..3 -> Inputs] /\ i=0 /\ t=0 /\ m=0 /\ v=0
        /\ w=8 /\ visible=8 /\ previous=8 /\ history = <<>> /\ phase="ready" /\ snap = <<0,0,0,8>>
Step ==
    /\ i<3 /\ phase="ready"
    /\ LET g == inputs[i+1] IN
       IF g=Missing THEN /\ UNCHANGED <<t,m,v,w,visible,history>>
       ELSE /\ t'=t+1 /\ m'=m+2^t*g /\ v'=v+2^t*g*g
            /\ history'=Append(history,g)
            /\ w'=Update(w,m',v',2^t'-1) /\ visible'=Publish(w')
    /\ i'=i+1 /\ previous'=w /\ phase'="boundary" /\ UNCHANGED <<inputs,snap>>
(* At every boundary, optionally lose all volatile optimizer state and recover
   it from the saved checkpoint and its separately published model weight. *)
Continue == /\ phase="boundary" /\ phase'="ready"
            /\ UNCHANGED <<inputs,i,t,m,v,w,visible,history,previous,snap>>
Save == /\ phase="boundary" /\ phase'="restart" /\ snap' = <<t,m,v,w>>
        /\ t'=0 /\ m'=0 /\ v'=0 /\ w'=0
        /\ UNCHANGED <<inputs,i,visible,history,previous>>
Restore == /\ phase="restart" /\ phase'="ready"
           /\ t'=snap[1] /\ m'=snap[2] /\ v'=snap[3]
           /\ w'=IF Publish(snap[4])=visible THEN snap[4] ELSE visible
           /\ UNCHANGED <<inputs,i,visible,history,previous,snap>>
Next == Step \/ Continue \/ Save \/ Restore
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
RECURSIVE RefMoment(_, _)
RefMoment(h,squared) ==
    IF Len(h)=0 THEN 0
    ELSE (IF squared THEN h[1]*h[1] ELSE h[1]) + 2*RefMoment(Tail(h),squared)
RECURSIVE Active(_)
Active(seq) ==
    IF Len(seq)=0 THEN <<>> ELSE
    IF seq[1]=Missing THEN Active(Tail(seq)) ELSE <<seq[1]>> \o Active(Tail(seq))
Expected == Active(SubSeq(inputs,1,i))
Matches == phase # "restart" =>
    /\ history=Expected /\ t=Len(Expected)
    /\ m=RefMoment(Expected,FALSE) /\ v=RefMoment(Expected,TRUE)
    /\ visible=Publish(w)
    /\ IF i=0 THEN w=8 ELSE
       IF inputs[i]=Missing THEN w=previous
       ELSE Cell(previous,RefMoment(Expected,FALSE),RefMoment(Expected,TRUE),2^Len(Expected)-1,w)
Completes == <>(i=3 /\ phase="ready")
=============================================================================
