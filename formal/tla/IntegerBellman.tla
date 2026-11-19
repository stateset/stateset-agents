--------------------------- MODULE IntegerBellman ---------------------------
EXTENDS Integers, Sequences
(* Raw signed and unsigned labels can have more bits than the critic.
   Each input is a numerator and a power-of-two denominator. Use units
   of 1/64 for the independent reference, including the virtual endpoint. *)
Sources == {<<n,1>> : n \in {-32,-17,-16,-3,-1,0,1,3,15,16,17,31,32,33,63}} \cup
           {<<n,d>> : n \in {-3,-1,1,3}, d \in {2,8}}
Discounts == {<<0,1>>,<<1,8>>,<<1,2>>,<<3,4>>,<<1,1>>}
Abs(x) == IF x < 0 THEN -x ELSE x
Max(a,b) == IF a > b THEN a ELSE b
Clamp(x) == IF x > 2048 THEN 2048 ELSE IF x < -2048 THEN -2048 ELSE x
(* Step is the smallest power of two meeting both the significand and
   minimum-spacing constraints. The model uses three significant bits. *)
Step[n \in Nat, d \in Nat, s \in Nat] ==
    IF n < 8*s /\ d <= 4*s THEN s ELSE Step[n,d,2*s]
Rounded[n \in Int, s \in Nat, d \in Nat] ==
    LET q == Abs(n) \div s
        rem == Abs(n) % s
        bump == 2*rem > s \/ (2*rem = s /\ q % 2 = 1)
        mag == (IF bump THEN q+1 ELSE q)*s*(64 \div d)
    IN Clamp(IF n < 0 THEN -mag ELSE mag)
VARIABLES r,v,g,terminal,phase,n,d,out
vars == <<r,v,g,terminal,phase,n,d,out>>
Init == /\ r \in Sources /\ v \in Sources /\ (r[2] = 1 \/ v[2] = 1)
        /\ g \in Discounts /\ terminal \in BOOLEAN /\ phase = 0
        /\ n = 0 /\ d = 1 /\ out = 0
Align == /\ phase = 0 /\ phase' = 1 /\ d' = Max(r[2],g[2]*v[2])
         /\ n' = r[1]*(d' \div r[2]) +
                (IF terminal THEN 0 ELSE g[1]*v[1]*(d' \div (g[2]*v[2])))
         /\ UNCHANGED <<r,v,g,terminal,out>>
Publish == /\ phase = 1 /\ phase' = 2 /\ out' = Rounded[n,Step[Abs(n),d,1],d]
           /\ UNCHANGED <<r,v,g,terminal,n,d>>
Next == Align \/ Publish
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Ref == r[1]*(64 \div r[2]) + (IF terminal THEN 0 ELSE g[1]*v[1]*(64 \div (g[2]*v[2])))
Aligned == phase > 0 => 64*n = d*Ref
(* Rather than round the reference, enumerate every representable endpoint.
   Successive positive encodings alternate even and odd, starting at zero. *)
Levels == <<0,16,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048>>
Optimal == phase = 2 =>
    /\ (out = 0 \/ (out > 0) = (Ref > 0))
    /\ \E i \in DOMAIN Levels : /\ Abs(out) = Levels[i]
       /\ \A j \in DOMAIN Levels :
           LET dist == Abs(Levels[j]-Abs(Ref))
               err == Abs(Levels[i]-Abs(Ref))
           IN err < dist \/ (err = dist /\ (i = j \/ i % 2 = 1))
Completes == <>(phase = 2)
=============================================================================
