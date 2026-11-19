-------------------------- MODULE BellmanRounding ---------------------------
EXTENDS Integers
(* Signed binary format, two significand bits. All calculations below use
   integer units of 1/64. 256 is the virtual endpoint for overflow. *)
Scale == 64
Positive == {0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256}
Grid == Positive \cup {-i : i \in Positive}
Finite == Grid \ {-256,256}
Even == {0,2,4,8,16,32,64,128,256}
Abs(x) == IF x < 0 THEN -x ELSE x
(* Function application evaluates its arguments once, including nested casts. *)
Step[a \in Nat, s \in Nat, k \in Nat] ==
    CASE a < k*s -> s [] a < 2*k*s -> 2*s [] a < 4*k*s -> 4*s
      [] a < 8*k*s -> 8*s [] a < 16*k*s -> 16*s [] a < 32*k*s -> 32*s
      [] a < 64*k*s -> 64*s [] a < 128*k*s -> 128*s [] OTHER -> 256*s
Nearest[x \in Int, s \in Nat] ==
    LET q == Abs(x) \div s
        r == Abs(x) % s
        up == 2*r > s \/ (2*r = s /\ q % 2 = 1)
        n == (IF up THEN q+1 ELSE q)*s
    IN IF x < 0 THEN -n ELSE n
Wide[x \in Int] == Nearest[x,Step[Abs(x),1,64]]
Clip[x \in Int] == IF x > 256*Scale THEN 256*Scale ELSE IF x < -256*Scale THEN -256*Scale ELSE x
Narrow[x \in Int] == Clip[Nearest[x,Step[Abs(x),Scale,4]]]

VARIABLES r,v,g,intermediate,done,result
vars == <<r,v,g,intermediate,done,result>>
Cast(x) == Narrow[IF intermediate THEN Nearest[x,8] ELSE x]
Exact == Scale*r + 2*g*v
Init == /\ r \in Finite /\ v \in Finite /\ g \in 0..32
        /\ intermediate \in BOOLEAN /\ done = FALSE /\ result = 0
Publish ==
    /\ ~done /\ done' = TRUE /\ UNCHANGED <<r,v,g,intermediate>>
    /\ \E d \in {Wide[Exact]} :
       \E c \in {Cast(d)} :
       \E b \in {Cast(Wide[d+Wide[d-c]])} :
       \E m \in {Wide[c+b] \div 2} :
       \E diff \in {Wide[Scale*r-m]} :
       \E tail \in {IF Abs(Scale*r) >= Abs(m) THEN Wide[Wide[Scale*r-diff]-m]
                   ELSE Wide[Scale*r-Wide[m+diff]]} :
       \E residual \in {Wide[Wide[diff+2*g*v]+tail]} :
          LET change == b # c /\ (IF b > c THEN residual > 0 ELSE residual < 0)
          IN result' = Cast(IF change THEN Wide[d-Wide[(d-b) \div 2]] ELSE d)
Spec == Init /\ [][Publish]_vars /\ WF_vars(Publish)

(* Compare to every endpoint's distance from the original exact numerator.
   Equal-distance endpoints with even significands win. *)
NearestEven == done =>
    /\ result \in {i*Scale : i \in Grid}
    /\ \A i \in Grid : LET dist == Abs(i*Scale-Exact)
                            err == Abs(result-Exact)
                        IN err < dist \/ (err = dist /\
                           (i*Scale = result \/ Abs(result) \div Scale \in Even))
(* A lost difference tail changes the first result. The optional
   intermediate cast alone changes the second uncorrected result. *)
Witnesses ==
    /\ (done /\ r = 1 /\ v = 128 /\ g = 20) =>
          (result = 96*Scale /\ Cast(Wide[Exact]) = 64*Scale /\ Wide[-79*Scale] = -80*Scale)
    /\ (done /\ intermediate /\ r = 1 /\ v = 1 /\ g = 15) =>
          (result = Scale /\ Narrow[Wide[Exact]] = Scale /\ Cast(Wide[Exact]) = 2*Scale)
Completes == <>done
=============================================================================
