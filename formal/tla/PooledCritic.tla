---------------------------- MODULE PooledCritic ----------------------------
EXTENDS Integers
(* Two states, two equally weighted observations per state. State 1 occurs
   once or twice in four visits. The second reward can differ by one, which
   makes opposite rounding errors cancel at a half-integer target. *)
States == {1,2}
Abs(x) == IF x < 0 THEN -x ELSE x
Max(a,b) == IF a > b THEN a ELSE b
Other(s) == 3-s
EvenQuarter(x) ==
    LET q == Abs(x) \div 4
        rem == Abs(x) % 4
        z == IF rem > 2 \/ (rem = 2 /\ q % 2 = 1) THEN q+1 ELSE q
    IN IF x < 0 THEN -z ELSE z

VARIABLES u,v,r,bonus,live,g,visits,self,seen,sq,noise,bias
vars == <<u,v,r,bonus,live,g,visits,self,seen,sq,noise,bias>>
W(s) == IF s = 1 THEN visits ELSE 4-visits
NextState(s) == IF self THEN s ELSE Other(s)
A(s) == IF s \in live THEN g ELSE 0
Rate == IF live = {} THEN 0 ELSE g
FourTarget(s,i) == 4*(r[s]+i*bonus) + A(s)*v[NextState(s)]
Y(s,i) == EvenQuarter(FourTarget(s,i))
Squared(s) == W(s)*((u[s]-Y(s,0))^2 + (u[s]-Y(s,1))^2)
Noise(s) == W(s)*(Y(s,0)-Y(s,1))^2
Drift(s) == Abs(4*(Y(s,0)+Y(s,1))-FourTarget(s,0)-FourTarget(s,1))

Init == /\ u \in [States -> {-1,1}] /\ v \in [States -> {-1,1}]
        /\ r \in [States -> {-1,1}] /\ bonus \in {0,1} /\ live \in SUBSET States
        /\ g \in {0,2,3} /\ visits \in {1,2} /\ self \in BOOLEAN
        /\ seen = {} /\ sq = 0 /\ noise = 0 /\ bias = 0
Collect(s) == /\ s \notin seen /\ seen' = seen \cup {s}
              /\ sq' = sq+Squared(s) /\ noise' = noise+Noise(s) /\ bias' = Max(bias,Drift(s))
              /\ UNCHANGED <<u,v,r,bonus,live,g,visits,self>>
Next == \E s \in States : Collect(s)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Measured ==
    /\ sq = (IF 1 \in seen THEN Squared(1) ELSE 0) + (IF 2 \in seen THEN Squared(2) ELSE 0)
    /\ noise = (IF 1 \in seen THEN Noise(1) ELSE 0) + (IF 2 \in seen THEN Noise(2) ELSE 0)
    /\ bias = Max(IF 1 \in seen THEN Drift(1) ELSE 0, IF 2 \in seen THEN Drift(2) ELSE 0)

(* Independent solution of the exact two-state Bellman equations. For self
   transitions use a common denominator for both single-state values. *)
R2(s) == 2*r[s]+bonus
D == IF self THEN (4-A(1))*(4-A(2)) ELSE 16-A(1)*A(2)
Fixed(s) == IF self THEN 2*R2(s)*(4-A(Other(s))) ELSE 8*R2(s)+2*A(s)*R2(Other(s))
FixedEquations == \A s \in States : 4*Fixed(s) = 2*R2(s)*D + A(s)*Fixed(NextState(s))
Err == Max(Abs(D*u[1]-Fixed(1)),Abs(D*u[2]-Fixed(2)))
Lag == Max(Abs(u[1]-v[1]),Abs(u[2]-v[2]))
(* sq/8 is the pooled MSE, noise/16 its within-state variance, and
   visits/4 the smallest state mass.
   X is 8*D times value error times (1-b), less signed bias and lag. *)
X == 2*(4-Rate)*Err-D*(bias+2*Rate*Lag)
Certificate == seen = States => (X <= 0 \/ 4*visits*X^2 <= (8*D)^2*(2*sq-noise))
Decomposition == seen = States =>
    2*sq-noise = W(1)*(Y(1,0)+Y(1,1)-2*u[1])^2 + W(2)*(Y(2,0)+Y(2,1)-2*u[2])^2
Completes == <>(seen = States)
=============================================================================
