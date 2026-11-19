--------------------------- MODULE CriticMoments ---------------------------
EXTENDS Integers, FiniteSets
(* Three observations, two declared states, half-integer Bellman targets.
   Read rows in any order. Compare the streaming moments with independently
   centered observations and each accepted tolerance with the exact fixed point. *)
States == 1..2
Rows == 1..3
Other(s) == 3-s
Abs(x) == IF x < 0 THEN -x ELSE x
Max(x,y) == IF x > y THEN x ELSE y
Total(f) == f[1]+f[2]+f[3]
HalfEven(x) == LET q == Abs(x) \div 2
                   v == IF Abs(x)%2 = 1 /\ q%2 = 1 THEN q+1 ELSE q
               IN IF x < 0 THEN -v ELSE v

VARIABLES source, u, v, g, live, seen, count, residual, bias, active, squares
vars == <<source,u,v,g,live,seen,count,residual,bias,active,squares>>
Reward == <<-1,0,1>>
TwiceBackup(i) == 2*Reward[i]+(IF i \in live THEN g*v[Other(source[i])] ELSE 0)
Y(i) == HalfEven(TwiceBackup(i))
E(i) == Y(i)-u[source[i]]
Init == /\ source \in {<<1,1,2>>,<<1,2,2>>,<<1,1,1>>}
        /\ u \in [States -> {-1,1}] /\ v \in [States -> {-1,1}]
        /\ g \in 0..2 /\ live \in SUBSET Rows
        /\ seen = {} /\ count = <<0,0>> /\ residual = <<0,0>>
        /\ bias = <<0,0>> /\ active = <<0,0>> /\ squares = 0
Read(i) == LET s == source[i] IN
    /\ i \notin seen /\ seen' = seen \cup {i}
    /\ count' = [count EXCEPT ![s] = @+1]
    /\ residual' = [residual EXCEPT ![s] = @+E(i)]
    /\ bias' = [bias EXCEPT ![s] = @+2*Y(i)-TwiceBackup(i)]
    /\ active' = [active EXCEPT ![s] = @+(IF i \in live THEN 1 ELSE 0)]
    /\ squares' = squares+E(i)^2
    /\ UNCHANGED <<source,u,v,g,live>>
Next == \E i \in Rows : Read(i)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
From(s) == {i \in seen : source[i] = s}
Measurements ==
    /\ squares = Total([i \in Rows |-> IF i \in seen THEN E(i)^2 ELSE 0])
    /\ \A s \in States :
        /\ count[s] = Cardinality(From(s))
        /\ active[s] = Cardinality(From(s) \cap live)
        /\ residual[s] = Total([i \in Rows |-> IF i \in From(s) THEN E(i) ELSE 0])
        /\ bias[s] = Total([i \in Rows |-> IF i \in From(s) THEN 2*Y(i)-TwiceBackup(i) ELSE 0])
Covered == count[1] > 0 /\ count[2] > 0
Scale(s) == 6 \div count[s]
Gap6 == residual[1]^2*Scale(1)+residual[2]^2*Scale(2)
(* Each entry below is six times its deviation from its own state's mean. *)
Centered(i) == 6*E(i)-residual[source[i]]*Scale(source[i])
Variance == (seen = Rows /\ Covered) =>
    6*(6*squares-Gap6) = Total([i \in Rows |-> Centered(i)^2])
Rate12 == g*Max(active[1]*Scale(1),active[2]*Scale(2))
Bias12 == Max(Abs(bias[1])*Scale(1),Abs(bias[2])*Scale(2))
Lag == Max(Abs(u[1]-v[1]),Abs(u[2]-v[2]))
MinimumCount == IF count[1] <= count[2] THEN count[1] ELSE count[2]
Accept(e) == IF seen # Rows \/ ~Covered THEN FALSE ELSE
    LET m == (12-Rate12)*e-Bias12-Rate12*Lag
    IN Rate12 < 12 /\ m >= 0 /\ 24*Gap6 <= MinimumCount*m^2

(* Independent two-equation solution from the original rows and their rewards. *)
C(s) == 2*Cardinality({i \in Rows : source[i] = s})
A(s) == g*Cardinality({i \in live : source[i] = s})
R(s) == 2*Total([i \in Rows |-> IF source[i] = s THEN Reward[i] ELSE 0])
D == C(1)*C(2)-A(1)*A(2)
V(s) == R(s)*C(Other(s))+A(s)*R(Other(s))
Sound == \A e \in 0..10 : Accept(e) =>
    /\ D > 0 /\ \A s \in States : Abs(D*u[s]-V(s)) <= D*e
Completes == <>(seen = Rows)
=============================================================================
