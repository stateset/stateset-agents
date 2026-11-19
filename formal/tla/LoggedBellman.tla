--------------------------- MODULE LoggedBellman ---------------------------
EXTENDS Integers, FiniteSets
States == 1..2
Other(s) == 3-s
Modes == {"live", "cutoff", "terminal", "both"}
Abs(x) == IF x < 0 THEN -x ELSE x
Max(a,b) == IF a > b THEN a ELSE b

(* Each state has two equally likely observations with mean reward r[s].
   One always terminates. The other carries the mode below and, if live,
   leads to the opposite state. g/2 is the configured discount. *)
VARIABLES mode, r, g, initial, step, table, pending, scratch, scored
vars == <<mode,r,g,initial,step,table,pending,scratch,scored>>
Zero == [s \in States |-> 0]
Init ==
    /\ mode \in [States -> Modes] /\ r \in [States -> {-1, 1}]
    /\ g \in 0..2 /\ initial \in [States -> {-1, 1}]
    /\ step = 0 /\ table = initial /\ pending = States
    /\ scratch = Zero /\ scored = {}
Terminal(s) == mode[s] \in {"terminal","both"}
Evaluate(s) ==
    LET read == ~Terminal(s) /\ g # 0
        v == 4*r[s]*(4^step) + (IF read THEN g*table[Other(s)] ELSE 0)
    IN /\ step < 3 /\ s \in pending /\ pending' = pending \ {s}
       /\ scratch' = [scratch EXCEPT ![s] = v]
       /\ scored' = IF read THEN scored \cup {s} ELSE scored
       /\ UNCHANGED <<mode,r,g,initial,step,table>>
Commit ==
    /\ step < 3 /\ pending = {}
    /\ table' = scratch /\ step' = step+1 /\ pending' = States
    /\ scratch' = Zero /\ scored' = {}
    /\ UNCHANGED <<mode,r,g,initial>>
Next == (\E s \in States : Evaluate(s)) \/ Commit
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent closed solution to V1 = r1 + a1 V2 and V2 = r2 + a2 V1.
   Coefficients A are in quarters. Only a cutoff with no termination lives. *)
A(s) == IF mode[s] \in {"live","cutoff"} THEN g ELSE 0
Den == 16-A(1)*A(2)
Fixed(s) == 16*r[s]+4*A(s)*r[Other(s)]
Rate == Max(A(1),A(2))
Error(s) == Abs(Den*table[s] - (4^step)*Fixed(s))
InitialError(s) == Abs(Den*initial[s]-Fixed(s))
Residual(s) == Abs(4*table[s]-4*r[s]*(4^step)-A(s)*table[Other(s)])
Certificate ==
    Max(Error(1),Error(2))*(4-Rate) <= Den*Max(Residual(1),Residual(2))
Shrink == Max(Error(1),Error(2)) <=
    (IF step = 0 THEN 1 ELSE Rate^step)*Max(InitialError(1),InitialError(2))

(* Unroll the path directly in the original reward and bootstrap table. *)
Path(s) ==
    LET o == Other(s)
        p == A(s)*A(o)
    IN CASE step = 0 -> initial[s]
         [] step = 1 -> 4*r[s]+A(s)*initial[o]
         [] step = 2 -> 16*r[s]+4*A(s)*r[o]+p*initial[s]
         [] OTHER -> 64*r[s]+16*A(s)*r[o]+4*p*r[s]+p*A(s)*initial[o]
History == \A s \in States : table[s] = Path(s)
ReadSet == scored = {s \in States \ pending : A(s) # 0}
Completes == <>(step = 3)
=============================================================================
