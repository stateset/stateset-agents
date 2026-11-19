---------------------------- MODULE OfflineBackup ----------------------------
EXTENDS Integers, FiniteSets
(* Exhaust the action proposals in arbitrary order, retain the maximum
   BCQ mixture or BEAR lower twin-Q value, then form the terminal-aware
   target. Scores use halves and the discount uses half units. *)
CONSTANT Actions
VARIABLES q1, q2, weight, discount, reward, terminal, pending, best, committed, target
Min(x,y) == IF x < y THEN x ELSE y
Max(x,y) == IF x > y THEN x ELSE y
Score(a) == weight*Min(q1[a],q2[a]) + (2-weight)*Max(q1[a],q2[a])
Init ==
    /\ q1 \in [Actions -> {-1,0,1}] /\ q2 \in [Actions -> {-1,0,1}]
    /\ weight \in 0..2 /\ discount \in 0..2 /\ reward \in {-1,1}
    /\ terminal \in BOOLEAN /\ pending = Actions
    /\ best = 0 /\ committed = FALSE /\ target = 0
Visit(a) ==
    /\ a \in pending
    /\ best' = IF pending = Actions THEN Score(a) ELSE Max(best,Score(a))
    /\ pending' = pending \ {a}
    /\ UNCHANGED <<q1,q2,weight,discount,reward,terminal,committed,target>>
Commit ==
    /\ pending = {} /\ ~committed /\ committed' = TRUE
    /\ target' = 4*reward + (IF terminal THEN 0 ELSE discount*best)
    /\ UNCHANGED <<q1,q2,weight,discount,reward,terminal,pending,best>>
Next == (\E a \in Actions : Visit(a)) \/ Commit
    \/ (committed /\ UNCHANGED <<q1,q2,weight,discount,reward,terminal,pending,best,committed,target>>)

(* Batch specification expressed without the running maximum. A selected
   value must be both attainable and an upper bound, even for negative Q. *)
Argmax(values,S,x) == (\E a \in S : x = values[a]) /\ (\A a \in S : x >= values[a])
Candidates == [a \in Actions |-> Score(a)]
VisitedOK == pending /= Actions => Argmax(Candidates, Actions \ pending, best)
TargetOK == committed =>
    /\ pending = {}
    /\ IF terminal THEN target = 4*reward ELSE
       Argmax([a \in Actions |-> 4*reward+discount*Score(a)], Actions, target)
=============================================================================
