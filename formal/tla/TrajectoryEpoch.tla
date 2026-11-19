-------------------------- MODULE TrajectoryEpoch ---------------------------
EXTENDS Integers, FiniteSets
(* Decision Transformer fits one action at every retained step and one state
   only when the same trajectory has a retained successor. The learner may
   change after every update. Record its version at each row's visit. *)
VARIABLES length, context, capacity, pending, visit, version, actNum, stateNum
vars == <<length,context,capacity,pending,visit,version,actNum,stateNum>>
Rows == 1..3
Slots == 1..3
Min(a,b) == IF a < b THEN a ELSE b
Len(i) == Min(length[i],context)
Action(i,j,v) == IF (i+j+v) % 2 = 0 THEN 1 ELSE 0
State(i,j,v) == IF (2*i+j+v) % 3 = 0 THEN 1 ELSE 0
Sum3(f) == f[1]+f[2]+f[3]
InBatch(b,i,x) == IF i \in b THEN x ELSE 0
Count(b,skip) == Sum3([i \in Rows |-> InBatch(b,i,Len(i)-skip)])
Loss(i,j,skip) == IF skip = 0 THEN Action(i,j,version) ELSE State(i,j,version)
BatchSum(b,skip) == Sum3([i \in Rows |-> InBatch(b,i,
    Sum3([j \in Slots |-> IF j <= Len(i)-skip THEN Loss(i,j,skip) ELSE 0]))])
(* 2520 is divisible by every nonzero target count in this model. *)
BatchMean(b,skip) == IF Count(b,skip) = 0 THEN 0
                     ELSE (2520*BatchSum(b,skip)) \div Count(b,skip)
Init == /\ length \in [Rows -> 1..3] /\ context \in 1..3 /\ capacity \in 1..4
        /\ pending = Rows /\ visit = [i \in Rows |-> 0]
        /\ version = 1 /\ actNum = 0 /\ stateNum = 0
Take(b) == /\ b \subseteq pending /\ b # {}
           /\ Cardinality(b) = Min(capacity,Cardinality(pending))
           /\ actNum' = actNum + Count(b,0)*BatchMean(b,0)
           /\ stateNum' = stateNum + (Count(b,0)-Cardinality(b))*BatchMean(b,1)
           /\ visit' = [i \in Rows |-> IF i \in b THEN version ELSE visit[i]]
           /\ pending' = pending \ b /\ version' = version+1
           /\ UNCHANGED <<length,context,capacity>>
Next == \E b \in SUBSET pending : Take(b)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
(* The reference counts each observed error in position space. *)
Targets == {p \in Rows \X Slots : visit[p[1]] > 0 /\ p[2] <= length[p[1]] /\ p[2] <= context}
Successors == {p \in Targets : <<p[1], p[2]+1>> \in Targets}
ActionMean == actNum = 2520*Cardinality({p \in Targets : Action(p[1],p[2],visit[p[1]]) = 1})
StateMean == stateNum = 2520*Cardinality({p \in Successors : State(p[1],p[2],visit[p[1]]) = 1})
Denominators == /\ Count(Rows \ pending,0) = Cardinality(Targets)
                /\ Count(Rows \ pending,0)-Cardinality(Rows \ pending) = Cardinality(Successors)
Progress == <>(pending = {})
=============================================================================
