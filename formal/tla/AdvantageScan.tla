---------------------------- MODULE AdvantageScan ----------------------------
EXTENDS Integers, Sequences

CONSTANT Horizon
VARIABLES rewards, values, done, gamma, lambda, cursor, advantages, rewardPart, valuePart
vars == <<rewards, values, done, gamma, lambda, cursor, advantages, rewardPart, valuePart>>

(* gamma and lambda are in half units. The common denominator makes every
    division below exact for this horizon, including the bootstrap term. *)
Scale == 2^(2*Horizon + 1)
Steps == 1..Horizon
Stopped(t) == done[t]
Delta(t) == Scale * (rewards[t] - values[t]) +
            (IF Stopped(t) THEN 0 ELSE gamma * values[t+1] * (Scale \div 2))

(* Forward definition. Each future TD residual is weighted by its distance
    to the queried step and excluded if an intervening transition terminates. *)
RECURSIVE Forward(_, _)
Forward(start, t) ==
    IF t > Horizon THEN 0
    ELSE (IF t = start THEN Delta(t)
          ELSE IF \E j \in start..(t-1) : Stopped(j) THEN 0
          ELSE (Delta(t) * (gamma * lambda)^(t-start)) \div 4^(t-start))
         + Forward(start, t+1)

Init ==
    /\ rewards \in [Steps -> {-1, 1}]
    /\ values \in [1..(Horizon+1) -> {-1, 1}]
    /\ done \in [Steps -> BOOLEAN]
    /\ gamma \in 0..2 /\ lambda \in 0..2
    /\ cursor = Horizon
    /\ advantages = [i \in 1..(Horizon+1) |-> 0]

    /\ rewardPart = 0 /\ valuePart = Scale*values[Horizon+1]

Scan ==
    /\ cursor > 0
    /\ rewardPart' = Scale*rewards[cursor] +
        (IF Stopped(cursor) THEN 0 ELSE (gamma*lambda*rewardPart) \div 4)
    /\ valuePart' = IF Stopped(cursor) THEN 0 ELSE
        (gamma*((2-lambda)*Scale*values[cursor+1] + lambda*valuePart)) \div 4
    /\ advantages' = [advantages EXCEPT ![cursor] = rewardPart' + valuePart' - Scale*values[cursor]]
    /\ cursor' = cursor - 1
    /\ UNCHANGED <<rewards, values, done, gamma, lambda>>

Next == Scan \/ (cursor = 0 /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Scan)
SuffixCorrect == \A t \in (cursor+1)..Horizon : advantages[t] = Forward(t,t)
TerminalReturn == \A t \in (cursor+1)..Horizon :
                  Stopped(t) => advantages[t] + Scale * values[t] = Scale * rewards[t]
Finished == <>(cursor = 0)
=============================================================================
