-------------------------- MODULE ActorCriticSegments --------------------------
EXTENDS Integers, FiniteSets

Segments == 1..2
Slots == 1..2
Scale == 16
VARIABLES length, terminal, decay, nextId, collecting, staged, ready, used,
          aborted, version, owner, targets
vars == <<length, terminal, decay, nextId, collecting, staged, ready, used,
          aborted, version, owner, targets>>

(* The discount is 1/2. Decay is in half units. Resets deliberately have
   different values from the cutoff successor of the preceding call. *)
Reward(k, t) == IF k = 1 THEN 2 - 2*t ELSE 3*t - 4
Value(k, t, policy) == (4*k + t)*(1 + policy)
Bootstrap(k, policy) == IF terminal[k] THEN 0 ELSE Value(k, length[k]+1, policy)

RECURSIVE Residual(_, _, _)
Residual(k, t, policy) ==
    LET v == Value(k, t, policy)
        successor == IF t = length[k] THEN Bootstrap(k, policy) ELSE Value(k, t+1, policy)
        future == IF t = length[k] THEN 0 ELSE (decay * Residual(k, t+1, policy)) \div 4
    IN Scale * (Reward(k, t) - v) + 8 * successor + future

(* Independently expand the one-step and two-step returns in each segment. *)
Expected(k, t, policy) ==
    IF t = length[k] THEN Scale * Reward(k, t) + 8 * Bootstrap(k, policy)
    ELSE Scale * Reward(k, t) + 4 * (2-decay) * Value(k, t+1, policy)
         + 4 * decay * Reward(k, t+1) + 2 * decay * Bootstrap(k, policy)

Init ==
    /\ length \in [Segments -> Slots]
    /\ terminal \in [Segments -> BOOLEAN]
    /\ decay \in 0..2
    /\ nextId = 1 /\ collecting = FALSE /\ staged = 0
    /\ ready = {} /\ used = {} /\ aborted = {}
    /\ version = 0 /\ owner = [k \in Segments |-> -1]
    /\ targets = [k \in Segments |-> [t \in Slots |-> 0]]

Begin ==
    /\ ~collecting /\ nextId \in Segments
    /\ collecting' = TRUE
    /\ owner' = [owner EXCEPT ![nextId] = version]
    /\ UNCHANGED <<length, terminal, decay, nextId, staged, ready, used, aborted, version, targets>>

Append ==
    /\ collecting /\ staged < length[nextId] /\ staged' = staged + 1
    /\ UNCHANGED <<length, terminal, decay, nextId, collecting, ready, used, aborted, version, owner, targets>>

Close(success) ==
    /\ collecting /\ (success => staged = length[nextId])
    /\ ready' = IF success THEN ready \cup {nextId} ELSE ready
    /\ aborted' = IF success THEN aborted ELSE aborted \cup {nextId}
    /\ collecting' = FALSE /\ staged' = 0 /\ nextId' = nextId + 1
    /\ UNCHANGED <<length, terminal, decay, used, version, owner, targets>>

Learn ==
    /\ ~collecting /\ ready # {}
    /\ targets' = [k \in Segments |-> [t \in Slots |->
          IF k \in ready /\ t <= length[k]
          THEN Scale * Value(k, t, version) + Residual(k, t, version)
          ELSE targets[k][t]]]
    /\ version' = version + 1 /\ used' = used \cup ready /\ ready' = {}
    /\ UNCHANGED <<length, terminal, decay, nextId, collecting, staged, aborted, owner>>

Next == Begin \/ Append \/ Close(TRUE) \/ Close(FALSE) \/ Learn
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Frozen ==
    /\ (collecting => owner[nextId] = version)
    /\ \A k \in ready : owner[k] = version
Isolation ==
    /\ ready \cup used \cup aborted = 1..(nextId-1)
    /\ ready \cap used = {} /\ used \cap aborted = {} /\ ready \cap aborted = {}
    /\ (used = {}) = (version = 0) /\ version <= Cardinality(used)
Returns == \A k \in Segments, t \in Slots :
    targets[k][t] = (IF k \in used /\ t <= length[k] THEN Expected(k, t, owner[k]) ELSE 0)
Finished == <>(nextId = 3 /\ ~collecting /\ ready = {})
=============================================================================
