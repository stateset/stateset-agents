---------------------------- MODULE AsyncRuntime ----------------------------
EXTENDS Naturals, Integers

CONSTANT MaxUpdates
ASSUME MaxUpdates \in Nat

VARIABLES phase, visible, published, completed, workersStarted, failed
vars == <<phase, visible, published, completed, workersStarted, failed>>

Init == /\ phase = "initial"
        /\ visible = 0
        /\ published = -1
        /\ completed = 0
        /\ workersStarted = FALSE
        /\ failed = FALSE

PublishInitial == /\ phase = "initial"
                  /\ published' = 0
                  /\ phase' = "ready"
                  /\ UNCHANGED <<visible, completed, workersStarted, failed>>

StartWorkers == /\ phase = "ready"
                /\ published = visible
                /\ workersStarted' = TRUE
                /\ phase' = "running"
                /\ UNCHANGED <<visible, published, completed, failed>>

Learn == /\ phase = "running"
         /\ completed < MaxUpdates
         /\ phase' = "publishing"
         /\ UNCHANGED <<visible, published, completed, workersStarted, failed>>

PublishUpdate == /\ phase = "publishing"
                 /\ published' = visible + 1
                 /\ phase' = "advancing"
                 /\ UNCHANGED <<visible, completed, workersStarted, failed>>

Advance == /\ phase = "advancing"
           /\ published = visible + 1
           /\ visible' = visible + 1
           /\ completed' = completed + 1
           /\ phase' = IF completed' = MaxUpdates THEN "done" ELSE "running"
           /\ UNCHANGED <<published, workersStarted, failed>>

Fail == /\ phase \in {"initial", "ready", "running", "publishing", "advancing"}
        /\ failed' = TRUE
        /\ phase' = "stopped"
        /\ UNCHANGED <<visible, published, completed, workersStarted>>

Next == PublishInitial \/ StartWorkers \/ Learn \/ PublishUpdate \/ Advance \/ Fail
Spec == Init /\ [][Next]_vars

TypeOK == /\ phase \in {"initial", "ready", "running", "publishing", "advancing", "done", "stopped"}
          /\ visible \in Nat
          /\ published \in Int
          /\ completed \in Nat
          /\ workersStarted \in BOOLEAN
          /\ failed \in BOOLEAN

VersionSafety == /\ visible <= published + 1
                 /\ (workersStarted => published >= visible)
                 /\ completed = visible
                 /\ completed <= MaxUpdates
                 /\ (phase = "done" => completed = MaxUpdates)
=============================================================================
