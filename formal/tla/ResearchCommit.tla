--------------------------- MODULE ResearchCommit ---------------------------
EXTENDS Naturals

VARIABLES phase, checkpoint, backup, loggedBest, pending
vars == <<phase, checkpoint, backup, loggedBest, pending>>

None == "none"
Old == "old"
New == "new"

Init == /\ phase = "idle"
        /\ checkpoint = Old
        /\ backup = None
        /\ loggedBest = Old
        /\ pending = FALSE

InstallCandidate ==
    /\ phase = "idle"
    /\ loggedBest = Old
    /\ backup' = checkpoint
    /\ checkpoint' = New
    /\ pending' = TRUE
    /\ phase' = "installed"
    /\ UNCHANGED loggedBest

AppendExperimentLog ==
    /\ phase = "installed"
    /\ loggedBest' = New
    /\ phase' = "logged"
    /\ UNCHANGED <<checkpoint, backup, pending>>

Finalize ==
    /\ phase = "logged"
    /\ pending' = FALSE
    /\ backup' = None
    /\ phase' = "idle"
    /\ UNCHANGED <<checkpoint, loggedBest>>

Crash ==
    /\ phase \in {"installed", "logged"}
    /\ phase' = "restarting"
    /\ UNCHANGED <<checkpoint, backup, loggedBest, pending>>

RollbackUnlogged ==
    /\ phase = "restarting"
    /\ loggedBest = Old
    /\ checkpoint' = backup
    /\ backup' = None
    /\ pending' = FALSE
    /\ phase' = "idle"
    /\ UNCHANGED loggedBest

CommitLogged ==
    /\ phase = "restarting"
    /\ loggedBest = New
    /\ backup' = None
    /\ pending' = FALSE
    /\ phase' = "idle"
    /\ UNCHANGED <<checkpoint, loggedBest>>

Next == InstallCandidate \/ AppendExperimentLog \/ Finalize \/ Crash \/
        RollbackUnlogged \/ CommitLogged
Spec == Init /\ [][Next]_vars

TypeOK == /\ phase \in {"idle", "installed", "logged", "restarting"}
          /\ checkpoint \in {Old, New}
          /\ backup \in {None, Old}
          /\ loggedBest \in {Old, New}
          /\ pending \in BOOLEAN

RecoverySafety == /\ (phase = "idle" => checkpoint = loggedBest)
                  /\ (pending => backup = Old /\ checkpoint = New)
                  /\ (loggedBest = New => checkpoint = New)
=============================================================================
