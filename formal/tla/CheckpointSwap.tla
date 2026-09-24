--------------------------- MODULE CheckpointSwap ---------------------------
EXTENDS Naturals

VARIABLES phase, best, backup, staged
vars == <<phase, best, backup, staged>>

None == "none"
Old == "old"
New == "new"

Init == /\ phase = "idle"
        /\ best = Old
        /\ backup = None
        /\ staged = None

Prepare == /\ phase = "idle"
           /\ staged' = New
           /\ phase' = "prepared"
           /\ UNCHANGED <<best, backup>>

MoveOld == /\ phase = "prepared"
           /\ backup' = best
           /\ best' = None
           /\ phase' = "moved"
           /\ UNCHANGED staged

Install == /\ phase = "moved"
           /\ best' = staged
           /\ staged' = None
           /\ phase' = "installed"
           /\ UNCHANGED backup

Cleanup == /\ phase = "installed"
           /\ backup' = None
           /\ phase' = "idle"
           /\ UNCHANGED <<best, staged>>

InstallFailure == /\ phase = "moved"
                  /\ best' = backup
                  /\ backup' = None
                  /\ staged' = None
                  /\ phase' = "idle"

Crash == /\ phase \in {"prepared", "moved", "installed"}
         /\ phase' = "restarting"
         /\ UNCHANGED <<best, backup, staged>>

RecoverMissing == /\ phase = "restarting"
                  /\ best = None
                  /\ backup # None
                  /\ best' = backup
                  /\ backup' = None
                  /\ staged' = None
                  /\ phase' = "idle"

RecoverPresent == /\ phase = "restarting"
                  /\ best # None
                  /\ backup' = None
                  /\ staged' = None
                  /\ phase' = "idle"
                  /\ UNCHANGED best

Next == Prepare \/ MoveOld \/ Install \/ Cleanup \/ InstallFailure \/
        Crash \/ RecoverMissing \/ RecoverPresent
Spec == Init /\ [][Next]_vars

TypeOK == /\ phase \in {"idle", "prepared", "moved", "installed", "restarting"}
          /\ best \in {None, Old, New}
          /\ backup \in {None, Old, New}
          /\ staged \in {None, New}

Recoverable == best # None \/ backup # None
ReadyHasBest == phase = "idle" => best # None
=============================================================================
