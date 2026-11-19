-------------------------- MODULE RecordedReturns --------------------------
EXTENDS Integers
Rows == 1..3
Modes == {"live","cutoff","terminal","both"}
VARIABLES mode, r, g, recorded, phase, i, time, times, labels
vars == <<mode,r,g,recorded,phase,i,time,times,labels>>
Zero == [j \in Rows |-> 0]
Terminal(j) == mode[j] \in {"terminal","both"}
End(j) == j = 3 \/ mode[j] # "live"
External(j) == 5*j+1
Init == /\ mode \in [Rows -> Modes] /\ r \in [Rows -> {-1,1}]
        /\ g \in 0..2 /\ recorded \in BOOLEAN
        /\ phase = "check" /\ i = 1 /\ time = 0 /\ times = Zero /\ labels = Zero
Known == recorded \/ g = 0 \/ (\A j \in Rows : End(j) => Terminal(j))
Check == /\ phase = "check" /\ phase' = IF Known THEN "positions" ELSE "reject"
         /\ UNCHANGED <<mode,r,g,recorded,i,time,times,labels>>
Position ==
    /\ phase = "positions" /\ times' = [times EXCEPT ![i] = time]
    /\ time' = IF End(i) THEN 0 ELSE time+1
    /\ phase' = IF i = 3 THEN "labels" ELSE phase
    /\ i' = IF i = 3 THEN i ELSE i+1
    /\ UNCHANGED <<mode,r,g,recorded,labels>>
Label ==
    LET value == IF recorded THEN External(i) ELSE
            4*r[i] + (IF End(i) THEN 0 ELSE (g*labels[i+1]) \div 2)
    IN /\ phase = "labels" /\ labels' = [labels EXCEPT ![i] = value]
       /\ phase' = (IF i = 1 THEN "done" ELSE phase) /\ i' = i-1
       /\ UNCHANGED <<mode,r,g,recorded,time,times>>
Next == Check \/ Position \/ Label
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent forward expansion. No reward can cross a recording boundary.
   A final live row is an implicit cutoff; a simultaneous timeout and actual
   termination has a known zero continuation. All labels use units of 1/4. *)
Expected(j) ==
    IF recorded THEN External(j) ELSE
    4*r[j] + (IF j < 3 /\ mode[j] = "live" THEN 2*g*r[j+1] ELSE 0) +
    (IF j = 1 /\ mode[1] = "live" /\ mode[2] = "live" THEN g*g*r[3] ELSE 0)
Offset(j) == IF j = 1 THEN 0 ELSE IF mode[j-1] # "live" THEN 0 ELSE
                 IF j = 3 /\ mode[1] = "live" THEN 2 ELSE 1
Acceptable == recorded \/ g = 0 \/
    ((\A j \in Rows : mode[j] # "cutoff") /\ mode[3] \in {"terminal","both"})
Published == /\ phase = "reject" => (~Acceptable /\ times = Zero /\ labels = Zero)
             /\ phase = "done" => (Acceptable /\
                    (\A j \in Rows : times[j] = Offset(j) /\ labels[j] = Expected(j)))
Completes == <>(phase \in {"done","reject"})
=============================================================================
