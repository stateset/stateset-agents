----------------------------- MODULE TargetPolicy -----------------------------
EXTENDS Integers, Sequences
(* Coordinates of one policy and two critic biases. Each unit is one eighth.
   A completed update publishes all three with tau=1/2 on steps 1 and 3. *)
VARIABLES history, live, target, saved, restored, read
vars == <<history, live, target, saved, restored, read>>
Initial == <<0,0,0>>
Moves == {<<8,-8,0>>, <<-8,0,8>>, <<0,8,-8>>}
Mix(old,new) == [i \in 1..3 |-> (old[i]+new[i]) \div 2]
Published(h) ==
    CASE Len(h) = 0 -> Initial
      [] Len(h) <= 2 -> [i \in 1..3 |-> h[1][i] \div 2]
      [] OTHER -> [i \in 1..3 |-> (h[1][i]+2*h[3][i]) \div 4]
Minimum(x,y) == IF x <= y THEN x ELSE y
Maximum(x,y) == IF x >= y THEN x ELSE y
Score(a,q,offset) == Minimum(q[2]+a+offset, q[3]-a-offset)
Future(a,q) == Maximum(Score(a,q,-2), Score(a,q,2))

Init == /\ history = <<>> /\ live = Initial /\ target = Initial
        /\ saved = <<>> /\ restored = FALSE /\ read = <<>>
Learn(new) ==
    /\ Len(history) < 4 /\ new \in Moves
    /\ history' = Append(history,new) /\ live' = new
    /\ target' = IF Len(history) % 2 = 0 THEN Mix(target,new) ELSE target
    /\ read' = <<>> /\ UNCHANGED <<saved,restored>>
Observe(done,g) ==
    /\ read = <<>>
    /\ LET active == ~done /\ g # 0
           action == target[1]
       IN read' = <<done, g, IF active THEN 1 ELSE 0,
                     IF active THEN 3 + g*Future(action,target) ELSE 3>>
    /\ UNCHANGED <<history,live,target,saved,restored>>
Save == /\ saved = <<>> /\ ~restored /\ saved' = <<history,live,target>>
        /\ UNCHANGED <<history,live,target,restored,read>>
Restore == /\ saved # <<>> /\ ~restored /\ restored' = TRUE
           /\ history' = saved[1] /\ live' = saved[2] /\ target' = saved[3]
           /\ saved' = <<>> /\ read' = <<>>
Next == (\E new \in Moves : Learn(new)) \/ Save \/ Restore
        \/ (\E done \in BOOLEAN, g \in 0..1 : Observe(done,g))
        \/ UNCHANGED vars
Snapshots == /\ target = Published(history)
             /\ live = IF Len(history) = 0 THEN Initial ELSE history[Len(history)]
Backup == read # <<>> =>
    LET ref == Published(history) IN
    IF read[1] \/ read[2] = 0 THEN read[3] = 0 /\ read[4] = 3
    ELSE read[3] = 1 /\ read[4] = 3 + read[2]*Future(ref[1],ref)
=============================================================================
