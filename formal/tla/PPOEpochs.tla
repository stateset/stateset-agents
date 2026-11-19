------------------------------ MODULE PPOEpochs ------------------------------
EXTENDS Integers, Sequences, FiniteSets
(* Two response rows share three action positions. Their categorical heads
   use probabilities 1/4 and 3/4. The score table is in units of 1/3000.
   The independent bounds use 1.098 < log(3) < 1.099. *)
VARIABLES actions, shortFirst, cap, behavior, live, commits, stopped, finished, scheduler
vars == <<actions,shortFirst,cap,behavior,live,commits,stopped,finished,scheduler>>
Heads == {1,3} \X {1,3}
Old == <<1,3>>
Row(i) == IF shortFirst THEN (IF i=1 THEN 1 ELSE 2) ELSE (IF i=3 THEN 2 ELSE 1)
Weight(h,i) == IF actions[i] THEN h[Row(i)] ELSE 4-h[Row(i)]
Term(old,new) == IF old=new THEN 0 ELSE IF old=1 THEN 2704 ELSE 1296
Score(i) == Term(Weight(behavior,i),Weight(live,i))
Pass == Score(1)+Score(2)+Score(3) <= 3*cap
Count(h,from,to) == Cardinality({i \in 1..3 : Weight(Old,i)=from /\ Weight(h,i)=to})
Lower(h) == 2703*Count(h,1,3)+1294*Count(h,3,1)
Upper(h) == 2706*Count(h,1,3)+1297*Count(h,3,1)
Init == /\ actions \in [1..3 -> BOOLEAN] /\ shortFirst \in BOOLEAN
        /\ cap \in {0,450,1000,1350,1800,3000}
        /\ behavior = Old /\ live = Old /\ commits = <<>>
        /\ stopped = FALSE /\ finished = FALSE /\ scheduler = 0
Update == /\ ~finished /\ ~stopped /\ Len(commits)<3 /\ Pass
          /\ commits' = Append(commits,live) /\ live' \in Heads
          /\ UNCHANGED <<actions,shortFirst,cap,behavior,stopped,finished,scheduler>>
Stop == /\ ~finished /\ ~stopped /\ Len(commits)<3 /\ ~Pass /\ stopped' = TRUE
        /\ UNCHANGED <<actions,shortFirst,cap,behavior,live,commits,finished,scheduler>>
Finish == /\ ~finished /\ (stopped \/ Len(commits)=3) /\ finished' = TRUE
          /\ scheduler' = scheduler + 1
          /\ UNCHANGED <<actions,shortFirst,cap,behavior,live,commits,stopped>>
Next == Update \/ Stop \/ Finish
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Accepted == /\ behavior = Old /\ Len(commits)<=3
            /\ \A k \in 1..Len(commits) : Upper(commits[k]) <= 3*cap
FirstCrossing == stopped => Len(commits)<3 /\ Lower(live)>3*cap
Accounting == /\ scheduler = IF finished THEN 1 ELSE 0
              /\ finished => Len(commits)>0 /\ (stopped \/ Len(commits)=3)
Done == <>finished
=============================================================================
