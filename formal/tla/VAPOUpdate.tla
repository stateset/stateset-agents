-------------------------- MODULE VAPOUpdate -------------------------------
EXTENDS Integers, FiniteSets
(* Three logged response rows, one shared-backbone coordinate and one critic
   head coordinate. Signed inputs are gradient contributions before clipping.
   Unit covers every represented token, sequence and positive-token count. *)
Rows == 1..3
Groups == {1,2}
Unit == 360
Group(i) == IF i=1 THEN 1 ELSE 2
Members(g) == {i \in Rows : Group(i)=g}
Total(S,f) == (IF 1 \in S THEN f[1] ELSE 0)+(IF 2 \in S THEN f[2] ELSE 0)+(IF 3 \in S THEN f[3] ELSE 0)
Safe(x) == IF x>0 THEN x ELSE 1
VARIABLES len, positive, mode, warm, bad, limit, phase, seen, shared, head,
          actorWeight, headWeight, joint, criticSteps, scheduler
vars == <<len,positive,mode,warm,bad,limit,phase,seen,shared,head,actorWeight,headWeight,joint,criticSteps,scheduler>>
Tokens(S) == Total(S,len)
Pos(S) == Total(S \cap positive,len)
Den(S) == CASE mode="tokens" -> Tokens(S) [] mode="rows" -> Cardinality(S) [] OTHER -> 3*Cardinality(S)
RewardSlope(i,t,k) == k+i-t-1
ValueSlope(i,t,k) == (k+2)*(t-i)
HeadSlope(i,t,k) == (k-2)*(t+i)
PositiveSlope(i,t,k) == IF i \in positive THEN k+i+t ELSE 0
Row(i,k,F(_,_,_)) == (IF len[i]>=1 THEN F(i,1,k) ELSE 0)+(IF len[i]=2 THEN F(i,2,k) ELSE 0)
Mean(S,k,F(_,_,_),d) == (Unit*Total(S,[i \in Rows |-> Row(i,k,F)])) \div Safe(d)
Share(x,n,d) == (x*n) \div Safe(d)
Pol(S,k) ==
  LET n == Den(S)
      local == Total(S,[i \in Rows |-> (Unit*Row(i,k,RewardSlope)) \div
                    (IF mode="rows" THEN Safe(len[i]) ELSE 1)]) \div Safe(n)
  IN Share(local,n,Den(Rows))
Crit(S,k,F(_,_,_)) == Share(Mean(S,k,F,Tokens(S)),Tokens(S),Tokens(Rows))
LM(S,k) == Share(Mean(S,k,PositiveSlope,Pos(S)),Pos(S),Pos(Rows))

Init == /\ len \in [Rows -> 0..2] /\ positive \in SUBSET Rows /\ mode \in {"tokens","rows","fixed"}
        /\ warm \in BOOLEAN /\ bad \in 0..2 /\ limit \in 1..2
        /\ phase="start" /\ seen={} /\ shared=0 /\ head=0 /\ actorWeight=0
        /\ headWeight=0 /\ joint=0 /\ criticSteps=0 /\ scheduler=0
WarmCount == IF warm /\ Tokens(Rows)>0 THEN 1 ELSE 0
Start == /\ phase="start" /\ phase'=(IF Tokens(Rows)=0 THEN "done" ELSE "groups")
         /\ criticSteps'=WarmCount /\ headWeight'=WarmCount*11
         /\ UNCHANGED <<len,positive,mode,warm,bad,limit,seen,shared,head,actorWeight,joint,scheduler>>
Visit(g) == /\ phase="groups" /\ g \in Groups \ seen
            /\ IF bad=joint+1 /\ g=2 THEN
                   /\ phase'="aborted" /\ seen'={} /\ shared'=0 /\ head'=0
               ELSE LET S == Members(g) IN
                   /\ phase'="groups" /\ seen'=seen \cup {g}
                   /\ shared'=shared+Pol(S,joint)+2*Crit(S,joint,ValueSlope)+3*LM(S,joint)
                   /\ head'=head+2*Crit(S,joint,HeadSlope)
            /\ UNCHANGED <<len,positive,mode,warm,bad,limit,actorWeight,headWeight,joint,criticSteps,scheduler>>
Commit == /\ phase="groups" /\ seen=Groups /\ joint'=joint+1
          /\ actorWeight'=actorWeight-shared /\ headWeight'=headWeight-head
          /\ criticSteps'=criticSteps+1 /\ scheduler'=scheduler+1 /\ seen'={} /\ shared'=0 /\ head'=0
          /\ phase'=(IF joint'=limit THEN "done" ELSE "groups")
          /\ UNCHANGED <<len,positive,mode,warm,bad,limit>>
Next == Start \/ Commit \/ (\E g \in Groups : Visit(g))
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
(* Independently enumerate token positions in a flat update. *)
Coordinate(i,t,k,actor) ==
  IF t>len[i] THEN 0 ELSE
  IF ~actor THEN (2*Unit*HeadSlope(i,t,k)) \div Tokens(Rows)
  ELSE LET d == CASE mode="tokens" -> Tokens(Rows) [] mode="rows" -> 3*len[i] [] OTHER -> 9
       IN ((Unit*RewardSlope(i,t,k)) \div d) + ((2*Unit*ValueSlope(i,t,k)) \div Tokens(Rows))
            + ((3*Unit*PositiveSlope(i,t,k)) \div Safe(Pos(Rows)))
Flat(G,k,actor) == Total({i \in Rows : Group(i) \in G},
                      [i \in Rows |-> Coordinate(i,1,k,actor)+Coordinate(i,2,k,actor)])
Past(actor) == (IF joint>=1 THEN Flat(Groups,0,actor) ELSE 0)+(IF joint>=2 THEN Flat(Groups,1,actor) ELSE 0)
AppliedWarm == IF phase="start" THEN 0 ELSE WarmCount
Correct == /\ shared=Flat(seen,joint,TRUE) /\ head=Flat(seen,joint,FALSE)
           /\ actorWeight=-Past(TRUE) /\ headWeight=11*AppliedWarm-Past(FALSE)
           /\ criticSteps=joint+AppliedWarm /\ scheduler=joint
           /\ phase="aborted" => joint+1=bad /\ seen={}
           /\ phase="done" => joint=(IF Tokens(Rows)=0 THEN 0 ELSE limit)
Completes == <>(phase \in {"done","aborted"})
=============================================================================
