--------------------------- MODULE LoggedExtraction --------------------------
EXTENDS Integers, FiniteSets, Sequences

(* Three actions share a prompt. Context 0 is the prompt, context 2 follows B.
   A gradient coordinate moves token A's logit. Odds are fixed within an
   accumulation window and each prefix can have a different distribution. *)
Rows == 1..3
Contexts == {0,2}
Paths == {<<1>>, <<2>>, <<2,1>>, <<2,2>>}
VARIABLES path, advantage, head, phase, seen, grad, count, resumed, snapshot
vars == <<path, advantage, head, phase, seen, grad, count, resumed, snapshot>>
Zero == [c \in Contexts |-> 0]
Blank == <<{},Zero,0>>
Init ==
  /\ path \in [Rows -> Paths]
  /\ advantage \in [Rows -> {-1,0,1}]
  /\ head \in {1,2}
  /\ phase = "run" /\ seen = {} /\ grad = Zero /\ count = 0
  /\ resumed = FALSE /\ snapshot = Blank

(* Advantage units are log(2), temperature is one and the cap is one.
   These integers represent twice the clipped weight. Policy probabilities
   have denominator three and token counts divide two. *)
Weight(i) == IF advantage[i] < 0 THEN 1 ELSE 2
Prob(c) == IF c = 0 THEN head ELSE 3-head
Prefix(i,t) == IF t = 1 THEN 0 ELSE path[i][1]
Score(c,token) == Prob(c) - (IF token = 1 THEN 3 ELSE 0)
Term(i,t,c) == IF t <= Len(path[i]) /\ Prefix(i,t) = c
               THEN Score(c,path[i][t]) ELSE 0
RowGrad(i,c) == (Weight(i)*2*(Term(i,1,c)+Term(i,2,c))) \div Len(path[i])
Total(f) == f[1]+f[2]+f[3]
Batch(batch) ==
  /\ phase = "run" /\ batch \in SUBSET (Rows \ seen)
  /\ Cardinality(batch) \in 1..2
  /\ seen' = seen \cup batch /\ count' = count + Cardinality(batch)
  /\ grad' = [c \in Contexts |-> grad[c] +
        Total([i \in Rows |-> IF i \in batch THEN RowGrad(i,c) ELSE 0])]
  /\ UNCHANGED <<path,advantage,head,phase,resumed,snapshot>>
Pause ==
  /\ phase = "run" /\ ~resumed /\ seen # {} /\ seen # Rows
  /\ snapshot' = <<seen,grad,count>> /\ seen' = {} /\ grad' = Zero /\ count' = 0
  /\ phase' = "paused" /\ resumed' = TRUE
  /\ UNCHANGED <<path,advantage,head>>
Restore ==
  /\ phase = "paused" /\ phase' = "run"
  /\ seen' = snapshot[1] /\ grad' = snapshot[2] /\ count' = snapshot[3]
  /\ UNCHANGED <<path,advantage,head,resumed,snapshot>>
Commit ==
  /\ phase = "run" /\ seen = Rows /\ phase' = "done"
  /\ UNCHANGED <<path,advantage,head,seen,grad,count,resumed,snapshot>>
Next == (\E b \in SUBSET Rows : Batch(b)) \/ Pause \/ Restore \/ Commit
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent count table for each of the four possible logged actions.
   Mass is in quarters and the expected gradient numerator in twelfths.
   The optimizer divides this numerator by the actual action count. *)
Visits(i,c,a) ==
  CASE path[i] = <<1>> -> IF c = 0 /\ a = 1 THEN 2 ELSE 0
    [] path[i] = <<2>> -> IF c = 0 /\ a = 2 THEN 2 ELSE 0
    [] OTHER -> IF (c = 0 /\ a = 2) \/ (c = 2 /\ a = path[i][2]) THEN 1 ELSE 0
Mass(s,c,a) == Total([i \in Rows |-> IF i \in s THEN
     (CASE advantage[i] = -1 -> 1 [] advantage[i] = 0 -> 2 [] OTHER -> 2)*Visits(i,c,a)
     ELSE 0])
Matches(s,g,n) == /\ n = Cardinality(s)
                  /\ \A c \in Contexts : g[c] =
                     (Mass(s,c,1)+Mass(s,c,2))*Prob(c)-3*Mass(s,c,1)
Correct == IF phase = "paused" THEN Matches(snapshot[1],snapshot[2],snapshot[3])
           ELSE Matches(seen,grad,count)
Complete == <>(phase = "done")
=============================================================================
