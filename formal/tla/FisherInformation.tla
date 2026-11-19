------------------------- MODULE FisherInformation -------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

\* Two-action bigram policy at equal logits. Each coordinate controls action 1
\* at one predecessor token. A score tick is twice the actual NLL derivative.
Coords == 0..1
Zero == [i \in Coords |-> 0]
Turn(p, r) == [prompt |-> p, response |-> r]
Rows == <<
  <<Turn(<<1,0>>, <<1,1>>), Turn(<<1>>, <<0>>)>>,
  <<Turn(<<0>>, <<0>>)>>,
  <<Turn(<<0>>, <<1>>), Turn(<<0>>, <<0>>)>>,
  <<Turn(<<0>>, <<1,0,0>>)>>,
  <<Turn(<<1>>, <<1>>), Turn(<<>>, <<1>>)>>,
  <<Turn(<<1>>, <<>>)>>,
  <<>>,
  <<Turn(<<0,1>>, <<1,1>>)>>
>>
\* Independent enumeration of six times the mean score ticks, and the
\* count of supervised positions for each coordinate. Bad rows have no vote.
Pair(a,b) == [i \in Coords |-> IF i=0 THEN a ELSE b]
Means == <<Pair(-2,0), Pair(6,0), Zero, Pair(0,2), Zero, Zero, Zero, Pair(0,-6)>>
Active == <<Pair(1,2), Pair(1,0), Pair(2,0), Pair(2,1), Zero, Zero, Zero, Pair(0,2)>>
Good == {1,2,3,4,8}
VARIABLES done, job, turn, pos, tokens, score, accepted, diag, phase
vars == <<done, job, turn, pos, tokens, score, accepted, diag, phase>>
Init == /\ done = <<>> /\ job=1 /\ turn=1 /\ pos=1 /\ tokens=0
        /\ score=Zero /\ accepted=0 /\ diag=Zero /\ phase="choose"
Start == /\ phase="choose" /\ \E j \in DOMAIN Rows : job'=j
         /\ phase'="read" /\ turn'=1 /\ pos'=1 /\ tokens'=0 /\ score'=Zero
         /\ UNCHANGED <<done, accepted, diag>>
NextTurn == /\ turn'=turn+1 /\ pos'=1
SameTurn == /\ turn'=turn /\ pos'=pos+1
Read ==
  /\ phase="read" /\ turn <= Len(Rows[job])
  /\ LET row == Rows[job][turn]
     IN IF Len(row.prompt)=0 \/ Len(row.response)=0
        THEN /\ phase'="reject" /\ UNCHANGED <<turn, pos, tokens, score>>
        ELSE /\ LET prev == IF pos=1 THEN row.prompt[Len(row.prompt)] ELSE row.response[pos-1]
                 IN score'=[score EXCEPT ![prev] = @ + (IF row.response[pos]=0 THEN 1 ELSE -1)]
             /\ tokens'=tokens+1 /\ phase'=phase
             /\ IF pos=Len(row.response) THEN NextTurn ELSE SameTurn
  /\ UNCHANGED <<done, job, accepted, diag>>
Finish ==
  /\ phase="reject" \/ (phase="read" /\ turn>Len(Rows[job]))
  /\ IF phase="read" /\ tokens>0
     THEN /\ accepted'=accepted+1
          /\ diag'=[i \in Coords |-> (accepted*diag[i] +
                2*((6*score[i]) \div tokens)^2) \div accepted']
     ELSE UNCHANGED <<accepted, diag>>
  /\ done'=Append(done, job)
  /\ phase'=IF Len(done')=2 THEN "end" ELSE "choose"
  /\ UNCHANGED <<job, turn, pos, tokens, score>>
Next == Start \/ Read \/ Finish
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Votes == {k \in DOMAIN done : done[k] \in Good}
RECURSIVE SumSq(_,_)
SumSq(k,i) == IF k=0 THEN 0 ELSE SumSq(k-1,i) + Means[done[k]][i]^2
Population ==
  /\ accepted=Cardinality(Votes)
  /\ \A i \in Coords : accepted*diag[i]=2*SumSq(Len(done),i)
ScoreLaw == (phase="read" /\ turn>Len(Rows[job]) /\ tokens>0) =>
  /\ job \in Good
  /\ \A i \in Coords :
       /\ 6*score[i]=Means[job][i]*tokens
       /\ score[i]^2 <= tokens*Active[job][i]
Finished == <>(phase="end")
=============================================================================
