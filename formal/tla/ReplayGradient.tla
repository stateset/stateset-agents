-------------------------- MODULE ReplayGradient --------------------------
EXTENDS Integers, FiniteSets, Sequences
Actions == 1..3
Pairs == Actions \X Actions
Total == 12
Behavior == <<6,3,3>>
Heads == <<<<4,4,4>>, <<3,6,3>>>>
VARIABLES mode, coord, epoch, phase, frozen, seen, ascent
vars == <<mode, coord, epoch, phase, frozen, seen, ascent>>
P == Heads[epoch]
Score(i) == (IF i=coord THEN Total ELSE 0)-P[coord]
Adv(i,j) == IF i>j THEN 1 ELSE IF i<j THEN -1 ELSE 0
Init == /\ mode \in {"uncorrected","row","clipped","joint"}
        /\ coord \in Actions /\ epoch=1 /\ phase="pack"
        /\ frozen = <<>> /\ seen={} /\ ascent=0
Pack == /\ phase="pack" /\ frozen'=Behavior /\ phase'="sum"
        /\ UNCHANGED <<mode,coord,epoch,seen,ascent>>
Ratio(i) == (Total*P[i]) \div frozen[i]
Term(i,j) ==
  LET a == Adv(i,j)
      r == Ratio(i)
      weight == CASE mode="uncorrected" -> Total*Total
                  [] mode="joint" -> r*Ratio(j)
                  [] mode="clipped" /\ ((a>0 /\ 5*r>6*Total) \/ (a<0 /\ 5*r<4*Total)) -> 0
                  [] OTHER -> r*Total
  IN Behavior[i]*Behavior[j]*weight*Score(i)*a
Visit(pair) == /\ phase="sum" /\ pair \in Pairs \ seen
               /\ seen'=seen \cup {pair}
               /\ ascent'=ascent+Term(pair[1],pair[2])+Term(pair[2],pair[1])
               /\ UNCHANGED <<mode,coord,epoch,phase,frozen>>
Finish == /\ phase="sum" /\ seen=Pairs
          /\ phase'=IF epoch=1 THEN "sum" ELSE "done"
          /\ epoch'=2 /\ seen'={} /\ ascent'=0
          /\ UNCHANGED <<mode,coord,frozen>>
Next == Pack \/ Finish \/ (\E pair \in Pairs : Visit(pair))
(* Independently sum the better- and worse-reward peer masses for each row.
   All quantities are in integer units of 2*Total^5 per mean gradient. *)
Below(i,w) == IF i=1 THEN 0 ELSE IF i=2 THEN w[1] ELSE w[1]+w[2]
Above(i,w) == Total-w[i]-Below(i,w)
Oracle(i) ==
  LET peers == IF mode="joint" THEN P ELSE Behavior
      own == IF mode="uncorrected" THEN Behavior[i] ELSE P[i]
      plus == IF mode="clipped" /\ 5*P[i]>6*Behavior[i] THEN 0 ELSE Below(i,peers)
      minus == IF mode="clipped" /\ 5*P[i]<4*Behavior[i] THEN 0 ELSE Above(i,peers)
  IN own*Score(i)*(plus-minus)
Provenance == phase="pack" \/ frozen=Behavior
Correct == seen=Pairs => ascent=2*Total*Total*(Oracle(1)+Oracle(2)+Oracle(3))
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Finished == <>(phase="done")
=============================================================================
