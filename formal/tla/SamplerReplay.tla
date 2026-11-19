--------------------------- MODULE SamplerReplay ---------------------------
EXTENDS Integers, Sequences, FiniteSets
(* Two prompts have one and two initial tokens. The raw categorical head
   has weights 2:1. Temperature is one or one half before optional top-p
   truncation to the most likely token. Token 2 terminates a response. *)
Rows == 1..2
Ids == 1..2
Columns == 1..3
VARIABLES sharp, narrow, empty, tokens, recorded, phase, frozen, head, lastHead, ratios, steps
vars == <<sharp,narrow,empty,tokens,recorded,phase,frozen,head,lastHead,ratios,steps>>
Blank == [r \in Rows |-> [c \in Columns |-> 0]]
End(r) == r \in empty \/ Len(tokens[r]) = 2 \/ (Len(tokens[r]) > 0 /\ tokens[r][Len(tokens[r])] = 2)
Prefix(r,k) == IF k=1 THEN r ELSE tokens[r][k-1]
Preferred(r,k,h) == IF h=0 THEN Prefix(r,k) ELSE 3-Prefix(r,k)
Mass(r,k,a) ==
    IF narrow /\ a # Preferred(r,k,0) THEN 0
    ELSE IF a=Preferred(r,k,0) THEN (IF sharp THEN 4 ELSE 2) ELSE 1
(* A sampled probability in fifteenths, computed by normalizing surviving weights. *)
DrawScore(r,k,a) == (15*Mass(r,k,a)) \div (Mass(r,k,1)+Mass(r,k,2))
Raw(r,k,a,h) == IF a=Preferred(r,k,h) THEN 10 ELSE 5
(* Independently tabulated conditional laws for the original prefix. *)
Truth(r,k) == LET favored == tokens[r][k]=Prefix(r,k) IN
    IF narrow THEN (IF favored THEN 15 ELSE 0)
    ELSE IF sharp THEN (IF favored THEN 12 ELSE 3) ELSE (IF favored THEN 10 ELSE 5)
Active(r,c) == r <= c /\ c < r+Len(tokens[r])
Init == /\ sharp \in BOOLEAN /\ narrow \in BOOLEAN /\ empty \in SUBSET Rows
        /\ tokens=[r \in Rows |-> <<>>] /\ recorded=tokens
        /\ phase="collect" /\ frozen=Blank /\ head=0 /\ lastHead=0 /\ ratios=Blank /\ steps=0
Draw(r,a) == LET k == Len(tokens[r])+1 IN
    /\ phase="collect" /\ ~End(r) /\ Mass(r,k,a)>0
    /\ tokens'=[tokens EXCEPT ![r]=Append(@,a)]
    /\ recorded'=[recorded EXCEPT ![r]=Append(@,DrawScore(r,k,a))]
    /\ UNCHANGED <<sharp,narrow,empty,phase,frozen,head,lastHead,ratios,steps>>
Pack == /\ phase="collect" /\ \A r \in Rows : End(r)
        /\ phase'=IF empty=Rows THEN "done" ELSE "learn"
        /\ head' \in {0,1}
        /\ frozen'=[r \in Rows |-> [c \in Columns |->
             IF Active(r,c) THEN recorded[r][c-r+1] ELSE 0]]
        /\ UNCHANGED <<sharp,narrow,empty,tokens,recorded,lastHead,ratios,steps>>
Update == /\ phase="learn" /\ lastHead'=head /\ head' \in {0,1}
          /\ ratios'=[r \in Rows |-> [c \in Columns |->
               IF Active(r,c) THEN
                  (60*Raw(r,c-r+1,tokens[r][c-r+1],head)) \div frozen[r][c]
               ELSE 0]]
          /\ steps'=steps+1 /\ phase'=IF steps'=2 THEN "done" ELSE "learn"
          /\ UNCHANGED <<sharp,narrow,empty,tokens,recorded,frozen>>
Next == (\E r \in Rows, a \in Ids : Draw(r,a)) \/ Pack \/ Update
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Provenance == \A r \in Rows : /\ Len(recorded[r])=Len(tokens[r])
    /\ \A k \in 1..Len(tokens[r]) : recorded[r][k]=Truth(r,k) /\ Truth(r,k)>0
Alignment == phase#"collect" => \A r \in Rows, c \in Columns :
    frozen[r][c] = IF Active(r,c) THEN Truth(r,c-r+1) ELSE 0
Measure == steps>0 => \A r \in Rows, c \in Columns :
    IF Active(r,c) THEN LET k == c-r+1 IN
        ratios[r][c]*Truth(r,k) = 60*Raw(r,k,tokens[r][k],lastHead)
    ELSE ratios[r][c]=0
CommitCount == phase="done" => steps=(IF (\A r \in Rows : Len(tokens[r]) = 0) THEN 0 ELSE 2)
Done == <>(phase="done")
=============================================================================
