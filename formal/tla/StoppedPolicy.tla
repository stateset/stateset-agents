-------------------------- MODULE StoppedPolicy ---------------------------
EXTENDS Naturals, Sequences, FiniteSets
(* At the root, h/4 continues under both policies. The child changes
   from (1/4,3/4) to (3/4,1/4). A root EOS ends the row immediately.
   Each history denotes its exact number of equally likely ticket pairs.
   Drift is in units of 1/3000, bracketed by PPOBatch.epoch_table. *)
VARIABLES h, batch, rows, atChild, tokens, drift, mass
vars == <<h,batch,rows,atChild,tokens,drift,mass>>
Caps == {0,100,200,350,600,750,900,1200,1500,1800}
Init == /\ h \in 1..3 /\ batch \in 1..3 /\ rows = <<>> /\ atChild = FALSE
        /\ tokens = 0 /\ drift = 0 /\ mass = 1
More == Len(rows) < batch
EOS == /\ More /\ ~atChild /\ rows' = Append(rows,"eos")
       /\ tokens' = tokens+1 /\ mass' = mass*(4-h)*4 /\ UNCHANGED <<atChild,drift>>
Continue == /\ More /\ ~atChild /\ atChild' = TRUE /\ tokens' = tokens+1
            /\ mass' = mass*h /\ UNCHANGED <<rows,drift>>
Child == /\ atChild /\ atChild' = FALSE /\ tokens' = tokens+1
         /\ \E a \in {"rare","common"} :
              /\ rows' = Append(rows,a)
              /\ drift' = drift + (IF a="rare" THEN 2704 ELSE 1296)
              /\ mass' = mass*(IF a="rare" THEN 1 ELSE 3)
Next == (EOS \/ Continue \/ Child) /\ UNCHANGED <<h,batch>>
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Count(a) == Cardinality({i \in DOMAIN rows : rows[i]=a})
Rare == Count("rare")
Common == Count("common")
Act(ticket) == IF ticket \div 4 >= h THEN "eos"
               ELSE IF ticket % 4 = 0 THEN "rare" ELSE "common"
Ways(a) == Cardinality({t \in 0..15 : Act(t)=a})
TicketLaw == mass = Ways("eos")^Count("eos") * Ways("rare")^Rare *
                     Ways("common")^Common * (IF atChild THEN h ELSE 1)
ActionCount == tokens = Len(rows) + Rare + Common + (IF atChild THEN 1 ELSE 0)
Decision == ~More =>
  /\ ~atChild /\ \A cap \in Caps :
     IF drift <= tokens*cap THEN 2706*Rare + 1297*Common <= tokens*cap
     ELSE 2703*Rare + 1294*Common > tokens*cap
Miss == (~More /\ drift=0) <=>
        (~More /\ Count("eos")=batch /\ mass=(16-4*h)^batch)
Done == <>(~More)
=============================================================================
