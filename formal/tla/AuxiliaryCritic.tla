------------------------- MODULE AuxiliaryCritic -------------------------
EXTENDS Integers
(* Two sampled scalar updates. Weights are in units of 1/48. The clipped
   gradients are +/-1; beta1=beta2=1/2, lr=2, epsilon=1, decay=1/4.
   The corrected second moment is one, so both AdamW updates are exact on
   this grid. The frozen target is published with tau=1/2. *)
VARIABLES seed, cursor, clock, v, q, frozen, label, phase, saved, resumed
vars == <<seed,cursor,clock,v,q,frozen,label,phase,saved,resumed>>
Head(w) == [w |-> w, m |-> 0, energy |-> 0, t |-> 0]
First == Head(96)
Second == Head(-48)
Other(i) == 3-i
Feature(i) == IF i=1 THEN -1 ELSE 1
Backup(i,f) == IF i=1 THEN 384+(f \div 2) ELSE -384
Sign(x) == IF x>0 THEN 1 ELSE IF x<0 THEN -1 ELSE 0
Gradient(w,x,y) == Sign((w*x-y)*x)
Move(h,g) ==
    LET m == h.m+2^h.t*g
        energy == h.energy+2^h.t*g^2
    IN [w |-> (h.w \div 2)-((48*m) \div (2^(h.t+1)-1)),
        m |-> m, energy |-> energy, t |-> h.t+1]
Init == /\ seed \in {1,2} /\ cursor=seed /\ clock=0 /\ phase="ready"
        /\ v=First /\ q=Second /\ frozen=-48 /\ label=0 /\ saved=0 /\ resumed=FALSE
Read == /\ phase="ready" /\ clock<2 /\ label'=Backup(cursor,frozen)
        /\ phase'="value" /\ UNCHANGED <<seed,cursor,clock,v,q,frozen,saved,resumed>>
Value == /\ phase="value" /\ v'=Move(v,Gradient(v.w,1,label)) /\ phase'="q"
         /\ UNCHANGED <<seed,cursor,clock,q,frozen,label,saved,resumed>>
ActionValue == /\ phase="q" /\ q'=Move(q,Gradient(q.w,Feature(cursor),label)) /\ phase'="publish"
               /\ UNCHANGED <<seed,cursor,clock,v,frozen,label,saved,resumed>>
Publish == /\ phase="publish" /\ frozen'=(v.w+frozen) \div 2 /\ clock'=clock+1
           /\ cursor'=Other(cursor) /\ phase'="ready"
           /\ UNCHANGED <<seed,v,q,label,saved,resumed>>
Pause == /\ phase="ready" /\ clock=1 /\ ~resumed
         /\ saved' = <<v,q,frozen,cursor,clock>> /\ phase'="paused"
         /\ v'=Head(0) /\ q'=Head(0) /\ frozen'=0 /\ cursor'=0 /\ clock'=0
         /\ UNCHANGED <<seed,label,resumed>>
Resume == /\ phase="paused" /\ v'=saved[1] /\ q'=saved[2] /\ frozen'=saved[3]
          /\ cursor'=saved[4] /\ clock'=saved[5]
          /\ phase'="ready" /\ resumed'=TRUE /\ UNCHANGED <<seed,label,saved>>
Next == Read \/ Value \/ ActionValue \/ Publish \/ Pause \/ Resume
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent closed-form weights and moments, by original sampling order. *)
G(i,isQ) == IF isQ THEN 1 ELSE IF i=1 THEN -1 ELSE 1
Weight(t,isQ) ==
    LET w == IF isQ THEN -48 ELSE 96
        a == G(seed,isQ)
        b == G(Other(seed),isQ)
    IN CASE t=0 -> w [] t=1 -> (w \div 2)-48*a
         [] OTHER -> (w \div 4)-24*a-16*(a+2*b)
Correct(h,t,isQ) ==
    /\ h.w=Weight(t,isQ) /\ h.t=t /\ h.energy=2^t-1
    /\ h.m=(CASE t=0 -> 0 [] t=1 -> G(seed,isQ) [] OTHER -> G(seed,isQ)+2*G(Other(seed),isQ))
Target(t) ==
    CASE t=0 -> -48 [] t=1 -> (-48+Weight(1,FALSE)) \div 2
      [] OTHER -> (-48+Weight(1,FALSE)+2*Weight(2,FALSE)) \div 4
CorrectUpdate == phase # "paused" =>
    /\ cursor=(IF clock=1 THEN Other(seed) ELSE seed)
    /\ frozen=Target(clock)
    /\ Correct(v,clock+(IF phase \in {"q","publish"} THEN 1 ELSE 0),FALSE)
    /\ Correct(q,clock+(IF phase="publish" THEN 1 ELSE 0),TRUE)
    /\ (phase \in {"value","q","publish"} => label=Backup(cursor,Target(clock)))
Completes == <>(clock=2 /\ phase="ready")
=============================================================================
