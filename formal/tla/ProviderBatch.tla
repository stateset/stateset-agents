---------------------------- MODULE ProviderBatch ----------------------------
EXTENDS Integers, FiniteSets
(* Two River datums, each with a prompt and two sampled tokens. The client
   truncates the input, aligns credits to predictions, scales once, and sends
   any partition of retained rows before one optimizer step. Units of 12
   represent thirds and quarters exactly. An incoming gradient bank may be dirty. *)
VARIABLES prompt, limit, credit, mode, sent, bank, phase, update, steps
vars == <<prompt,limit,credit,mode,sent,bank,phase,update,steps>>
Rows == 1..2
Slots == 1..2
Min(a,b) == IF a <= b THEN a ELSE b
N(i) == Min(limit, prompt[i]+2)
Prefix(i) == Min(prompt[i], N(i))
Wire(i,j) == IF j < Prefix(i) \/ j >= N(i) THEN 0
             ELSE credit[i][j-Prefix(i)+1]
Count(i) == Cardinality({j \in 1..N(i) : Wire(i,j) # 0})
Live == {i \in Rows : Count(i) > 0}
Total == Count(1) + Count(2)
Den(i) == CASE mode = "sum" -> 1 [] mode = "token" -> Total
            [] OTHER -> Cardinality(Live)*Count(i)
(* A nonconstant, signed test direction in the model Jacobian detects shifts. *)
Direction(i,j) == 2*i-j
Contribution(i,j) == IF Wire(i,j) = 0 THEN 0
                     ELSE (12 \div Den(i)) * Wire(i,j) * Direction(i,j)
Row(i) == Contribution(i,1)+Contribution(i,2)+Contribution(i,3)+Contribution(i,4)
Batch(s) == (IF 1 \in s THEN Row(1) ELSE 0)+(IF 2 \in s THEN Row(2) ELSE 0)

Init == /\ prompt \in [Rows -> 1..2] /\ limit \in 1..4
        /\ credit \in [Rows -> [Slots -> {-1,0,1}]]
        /\ mode \in {"token","sequence","sum"}
        /\ sent = {} /\ bank = 17 /\ phase = "collect" /\ update = 0 /\ steps = 0
Backward(s) == /\ phase = "collect" /\ s # {} /\ s \subseteq (Live \ sent)
               /\ bank' = (IF sent = {} THEN 0 ELSE bank) + Batch(s)
               /\ sent' = sent \cup s /\ UNCHANGED <<phase,update,steps>>
Finish == /\ phase = "collect" /\ sent = Live
          /\ phase' = "done" /\ update' = IF sent = {} THEN 0 ELSE bank
          /\ steps' = IF sent = {} THEN steps ELSE steps+1
          /\ UNCHANGED <<sent,bank>>
Move == (\E s \in SUBSET Rows : Backward(s)) \/ Finish
Next == /\ Move /\ UNCHANGED <<prompt,limit,credit,mode>>
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent oracle enumerates retained sampled target IDs, without
   materializing the prediction-position layout or using its counts. *)
Active == {a \in Rows \X Slots : prompt[a[1]]+a[2] <= limit /\ credit[a[1]][a[2]] # 0}
RefRows == {a[1] : a \in Active}
Reference(i,t) ==
    IF <<i,t>> \notin Active THEN 0 ELSE
    LET denom == CASE mode = "sum" -> 1 [] mode = "token" -> Cardinality(Active)
                   [] OTHER -> Cardinality(RefRows)*Cardinality({a \in Active : a[1]=i})
    IN (12 \div denom)*credit[i][t]*(2*i-(prompt[i]+t-1))
SumRef(i) == Reference(i,1)+Reference(i,2)
Ref(s) == (IF 1 \in s THEN SumRef(1) ELSE 0)+(IF 2 \in s THEN SumRef(2) ELSE 0)
Alignment == /\ Live = RefRows /\ Total = Cardinality(Active)
             /\ \A i \in Rows : Wire(i,N(i)) = 0
Accumulation == sent = {} \/ bank = Ref(sent)
Commit == phase = "done" => update = Ref(Rows)
StepCount == steps = (IF phase = "done" /\ RefRows # {} THEN 1 ELSE 0)
Progress == <>(phase = "done")
=============================================================================
