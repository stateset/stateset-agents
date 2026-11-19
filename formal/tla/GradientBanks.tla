--------------------------- MODULE GradientBanks -----------------------------
EXTENDS Integers
(* Two ranks and two local backwards per rank. Each input gives the number of
   responses and tokens, followed by the local policy and entropy gradients
   (one parameter coordinate of each mean loss). Empty ranks contribute zero. *)
Ranks == 1..2
Steps == 1..2
Samples == {<<0,0,0,0>>, <<1,1,-1,1>>, <<1,3,1,-1>>, <<2,3,0,1>>}
VARIABLES data, used, policy, entropy, finished, result
vars == <<data, used, policy, entropy, finished, result>>
InitialScale(k) == 2 * ((data[1][1][k] + data[2][1][k] + 1) \div 2)
(* 240 represents all possible fixed-scale backward contributions exactly. *)
Base == 240
Init == /\ data \in [Ranks -> [Steps -> Samples]]
        /\ InitialScale(1) > 0
        /\ used = [r \in Ranks |-> 0]
        /\ policy = [r \in Ranks |-> 0] /\ entropy = policy
        /\ finished = FALSE /\ result = 0

Backward(r) ==
    /\ used[r] < 2
    /\ LET s == data[r][used[r]+1] IN
       /\ policy' = [policy EXCEPT ![r] = @ + s[3]*s[1]*(Base \div InitialScale(1))]
       /\ entropy' = [entropy EXCEPT ![r] = @ + s[4]*s[2]*(Base \div InitialScale(2))]
    /\ used' = [used EXCEPT ![r] = @+1]
    /\ UNCHANGED <<data,finished,result>>

Total(k) == data[1][1][k] + data[1][2][k] + data[2][1][k] + data[2][2][k]
AllUsed == \A r \in Ranks : used[r] = 2
Finalize ==
    /\ ~finished /\ AllUsed
    /\ LET scaled(bank,k) == (bank[1]+bank[2]) * InitialScale(k)
       IN result' = scaled(policy,1)*Total(2) + scaled(entropy,2)*Total(1)
    /\ finished' = TRUE
    /\ UNCHANGED <<data,used,policy,entropy>>
Next == (\E r \in Ranks : Backward(r)) \/ Finalize
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Compare against the two independently weighted averages on the concatenated
   samples by multiplying through by Base * total responses * total tokens. *)
Part(s,k) == s[k]*s[k+2]
Reference(k) == Part(data[1][1],k) + Part(data[1][2],k)
                + Part(data[2][1],k) + Part(data[2][2],k)
MatchesGlobalGradient == finished =>
    AllUsed /\ result = Base*(Reference(1)*Total(2) + Reference(2)*Total(1))
Completes == <>finished
=============================================================================
