--------------------------- MODULE PromptReduction ---------------------------
EXTENDS Integers, FiniteSets
(* Two prompts with unequal response counts. Policy units are a signed loss
   per token, while KL units are per response. Cross multiplication checks
   the resulting rational objectives without TLC integer division. *)
VARIABLES length, policy, kl, seen, policySum, tokens, klSum, rows, promptSum
vars == <<length, policy, kl, seen, policySum, tokens, klSum, rows, promptSum>>
Rows == 1..3
GroupRow(g,i) == IF g = 1 THEN i = 1 ELSE i \in {2,3}
Len(g) == IF g = 1 THEN length[1] ELSE length[2]+length[3]
Size(g) == IF g = 1 THEN 1 ELSE 2
Pol(i) == length[i]*policy[i]
Native(i) == (IF length[i] = 0 THEN 0 ELSE policy[i]) + kl[i]

Init == /\ length \in [Rows -> 0..2]
        /\ policy \in [Rows -> {-1,1}]
        /\ kl \in [Rows -> 0..1]
        /\ \A i \in Rows : length[i] = 0 => kl[i] = 0
        /\ seen = {} /\ policySum = 0 /\ tokens = 0
        /\ klSum = 0 /\ rows = 0 /\ promptSum = 0
Add(g) == /\ g \in {1,2} \ seen
          /\ seen' = seen \cup {g}
          /\ policySum' = policySum + (IF g = 1 THEN Pol(1) ELSE Pol(2)+Pol(3))
          /\ tokens' = tokens + Len(g)
          /\ klSum' = klSum + (IF g = 1 THEN kl[1] ELSE kl[2]+kl[3])
          /\ rows' = rows + Size(g)
          /\ promptSum' = promptSum + (IF g = 1 THEN 2*Native(1) ELSE Native(2)+Native(3))
          /\ UNCHANGED <<length,policy,kl>>
Next == (\E g \in {1,2} : Add(g)) \/ (seen = {1,2} /\ UNCHANGED vars)
Included == {i \in Rows : \E g \in seen : GroupRow(g,i)}
Active == {p \in Rows \X (1..2) : p[1] \in Included /\ p[2] <= length[p[1]]}
SignedTokens == Cardinality({p \in Active : policy[p[1]] = 1})
                - Cardinality({p \in Active : policy[p[1]] = -1})
KLMass == Cardinality({i \in Included : kl[i] = 1})
Den(n) == IF n = 0 THEN 1 ELSE n
Equal(a,b,c,d) == a*d = c*b
(* Policy/token + KL/row and policy/(2*row) + KL/row. *)
TokenObjective == Equal(policySum*Den(rows)+klSum*Den(tokens), Den(tokens)*Den(rows),
                        SignedTokens*Den(Cardinality(Included))+KLMass*Den(Cardinality(Active)),
                        Den(Cardinality(Active))*Den(Cardinality(Included)))
FixedObjective == Equal(policySum+2*klSum, 2*Den(rows),
                        SignedTokens+2*KLMass, 2*Den(Cardinality(Included)))
ByPrompt(i) == IF i \notin Included THEN 0 ELSE
                (IF i = 1 THEN 2 ELSE 1) *
                ((IF <<i,1>> \in Active THEN policy[i] ELSE 0) + kl[i])
NativeObjective == promptSum = ByPrompt(1)+ByPrompt(2)+ByPrompt(3)
Coverage == tokens = Cardinality(Active) /\ rows = Cardinality(Included)
=============================================================================
