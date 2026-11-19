------------------------ MODULE DomainAdversarial --------------------------
EXTENDS Integers, FiniteSets
(* A balanced binary domain loss with unequal source/target minibatches.
   Logits at each feature bin are -log(2), 0, or log(2). Their real-domain
   probabilities in sixths are exactly 2, 3, or 4. Gain is twice lambda.
   Scale 72 represents every domain mean and every allowed reversal exactly. *)
Rows == 1..3
Bins == {0,1}
Scale == 72
Task == 12
VARIABLES source, bin, score, gain, phase, seen, featureGrad, criticGrad,
          encoder, discriminator
vars == <<source,bin,score,gain,phase,seen,featureGrad,criticGrad,encoder,discriminator>>
RealRows == Rows \ source
Count(i) == IF i \in source THEN Cardinality(source) ELSE Cardinality(RealRows)
Residual(i) == score[bin[i]] - (IF i \in source THEN 0 ELSE 6)
FeatureJac(i) == i-2
CriticJac(i) == 1+bin[i]
Grad(i,J(_)) == (Scale*Residual(i)*J(i)) \div (6*Count(i))
Init == /\ source \in SUBSET Rows /\ bin \in [Rows -> Bins] /\ score \in [Bins -> 2..4]
        /\ gain \in {0,1,2,4} /\ phase="collect" /\ seen={}
        /\ encoder=3 /\ discriminator=-2 /\ featureGrad=0 /\ criticGrad=0
Ready == source # {} /\ RealRows # {}
Visit(i) == /\ phase="collect" /\ Ready /\ i \in Rows \ seen /\ seen'=seen \cup {i}
            /\ featureGrad'=featureGrad-(gain*Grad(i,FeatureJac)) \div 2
            /\ criticGrad'=criticGrad+Grad(i,CriticJac)
            /\ UNCHANGED <<source,bin,score,gain,phase,encoder,discriminator>>
Commit == /\ phase="collect" /\ seen=Rows /\ phase'="done"
          /\ encoder'=encoder-Task-featureGrad /\ discriminator'=discriminator-criticGrad
          /\ UNCHANGED <<source,bin,score,gain,seen,featureGrad,criticGrad>>
Reject == /\ phase="collect" /\ ~Ready /\ phase'="rejected"
          /\ UNCHANGED <<source,bin,score,gain,seen,featureGrad,criticGrad,encoder,discriminator>>
Next == Commit \/ Reject \/ (\E i \in Rows : Visit(i))
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Independent calculation groups the logit Jacobians by feature value and
   domain before differentiating the two domain-mean likelihoods. *)
Jac(S,b,J(_)) ==
    (IF 1 \in S /\ bin[1]=b THEN J(1) ELSE 0) +
    (IF 2 \in S /\ bin[2]=b THEN J(2) ELSE 0) +
    (IF 3 \in S /\ bin[3]=b THEN J(3) ELSE 0)
Domain(S,D,y,J(_)) == IF D={} THEN 0 ELSE
    (Scale*((score[0]-y)*Jac(S \cap D,0,J)+(score[1]-y)*Jac(S \cap D,1,J)))
      \div (6*Cardinality(D))
Slope(S,J(_)) == Domain(S,source,0,J)+Domain(S,RealRows,6,J)
Correct == /\ featureGrad=-(gain*Slope(seen,FeatureJac)) \div 2
           /\ criticGrad=Slope(seen,CriticJac)
           /\ encoder=IF phase="done" THEN 3-Task+(gain*Slope(Rows,FeatureJac)) \div 2 ELSE 3
           /\ discriminator=IF phase="done" THEN -2-Slope(Rows,CriticJac) ELSE -2
           /\ phase="done" => Ready /\ seen=Rows
           /\ phase="rejected" => ~Ready /\ seen={}
Finished == <>(phase \in {"done","rejected"})
=============================================================================
