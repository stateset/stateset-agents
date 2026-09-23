---------------------------- MODULE AutoResearch ----------------------------
EXTENDS Naturals, Integers, FiniteSets

CONSTANT Experiments, Scores, Direction
ModelScores == {-2, -1, 0, 1, 2}
NoScore == 999
ASSUME /\ Experiments # {}
       /\ IsFiniteSet(Experiments)
       /\ Scores # {}
       /\ Scores \subseteq Int
       /\ IsFiniteSet(Scores)
       /\ NoScore \notin Scores
       /\ Direction \in {"maximize", "minimize"}

VARIABLES phase, active, recorded, crashed, bestScore, bestId, modelId,
          candidateScore
vars == <<phase, active, recorded, crashed, bestScore, bestId, modelId,
          candidateScore>>

None == "none"
Better(s) == IF bestId = None THEN TRUE
             ELSE IF Direction = "maximize" THEN s > bestScore ELSE s < bestScore

Init == /\ phase = "idle"
        /\ active = None
        /\ recorded = {}
        /\ crashed = {}
        /\ bestScore = NoScore
        /\ bestId = None
        /\ modelId = None
        /\ candidateScore = NoScore

Begin(e) == /\ phase = "idle"
            /\ e \in Experiments \ recorded
            /\ active' = e
            /\ modelId' = e
            /\ phase' = "training"
            /\ UNCHANGED <<recorded, crashed, bestScore, bestId, candidateScore>>

Evaluate(s) == /\ phase = "training"
               /\ s \in Scores
               /\ candidateScore' = s
               /\ phase' = "evaluated"
               /\ UNCHANGED <<active, recorded, crashed, bestScore, bestId,
                              modelId>>

Keep == /\ phase = "evaluated"
        /\ Better(candidateScore)
        /\ bestScore' = candidateScore
        /\ bestId' = active
        /\ recorded' = recorded \cup {active}
        /\ active' = None
        /\ phase' = "idle"
        /\ candidateScore' = NoScore
        /\ UNCHANGED <<modelId, crashed>>

Discard == /\ phase = "evaluated"
           /\ ~Better(candidateScore)
           /\ modelId' = bestId
           /\ recorded' = recorded \cup {active}
           /\ active' = None
           /\ phase' = "idle"
           /\ candidateScore' = NoScore
           /\ UNCHANGED <<bestScore, bestId, crashed>>

Crash == /\ phase \in {"training", "evaluated"}
         /\ modelId' = bestId
         /\ recorded' = recorded \cup {active}
         /\ crashed' = crashed \cup {active}
         /\ active' = None
         /\ phase' = "idle"
         /\ candidateScore' = NoScore
         /\ UNCHANGED <<bestScore, bestId>>

Resume == /\ phase = "idle"
          /\ modelId' = bestId
          /\ UNCHANGED <<phase, active, recorded, crashed, bestScore, bestId,
                         candidateScore>>

Next == (\E e \in Experiments : Begin(e)) \/
        (\E s \in Scores : Evaluate(s)) \/ Keep \/ Discard \/ Crash \/ Resume
Spec == Init /\ [][Next]_vars

TypeOK == /\ phase \in {"idle", "training", "evaluated"}
          /\ active \in Experiments \cup {None}
          /\ recorded \subseteq Experiments
          /\ crashed \subseteq Experiments
          /\ bestScore \in Scores \cup {NoScore}
          /\ bestId \in Experiments \cup {None}
          /\ modelId \in Experiments \cup {None}
          /\ candidateScore \in Scores \cup {NoScore}

BestSafety == /\ (bestId = None) = (bestScore = NoScore)
              /\ (bestId # None => bestId \in recorded)
              /\ crashed \subseteq recorded
              /\ bestId \notin crashed
              /\ (phase = "idle" => active = None)
              /\ (phase = "idle" => modelId = bestId)
              /\ (active # None => active \notin recorded)
=============================================================================
