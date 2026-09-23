--------------------------- MODULE RolloutControl ---------------------------
EXTENDS Naturals, FiniteSets, Sequences

CONSTANT Workers, Ids, BadIds, QueueCapacity, MaxLag, ArtifactCapacity,
         MaxVersion
ASSUME /\ Workers # {} /\ Ids # {}
       /\ IsFiniteSet(Workers) /\ IsFiniteSet(Ids)
       /\ BadIds \subseteq Ids
       /\ QueueCapacity \in Nat \ {0}
       /\ MaxLag \in Nat
       /\ ArtifactCapacity \in Nat \ {0}
       /\ MaxVersion \in Nat

VARIABLES version, artifacts, live, generation, assignment, pending,
          pendingWorker, pendingGeneration, pendingVersion, queue, seen,
          lateAccepted, badAccepted, accepted
vars == <<version, artifacts, live, generation, assignment, pending,
          pendingWorker, pendingGeneration, pendingVersion, queue, seen,
          lateAccepted, badAccepted, accepted>>

Init == /\ version = 0
        /\ artifacts = {0}
        /\ live = {}
        /\ generation = [w \in Workers |-> 0]
        /\ assignment = [w \in Workers |-> 0]
        /\ pending = {}
        /\ pendingWorker = [id \in Ids |-> CHOOSE w \in Workers : TRUE]
        /\ pendingGeneration = [id \in Ids |-> 0]
        /\ pendingVersion = [id \in Ids |-> 0]
        /\ queue = <<>>
        /\ seen = {}
        /\ lateAccepted = {}
        /\ badAccepted = {}
        /\ accepted = 0

Register(w) == /\ w \in Workers
               /\ generation[w] < 2
               /\ live' = live \cup {w}
               /\ generation' = [generation EXCEPT ![w] = @ + 1]
               /\ assignment' = [assignment EXCEPT ![w] = version]
               /\ UNCHANGED <<version, artifacts, pending, pendingWorker,
                              pendingGeneration, pendingVersion, queue, seen,
                              lateAccepted, badAccepted, accepted>>

Heartbeat(w) == /\ w \in live
                /\ assignment' = [assignment EXCEPT ![w] = version]
                /\ UNCHANGED <<version, artifacts, live, generation, pending,
                               pendingWorker, pendingGeneration, pendingVersion,
                               queue, seen, lateAccepted, badAccepted,
                               accepted>>

Expire(w) == /\ w \in live
             /\ live' = live \ {w}
             /\ UNCHANGED <<version, artifacts, generation, assignment,
                            pending, pendingWorker, pendingGeneration,
                            pendingVersion, queue, seen, lateAccepted,
                            badAccepted, accepted>>

BeginSubmit(w, id) ==
    /\ w \in live
    /\ id \in Ids \ (pending \cup seen)
    /\ assignment[w] \in artifacts
    /\ id \notin BadIds
    /\ pending' = pending \cup {id}
    /\ pendingWorker' = [pendingWorker EXCEPT ![id] = w]
    /\ pendingGeneration' = [pendingGeneration EXCEPT ![id] = generation[w]]
    /\ pendingVersion' = [pendingVersion EXCEPT ![id] = assignment[w]]
    /\ UNCHANGED <<version, artifacts, live, generation, assignment, queue,
                   seen, lateAccepted, badAccepted, accepted>>

FinishSubmit(id) ==
    /\ id \in pending
    /\ id \notin seen
    /\ pendingWorker[id] \in live
    /\ generation[pendingWorker[id]] = pendingGeneration[id]
    /\ assignment[pendingWorker[id]] = pendingVersion[id]
    /\ pendingVersion[id] \in artifacts
    /\ id \notin BadIds
    /\ Len(queue) < QueueCapacity
    /\ pendingVersion[id] <= version
    /\ version - pendingVersion[id] <= MaxLag
    /\ queue' = Append(queue, id)
    /\ seen' = seen \cup {id}
    /\ accepted' = accepted + 1
    /\ pending' = pending \ {id}
    /\ lateAccepted' = IF generation[pendingWorker[id]] # pendingGeneration[id]
                        THEN lateAccepted \cup {id} ELSE lateAccepted
    /\ badAccepted' = IF id \in BadIds THEN badAccepted \cup {id}
                      ELSE badAccepted
    /\ UNCHANGED <<version, artifacts, live, generation, assignment,
                   pendingWorker, pendingGeneration, pendingVersion>>

DropPending(id) == /\ id \in pending
                   /\ pending' = pending \ {id}
                   /\ UNCHANGED <<version, artifacts, live, generation,
                                  assignment, pendingWorker, pendingGeneration,
                                  pendingVersion, queue, seen, lateAccepted,
                                  badAccepted, accepted>>

Consume == /\ Len(queue) > 0
           /\ queue' = Tail(queue)
           /\ UNCHANGED <<version, artifacts, live, generation, assignment,
                          pending, pendingWorker, pendingGeneration,
                          pendingVersion, seen, lateAccepted, badAccepted,
                          accepted>>

Oldest(S) == CHOOSE x \in S : \A y \in S : x <= y

Publish == /\ version < MaxVersion
           /\ version' = version + 1
           /\ artifacts' = IF Cardinality(artifacts) < ArtifactCapacity
                            THEN artifacts \cup {version'}
                            ELSE (artifacts \ {Oldest(artifacts)}) \cup {version'}
           /\ queue' = SelectSeq(queue,
                           LAMBDA id : version' - pendingVersion[id] <= MaxLag)
           /\ UNCHANGED <<live, generation, assignment, pending,
                          pendingWorker, pendingGeneration, pendingVersion,
                          seen, lateAccepted, badAccepted, accepted>>

Next == (\E w \in Workers : Register(w) \/ Heartbeat(w) \/ Expire(w))
        \/ (\E w \in Workers : \E id \in Ids : BeginSubmit(w, id))
        \/ (\E id \in Ids : FinishSubmit(id) \/ DropPending(id))
        \/ Consume \/ Publish
Spec == Init /\ [][Next]_vars

CoreSafety == /\ Len(queue) <= QueueCapacity
              /\ \A i \in 1..Len(queue) : queue[i] \in seen
              /\ Cardinality({queue[i] : i \in 1..Len(queue)}) = Len(queue)
              /\ accepted = Cardinality(seen)
              /\ \A i \in 1..Len(queue) :
                   pendingVersion[queue[i]] <= version /\
                   version - pendingVersion[queue[i]] <= MaxLag
              /\ Cardinality(artifacts) <= ArtifactCapacity
              /\ version \in artifacts

FencingSafety == lateAccepted = {}
ArtifactSafety == badAccepted = {}
=============================================================================
