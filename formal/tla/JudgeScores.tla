--------------------------- MODULE JudgeScores ------------------------------
EXTENDS Integers
Rows == 1..2
Queries == 1..2
Rubrics == {<<-1,1>>,<<0,1>>,<<0,2>>}
Keys == Queries \X Rubrics
Invalid == 99
Missing == -1
VARIABLES requests, raw, seen, cache, scores
vars == <<requests,raw,seen,cache,scores>>
Init == /\ requests \in [Rows -> Keys] /\ raw \in [Queries -> {-1,0,1,2,Invalid}]
        /\ seen = {} /\ cache = [k \in Keys |-> Missing] /\ scores = [i \in Rows |-> Missing]
(* Quarter units suffice for every represented interval. An invalid single
   evaluation fails before caching; the batch layer supplies its neutral 1/2. *)
Bound(x,l,h) == IF x < l THEN l ELSE IF x > h THEN h ELSE x
Evaluate(i) ==
    /\ i \notin seen /\ seen' = seen \cup {i}
    /\ LET k == requests[i]
           r == raw[k[1]]
           l == k[2][1]
           h == k[2][2]
           value == IF cache[k] # Missing THEN cache[k]
                    ELSE IF r = Invalid THEN 2 ELSE (4*(Bound(r,l,h)-l)) \div (h-l)
       IN /\ scores' = [scores EXCEPT ![i] = value]
          /\ cache' = IF r = Invalid THEN cache ELSE [cache EXCEPT ![k] = value]
    /\ UNCHANGED <<requests,raw>>
Next == \E i \in Rows : Evaluate(i)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Correct == \A i \in seen :
    LET k == requests[i]
        r == raw[k[1]]
        l == k[2][1]
        h == k[2][2]
        v == scores[i]
    IN IF r = Invalid THEN v = 2 /\ cache[k] = Missing
       ELSE /\ v \in 0..4 /\ cache[k] = v
            /\ (r <= l => v = 0) /\ (r >= h => v = 4)
            /\ (l < r /\ r < h => v*(h-l) = 4*(r-l))
Completes == <>(seen = Rows)
=============================================================================
