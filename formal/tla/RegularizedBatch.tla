-------------------------- MODULE RegularizedBatch --------------------------
EXTENDS Integers, FiniteSets
(* Three response rows in two prompt groups. All possible local and global
   denominators divide 360. Signed terms can be losses or parameter slopes.
   The independent reference below enumerates individual response positions. *)
Rows == 1..3
Groups == {1,2}
Unit == 360
VARIABLES length, sign, mode, limit, seen, partial, history, commits, schedules, done
vars == <<length,sign,mode,limit,seen,partial,history,commits,schedules,done>>
Group(i) == IF i = 1 THEN 1 ELSE 2
Count(f) == f[1]+f[2]+f[3]
Sum(S,f) == Count([i \in Rows |-> IF i \in S THEN f[i] ELSE 0])
Members(g) == {i \in Rows : Group(i) = g}
Active(i,t) == t <= length[i]
Policy(i,t,k) == sign[i]*(k+i-t)
Entropy(i,t,k) == sign[4-i]*(i-2*t+k)
Kl(i,k) == IF length[i]=0 THEN 0 ELSE (2-i)*sign[i]*(1+k)
Row(i,k,term(_,_,_)) == (IF Active(i,1) THEN term(i,1,k) ELSE 0) +
                 (IF Active(i,2) THEN term(i,2,k) ELSE 0)
Max1(x) == IF x > 0 THEN x ELSE 1
Mass(S) == CASE mode = "tokens" -> Sum(S,length)
             [] mode = "fixed" -> 3*Cardinality(S)
             [] OTHER -> Cardinality(S)
LocalPolicy(S,k) ==
  Sum(S,[i \in Rows |-> (Unit*Row(i,k,Policy)) \div
                (IF mode="rows" THEN Max1(length[i]) ELSE 1)]) \div Max1(Mass(S))
Local(S,k) ==
  LET tok == Sum(S,length)
      h == (Unit*Sum(S,[i \in Rows |-> Row(i,k,Entropy)])) \div Max1(tok)
      q == (Unit*Sum(S,[i \in Rows |-> Kl(i,k)])) \div Cardinality(S)
  IN (LocalPolicy(S,k)*Mass(S)) \div Max1(Mass(Rows))
     + (q*Cardinality(S)) \div 3 - (h*tok) \div Max1(Count(length))

Init == /\ length \in [Rows -> 0..2] /\ sign \in [Rows -> {-1,1}]
        /\ mode \in {"tokens","rows","fixed"} /\ limit \in 1..3
        /\ seen={} /\ partial=0 /\ history=0 /\ commits=0 /\ schedules=0 /\ done=FALSE
Visit(g) == /\ ~done /\ Count(length)>0 /\ g \in Groups \ seen
            /\ seen'=seen \cup {g} /\ partial'=partial+Local(Members(g),commits)
            /\ UNCHANGED <<length,sign,mode,limit,history,commits,schedules,done>>
Commit == /\ ~done /\ seen=Groups /\ commits<limit
          /\ commits'=commits+1 /\ schedules'=schedules+1
          /\ history'=history+partial /\ seen'={} /\ partial'=0 /\ done'=(commits'=limit)
          /\ UNCHANGED <<length,sign,mode,limit>>
Empty == /\ ~done /\ Count(length)=0 /\ done'=TRUE
         /\ UNCHANGED <<length,sign,mode,limit,seen,partial,history,commits,schedules>>
Next == (\E g \in Groups : Visit(g)) \/ Commit \/ Empty
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

PerToken(i,t,k) ==
  IF ~Active(i,t) THEN 0 ELSE
    LET d == CASE mode="tokens" -> Count(length) [] mode="fixed" -> 9 [] OTHER -> 3*length[i]
    IN (Unit*Policy(i,t,k)) \div d - (Unit*Entropy(i,t,k)) \div Count(length)
Reference(G,k) ==
  Count([i \in Rows |-> IF Group(i) \in G THEN
        PerToken(i,1,k)+PerToken(i,2,k)+(Unit*Kl(i,k)) \div 3 ELSE 0])
Correct ==
  /\ partial=Reference(seen,commits) /\ schedules=commits
  /\ history=(IF commits>=1 THEN Reference(Groups,0) ELSE 0)
            +(IF commits>=2 THEN Reference(Groups,1) ELSE 0)
            +(IF commits>=3 THEN Reference(Groups,2) ELSE 0)
  /\ done => commits=(IF Count(length)=0 THEN 0 ELSE limit)
Completes == <>done
=============================================================================
