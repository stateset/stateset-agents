-------------------------- MODULE ReplayMixture --------------------------
EXTENDS Integers, FiniteSets, Sequences
(* Groups 1 and 2 have priority 1 and ascent +1/2. Group 3 has priority 5
   and ascent -1/2. The fresh batch has zero ascent and a half chance of
   adding one replay group. Tickets is divisible by every possible pool sum. *)
Tickets == 210
Weight(i) == IF i=3 THEN 5 ELSE 1
VARIABLES k, fresh, heavySize, seen, paths, visited, heavy, mixed, draws
vars == <<k,fresh,heavySize,seen,paths,visited,heavy,mixed,draws>>
Init == /\ k \in 1..3 /\ fresh \in 1..2 /\ heavySize \in 1..2 /\ seen=0
        /\ paths = <<<<>>>> /\ visited={} /\ heavy=0 /\ mixed=0 /\ draws=0
Insert ==
  /\ seen<3
  /\ LET n == seen+1
     IN paths' = IF seen<k THEN [p \in DOMAIN paths |-> Append(paths[p],n)]
                 ELSE [i \in 1..(Len(paths)*n) |->
                          LET prev == paths[((i-1) \div n)+1]
                              ticket == ((i-1) % n)+1
                          IN IF ticket<=k THEN [prev EXCEPT ![ticket]=n] ELSE prev]
  /\ seen'=seen+1 /\ UNCHANGED <<k,fresh,heavySize,visited,heavy,mixed,draws>>
PoolSum(pool) == Len(pool) + (IF 3 \in {pool[i] : i \in DOMAIN pool} THEN 4 ELSE 0)
Pick(pool,t) == LET draw == t % PoolSum(pool)
                IN IF draw < Weight(pool[1]) THEN pool[1]
                   ELSE IF draw < Weight(pool[1])+Weight(pool[2]) THEN pool[2]
                   ELSE pool[3]
Visit(p) ==
  /\ seen=3 /\ p \in DOMAIN paths \ visited
  /\ LET pool == paths[p]
         hit == Cardinality({t \in 0..(Tickets-1) : Pick(pool,t)=3})
         (* Mixed ascent in units of 1/24. The skip branch contributes zero. *)
         up == 12 \div (fresh+1)
         down == (12*heavySize) \div (fresh+heavySize)
     IN /\ heavy'=heavy+hit /\ mixed'=mixed+(Tickets-hit)*up-hit*down
  /\ visited'=visited \cup {p} /\ draws'=draws+2*Tickets
  /\ UNCHANGED <<k,fresh,heavySize,seen,paths>>
Next == Insert \/ (\E p \in DOMAIN paths : Visit(p))
Done == seen=3 /\ visited=DOMAIN paths
TicketLaw == seen=3 => \A p \in DOMAIN paths : \A slot \in DOMAIN paths[p] :
  LET pool == paths[p]
      selected == {t \in 0..(Tickets-1) : Pick(pool,t)=pool[slot]}
  IN Cardinality(selected)*PoolSum(pool)=Tickets*Weight(pool[slot])
Law == Done =>
  LET den == 3*(k+4)
      num == 5*k
      count == Len(paths)*Tickets
      up == 12 \div (fresh+1)
      down == (12*heavySize) \div (fresh+heavySize)
  IN /\ heavy*den=count*num
     /\ draws=2*count
     /\ mixed*den=count*((den-num)*up-num*down)
     /\ (k=2 /\ fresh=1 /\ heavySize=1 => 72*mixed = -24*draws)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Finished == <>Done
=============================================================================
