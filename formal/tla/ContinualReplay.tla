-------------------------- MODULE ContinualReplay --------------------------
EXTENDS Integers, Sequences, FiniteSets
CONSTANT StreamLength, MaxCapacity
VARIABLES capacity, seen, paths
vars == <<capacity, seen, paths>>

(* Each path holds a native slot vector. At a full reservoir there is one
   branch per possible randint result, including every skip multiplicity. *)
Init == /\ capacity \in 1..MaxCapacity /\ seen = 0 /\ paths = <<<<>>>>
Advance ==
  /\ seen < StreamLength
  /\ LET next == seen + 1
         slots == Len(paths)
     IN paths' =
        IF seen < capacity THEN [p \in 1..slots |-> Append(paths[p], next)]
        ELSE [i \in 1..(slots*next) |->
               LET old == paths[((i-1) \div next)+1]
                   ticket == ((i-1) % next)+1
               IN IF ticket <= capacity THEN [old EXCEPT ![ticket] = next] ELSE old]
  /\ seen' = seen+1 /\ UNCHANGED capacity

Contents(pool) == {pool[j] : j \in DOMAIN pool}
Size == IF seen < capacity THEN seen ELSE capacity
RECURSIVE Factorial(_)
Factorial(x) == IF x <= 0 THEN 1 ELSE x*Factorial(x-1)
Uniform ==
  /\ \A p \in DOMAIN paths :
       /\ Len(paths[p]) = Size /\ Cardinality(Contents(paths[p])) = Size
       /\ Contents(paths[p]) \subseteq 1..seen
  /\ \A target \in {s \in SUBSET (1..seen) : Cardinality(s) = Size} :
       Cardinality({p \in DOMAIN paths : Contents(paths[p]) = target}) = Factorial(seen-Size)
Spec == Init /\ [][Advance]_vars /\ WF_vars(Advance)
Finished == <>(seen=StreamLength)
=============================================================================
