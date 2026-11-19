------------------------- MODULE DecisionConditioning ------------------------
EXTENDS Integers, Sequences, FiniteSets

(* Four raw conversation turns. A bit identifies each assistant decision.
   The discount numerator is in halves. A coarse grid forces outward
   rounding; a finer grid is exact everywhere on this domain. *)
Turns == 1..4
VARIABLES reward, assistant, discount, grid, cap, scan, lo, hi, lower, upper, turn, ctx
vars == <<reward,assistant,discount,grid,cap,scan,lo,hi,lower,upper,turn,ctx>>
Zeros == [i \in Turns |-> 0]
Init == /\ reward \in [Turns -> -1..1] /\ assistant \in SUBSET Turns
        /\ discount \in {-1,0,1,2} /\ grid \in {1,16} /\ cap \in 1..2
        /\ scan = 4 /\ lo = 0 /\ hi = 0 /\ lower = Zeros /\ upper = Zeros
        /\ turn = 1 /\ ctx = <<>>
Back ==
  /\ scan > 0
  /\ LET small == IF discount < 0 THEN discount*hi ELSE discount*lo
         large == IF discount < 0 THEN discount*lo ELSE discount*hi
     IN /\ lo' = grid*reward[scan]+(small \div 2)
        /\ hi' = grid*reward[scan]-((-large) \div 2)
  /\ lower' = [lower EXCEPT ![scan] = lo'] /\ upper' = [upper EXCEPT ![scan] = hi']
  /\ scan' = scan-1
  /\ UNCHANGED <<reward,assistant,discount,grid,cap,turn,ctx>>
Emit ==
  /\ scan = 0 /\ turn \in Turns /\ turn' = turn+1
  /\ LET next == IF turn \in assistant THEN Append(ctx, <<turn,lower[turn],upper[turn]>>) ELSE ctx
     IN ctx' = IF Len(next) > cap THEN Tail(next) ELSE next
  /\ UNCHANGED <<reward,assistant,discount,grid,cap,scan,lo,hi,lower,upper>>
Next == Back \/ Emit
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* The oracle sums all subsequent rewards directly at the original turn
   distances. It selects the last cap assistant positions by their rank. *)
Term(i,j) == IF j < i THEN 0 ELSE IF i = j THEN 16*reward[i] ELSE
             (2^(4-(j-i))) * discount^(j-i) * reward[j]
Exact(i) == Term(i,1)+Term(i,2)+Term(i,3)+Term(i,4)
Bounds == \A i \in Turns : i > scan =>
    /\ lower[i]*16 <= grid*Exact(i) /\ grid*Exact(i) <= upper[i]*16
    /\ (grid = 16 => lower[i] = upper[i])
Visited == {i \in assistant : i < turn}
Recent == {i \in Visited : Cardinality({j \in Visited : i < j}) < cap}
Context ==
  /\ {ctx[j][1] : j \in 1..Len(ctx)} = Recent
  /\ Len(ctx) = Cardinality(Recent)
  /\ \A j \in 1..Len(ctx) :
      /\ (j > 1 => ctx[j-1][1] < ctx[j][1])
      /\ ctx[j][2]*16 <= grid*Exact(ctx[j][1])
      /\ grid*Exact(ctx[j][1]) <= ctx[j][3]*16
Complete == <>(turn = 5)
=============================================================================
