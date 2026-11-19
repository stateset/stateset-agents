--------------------------- MODULE PolicyReduction ---------------------------
EXTENDS Integers, FiniteSets
(* A two-response prompt group, two token slots per row. Units for ratios
   are halves. All numerators carry a common factor of two so division by
   either nonempty row length is exact. *)
VARIABLES reward, length, ratios, mode, seen, numerator, denominator
vars == <<reward, length, ratios, mode, seen, numerator, denominator>>
Rows == 1..2
Slots == 1..2
Min(a,b) == IF a < b THEN a ELSE b
Max(a,b) == IF a > b THEN a ELSE b
Adv(i) == 2*reward[i] - reward[1] - reward[2]
Clip(r) == Max(1, Min(3,r))
Loss(i,t) == -Min(ratios[i][t]*Adv(i), Clip(ratios[i][t])*Adv(i))
Weight(i) == CASE mode = "token" -> length[i]
               [] mode = "sequence" -> 1
               [] mode = "fixed" -> 2
SumRow(i) == (IF length[i] >= 1 THEN Loss(i,1) ELSE 0) +
             (IF length[i] = 2 THEN Loss(i,2) ELSE 0)
Contribution(i) == IF mode = "sequence"
                   THEN (2 * SumRow(i)) \div Max(1,length[i])
                   ELSE 2 * SumRow(i)

Init == /\ reward \in [Rows -> {-1,1}]
        /\ length \in [Rows -> 0..2]
        /\ ratios \in [Rows -> [Slots -> {0,2,4}]]
        /\ mode \in {"sequence","token","fixed"}
        /\ seen = {} /\ numerator = 0 /\ denominator = 0

Add(i) == /\ i \notin seen
          /\ numerator' = numerator + Contribution(i)
          /\ denominator' = denominator + Weight(i)
          /\ seen' = seen \cup {i}
          /\ UNCHANGED <<reward,length,ratios,mode>>
Next == (\E i \in Rows : Add(i)) \/ (seen = Rows /\ UNCHANGED vars)

(* Independent closed form over visited tokens. Only a beneficial ratio move
   can saturate the clipped surrogate. Empty rows have a zero contribution. *)
Term(i,t) == IF i \notin seen \/ t > length[i] THEN 0 ELSE
             LET r == ratios[i][t]
                 a == reward[i] - reward[3-i]
                 effective == IF a > 0 THEN Min(r,3) ELSE Max(r,1)
                 scale == IF mode = "sequence" /\ length[i] = 2 THEN 1 ELSE 2
             IN -effective*a*scale
ReferenceSum == Term(1,1) + Term(1,2) + Term(2,1) + Term(2,2)
ReferenceCount == CASE mode = "sequence" -> Cardinality(seen)
                    [] mode = "fixed" -> 2*Cardinality(seen)
                    [] OTHER -> Cardinality({p \in Rows \X Slots :
                                             p[1] \in seen /\ p[2] <= length[p[1]]})
MatchesBatch == numerator = ReferenceSum /\ denominator = ReferenceCount
=============================================================================
