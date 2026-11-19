------------------------- MODULE OptimizerWindow ----------------------------
EXTENDS Integers
(* Two logical windows, one parameter vector along (3,4). LR for commit j is
   1/j. Rational arithmetic uses <numerator, positive denominator>. The
   clipping epsilon is 1/2 so all TLC integer cross-products stay small. *)
Windows == 1..2
Samples == {<<"empty",0,1,0>>, <<"overflow",0,1,0>>} \cup
           {<<"finite",m,c,k>> : m \in -2..2, c \in 1..2, k \in {0,6}}
Abs(x) == IF x < 0 THEN -x ELSE x
Min(x,y) == IF x < y THEN x ELSE y
Zero == <<0,1>>
Add(a,b) == <<a[1]*b[2]+b[1]*a[2],a[2]*b[2]>>
Eq(a,b) == a[1]*b[2] = b[1]*a[2]
VARIABLES rows, i, phase, raw, delta, weight, commits, scheduler, published
vars == <<rows,i,phase,raw,delta,weight,commits,scheduler,published>>
Init == /\ rows \in [Windows -> Samples] /\ i = 1 /\ phase = "ready"
        /\ raw = 0 /\ delta = Zero /\ weight = Zero
        /\ commits = 0 /\ scheduler = 0 /\ published = 0
Prepare ==
    /\ i \in Windows /\ phase = "ready"
    /\ IF rows[i][1] = "finite"
       THEN /\ phase' = "corrected" /\ raw' = rows[i][2]*rows[i][3] /\ i' = i
       ELSE /\ phase' = "ready" /\ i' = i+1 /\ UNCHANGED raw
    /\ UNCHANGED <<rows,delta,weight,commits,scheduler,published>>
Clip ==
    /\ phase = "corrected"
    /\ LET d == 10*Abs(raw)+1 IN delta' = <<raw*Min(d,2*rows[i][4]),d>>
    /\ phase' = "clipped" /\ UNCHANGED <<rows,i,raw,weight,commits,scheduler,published>>
Apply ==
    /\ phase = "clipped" /\ phase' = "schedule" /\ commits' = commits+1
    /\ weight' = Add(weight,<<-delta[1],delta[2]*(scheduler+1)>>)
    /\ UNCHANGED <<rows,i,raw,delta,scheduler,published>>
Schedule ==
    /\ phase = "schedule" /\ phase' = "publish" /\ scheduler' = scheduler+1
    /\ UNCHANGED <<rows,i,raw,delta,weight,commits,published>>
Publish ==
    /\ phase = "publish" /\ phase' = "ready" /\ published' = published+1 /\ i' = i+1
    /\ UNCHANGED <<rows,raw,delta,weight,commits,scheduler>>
Next == Prepare \/ Clip \/ Apply \/ Schedule \/ Publish
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Valid(r) == IF rows[r][1] = "finite" THEN 1 ELSE 0
Applied == IF phase \in {"schedule","publish"} THEN i ELSE i-1
ExpectedCount == (IF Applied >= 1 THEN Valid(1) ELSE 0) +
                 (IF Applied >= 2 THEN Valid(2) ELSE 0)
Reference(r,rateDen) ==
    LET row == rows[r]
        a == row[2]*row[3]
        d == 10*Abs(a)+1
    IN IF row[1] # "finite" THEN Zero
       ELSE IF 2*row[4] >= d THEN <<-a,rateDen>>
       ELSE <<-2*row[4]*a,rateDen*d>>
UpdateCorrect ==
    /\ commits = ExpectedCount
    /\ scheduler = commits - (IF phase = "schedule" THEN 1 ELSE 0)
    /\ published = commits - (IF phase \in {"schedule","publish"} THEN 1 ELSE 0)
    /\ Eq(weight, Add(IF Applied >= 1 THEN Reference(1,1) ELSE Zero,
                      IF Applied >= 2 THEN Reference(2,1+Valid(1)) ELSE Zero))
BoundedClip == phase = "clipped" =>
    /\ 5*Abs(delta[1]) <= rows[i][4]*delta[2]
    /\ 5*Abs(delta[1]) <= 5*Abs(raw)*delta[2] /\ raw*delta[1] >= 0
Completes == <>(i = 3)
=============================================================================
