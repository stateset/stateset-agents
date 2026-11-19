---------------------------- MODULE PreferenceBatch ----------------------------
EXTENDS Integers, Sequences
Paths == {<<1>>, <<2>>, <<1,1>>, <<1,2>>, <<2,1>>, <<2,2>>}
Rows == 1..2
Zero == <<0,1>>
Plus(a,b) == <<a[1]*b[2] + b[1]*a[2], a[2]*b[2]>>
Equal(a,b) == a[2] > 0 /\ b[2] > 0 /\ a[1]*b[2] = b[1]*a[2]

VARIABLES chosen, rejected, reverse, limit, version, cursor, grad, last, failed
vars == <<chosen, rejected, reverse, limit, version, cursor, grad, last, failed>>
Path(i, win) == IF (i=2 /\ reverse) THEN (IF win THEN rejected ELSE chosen)
                ELSE (IF win THEN chosen ELSE rejected)
Keep(path) == IF limit=1 THEN <<path[1]>> ELSE path
(* Conditional action-one mass in thirds. The second score depends on its prefix. *)
One(t, path, head) ==
    LET base == IF t=1 THEN 1 ELSE (IF path[1]=1 THEN 2 ELSE 1)
    IN IF head=0 THEN base ELSE 3-base
Atom(t, path, head) == IF path[t]=1 THEN One(t,path,head) ELSE 3-One(t,path,head)
Slope(t, path, head) == (IF path[t]=1 THEN 3 ELSE 0) - One(t,path,head)
Mass(path,h) == Atom(1,path,h) * (IF Len(path)=2 THEN Atom(2,path,h) ELSE 3)
Score(path,h) == Slope(1,path,h) + (IF Len(path)=2 THEN Slope(2,path,h) ELSE 0)

(* For beta=1 the weight on a fixed preference draw is 1/(1+adjusted odds).
   Score is three times the log-likelihood slope of a shared action-one logit. *)
RowGrad(i,h) ==
    LET c == Keep(Path(i,TRUE)) r == Keep(Path(i,FALSE))
        win == Mass(c,h)*Mass(r,0)
        lose == Mass(r,h)*Mass(c,0)
    IN <<lose*(Score(r,h)-Score(c,h)), 3*(win+lose)>>

(* Independent exact full-response (probability in ninths, slope in thirds) table. *)
Truth(p,h) ==
    CASE p = <<1>> -> (IF h=0 THEN <<3,2>> ELSE <<6,1>>)
      [] p = <<2>> -> (IF h=0 THEN <<6,-1>> ELSE <<3,-2>>)
      [] p = <<1,1>> -> <<2,3>>
      [] p = <<1,2>> -> (IF h=0 THEN <<1,0>> ELSE <<4,0>>)
      [] p = <<2,1>> -> (IF h=0 THEN <<2,1>> ELSE <<2,-1>>)
      [] OTHER -> (IF h=0 THEN <<4,-2>> ELSE <<1,-4>>)
Oracle(i,h) ==
    LET c == Keep(Path(i,TRUE)) r == Keep(Path(i,FALSE))
        a == Truth(c,h) b == Truth(r,h)
        num == b[1]*Truth(c,0)[1]
        den == num + a[1]*Truth(r,0)[1]
    IN <<num*(b[2]-a[2]),3*den>>
Mean(h) == LET a == Oracle(1,h) b == Oracle(2,h)
           IN <<a[1]*b[2]+b[1]*a[2], 2*a[2]*b[2]>>

Init ==
    /\ chosen \in Paths /\ rejected \in Paths /\ reverse \in BOOLEAN /\ limit \in 1..2
    /\ version=0 /\ cursor=1 /\ grad=Zero /\ last=Zero /\ failed=FALSE
Accumulate ==
    /\ version<2 /\ cursor \in Rows
    /\ grad'=Plus(grad,RowGrad(cursor,version)) /\ cursor'=cursor+1
    /\ UNCHANGED <<chosen,rejected,reverse,limit,version,last,failed>>
Commit ==
    /\ version<2 /\ cursor=3
    /\ last' = <<grad[1],2*grad[2]>> /\ version'=version+1 /\ cursor'=1 /\ grad'=Zero
    /\ UNCHANGED <<chosen,rejected,reverse,limit,failed>>
Abort ==
    /\ version<2 /\ ~failed /\ cursor \in Rows
    /\ failed'=TRUE /\ cursor'=1 /\ grad'=Zero
    /\ UNCHANGED <<chosen,rejected,reverse,limit,version,last>>
Next == Accumulate \/ Commit \/ Abort
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Gradient == Equal(grad, CASE cursor=1 -> Zero [] cursor=2 -> Oracle(1,version) []
                             OTHER -> Plus(Oracle(1,version),Oracle(2,version)))
Committed == Equal(last, IF version=0 THEN Zero ELSE Mean(version-1))
Finished == <>(version=2)
=============================================================================
