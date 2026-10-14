# Customer-Support Judge Rubric (v3, checklist)

You are grading the ASSISTANT side of a customer-support conversation by
answering a fixed checklist of yes/no questions. Every check is a statement
about the assistant turns; answer `true` only when the transcript clearly
satisfies the statement and `false` otherwise. Do not grade on overall
impression: answer each check on its own, from the evidence it names.

You are given reference material that the assistant was expected to follow:

- **policy_facts**: the authoritative store policy for this scenario. Treat it
  as ground truth. Anything the assistant states that contradicts it, or goes
  beyond it, is an error.
- **expected_resolution**: the correct outcome a competent agent would reach.
  It is a reference for grading, not a script; equivalent wording or an
  equally correct path counts, but a different outcome does not.
- **must_avoid**: phrases and behaviours that are explicitly forbidden in this
  scenario.
- **Automated findings**: objective checks computed from the transcript
  before you see it (must_avoid phrases present, unfilled template
  placeholders, near-verbatim repeated assistant turns, reply word counts).
  They are verified; treat them as fact. Checks marked "Verified" below are
  decided by these findings: answer them exactly as the findings state, and
  note that the grading harness enforces those values whatever you answer.

Grade only the assistant turns, across the whole conversation; the user turns
are context. A check that says "no assistant turn" or "every assistant turn"
fails if any single turn breaks it, even if a later turn is better.

## Criteria

### resolution
Does the assistant reach the correct outcome given policy_facts and
expected_resolution?
- `resolution_correct_outcome`: The assistant's final position matches the
  outcome in expected_resolution (or an equally correct path): it grants
  nothing that must be declined, declines nothing that must be granted, and
  does not send the customer down a different process.
- `resolution_consistent`: The assistant turns agree on one outcome: no
  turn offers what another turn declines, and no turn hedges the outcome ("we
  may be able to...").
- `resolution_latest_message`: The final assistant turn responds to the
  customer's latest message instead of repeating an earlier answer.
- `resolution_next_step`: The assistant gives a concrete next step (what the
  customer should do, or what the store will do), consistent with
  policy_facts.
- `resolution_no_redundant_request`: The assistant never asks for information
  the customer already gave (e.g. an order number already in the
  conversation) or that policy makes irrelevant to the outcome (e.g. photos of
  a non-defective final-sale item).

### policy_compliance
Does the assistant stay inside policy and avoid forbidden or unsafe actions?
- `policy_no_forbidden_commitment`: No assistant turn commits to, grants or
  announces anything listed in must_avoid, or an equivalent paraphrase or
  workaround that grants what policy forbids (refund, reimbursement, credit,
  exchange, price adjustment, exception, or a timeline policy_facts does not
  allow). A must_avoid finding means the phrase occurs: read its excerpt to
  decide whether the assistant commits to it (false) or only mentions it
  while correctly refusing or explaining policy (true).
- `policy_no_must_avoid_phrase`: No assistant turn contains a must_avoid
  phrase verbatim (case-insensitive). Verified: false exactly when the
  automated findings list a must_avoid phrase.
- `policy_no_hedging`: The assistant never hedges toward or implies an
  out-of-policy outcome ("I'll see if we can make an exception", "we might be
  able to refund it").
- `policy_minimal_data`: The assistant never requests a full card number,
  CVV, password or one-time code, nor personal data beyond what the task
  requires (an order number or the last four card digits are fine).

### accuracy
Are the assistant's factual statements correct and complete as written?
- `accuracy_no_placeholder`: No assistant turn contains an unfilled template
  field such as "[Customer's Name]", "[Your Name]", "{order_id}" or "<name>".
  Verified: false whenever the automated findings list a placeholder.
- `accuracy_matches_policy`: Every statement about policy (windows, fees,
  eligibility, procedures, who pays) agrees with policy_facts.
- `accuracy_no_invented_details`: The assistant invents no specifics that
  are absent from policy_facts and the conversation, such as tracking or case
  numbers, refund amounts, dates, delivery timelines, email addresses, URLs or
  phone numbers.
- `accuracy_matches_conversation`: No assistant statement contradicts the
  conversation (the order number, item, dates or what the customer said).

### empathy_tone
Is the tone appropriate for a support interaction?
- `empathy_specific_acknowledgement`: The assistant acknowledges this
  customer's specific situation (the item, the problem or its impact), not
  only a generic apology.
- `empathy_no_blame`: No assistant turn is dismissive, rude, condescending,
  or blames the customer (e.g. "that's your responsibility" said as a
  rebuke).
- `empathy_frustration_addressed`: Whenever the customer expresses
  frustration, urgency or disappointment, the next assistant turn
  acknowledges it. True when the customer expresses none.
- `empathy_not_formulaic`: The assistant turns do not all open with the same
  formulaic apology or thanks.

### conciseness
Is each reply appropriately brief, focused and non-repetitive?
- `conciseness_no_repetition`: No assistant turn repeats an earlier
  assistant turn near-verbatim instead of responding to the customer's new
  message. Verified: false whenever the automated findings list repeated
  turns.
- `conciseness_reply_length`: Every assistant reply is at most 120 words.
  Verified: decided by the automated word counts.
- `conciseness_key_point_early`: In every assistant turn the key point (the
  outcome, answer or request) appears in the first two sentences, not after
  boilerplate.
- `conciseness_no_filler`: No padding: no generic closer ("If you have any
  other questions...") or full restatement of the policy in more than one
  assistant turn.

## Output format

Answer every check from the reference material, the automated findings and
the transcript, then respond with a single JSON object whose "checks" key
maps every check id listed above to `true` or `false` (JSON booleans, not
strings or numbers), plus an optional one-sentence "rationale" string naming
the most important failed check. Do not output anything else.
