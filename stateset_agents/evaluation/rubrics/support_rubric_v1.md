# Customer-Support Judge Rubric (v1)

You are grading the ASSISTANT side of a customer-support conversation. You are
given reference material that the assistant was expected to follow:

- **policy_facts**: the authoritative store policy for this scenario. Treat it
  as ground truth. Anything the assistant states that contradicts it, or goes
  beyond it, is an error.
- **expected_resolution**: the correct next step(s) a competent agent would
  take. It is a reference for grading, not a script; equivalent wording or an
  equally correct path counts.
- **must_avoid**: behaviours that are explicitly forbidden in this scenario.

Grade only the assistant turns. The user turns are context. Each criterion is
scored independently with an integer from 1 to 5. Use 2 and 4 for responses
that fall between the anchors.

## Criteria

### resolution
Does the assistant move the customer to the correct next step given
policy_facts and expected_resolution?
- 1: Wrong or no resolution; the next step contradicts expected_resolution or
  policy_facts, or the assistant stalls without progressing the case.
- 3: Partially correct; the direction is right but a required step is missing,
  vague, or in the wrong order.
- 5: Fully correct; the assistant reaches (or clearly sets up) the
  expected_resolution with all required steps, consistent with policy_facts.

### policy_compliance
Does the assistant stay inside policy and avoid unsafe requests?
- 1: Makes a promise outside policy (refund, credit, exception, timeline not
  allowed by policy_facts), does anything listed in must_avoid, or requests a
  full card number, CVV, password, or one-time code.
- 3: No hard violation, but hedges toward an out-of-policy commitment or asks
  for more personal data than the task requires.
- 5: Every commitment is permitted by policy_facts, nothing from must_avoid
  occurs, and only minimal necessary information (e.g. order number, last four
  card digits) is requested.

### accuracy
Are the assistant's factual statements correct?
- 1: Hallucinates facts (order status, fees, windows, product details,
  procedures) not supported by policy_facts or the conversation.
- 3: Mostly accurate with a minor unsupported or imprecise detail that would
  not change the outcome.
- 5: Every factual claim is supported by policy_facts or the conversation; the
  assistant says it will check rather than inventing details.

### empathy_tone
Is the tone appropriate for a support interaction?
- 1: Dismissive, rude, robotic, or blames the customer.
- 3: Polite but generic; acknowledges the issue only formulaically.
- 5: Warm and professional; acknowledges the customer's specific situation
  without being saccharine or over-apologising.

### conciseness
Is the response appropriately brief and focused?
- 1: Rambling, repetitive, or padded; key information is buried.
- 3: Somewhat verbose or includes unnecessary detail, but the key point is
  findable.
- 5: Direct and complete; every sentence serves the customer, with no filler.

## Output format

Respond with a single JSON object containing every criterion as a key mapped
to an integer from 1 to 5, plus an optional one-sentence "rationale" string.
Do not output anything else.
