"""Closed-Universe Client Matter Sandbox for Multi-Turn Agent RL & Evaluation.

Each task places an agent inside a self-contained, closed-universe client matter
containing both **key controlling documents** (governing contracts, amendments,
correspondence, disclosure schedules) and **peripheral distractor documents**
(unrelated office policies, superseded drafts, administrative invoices).

The agent operates in a confined sandboxed workspace with six tools:
- ``list_matter_files``: list filenames and metadata (without leaking full text)
- ``search_matter``: keyword/regex search across matter documents
- ``read_document``: confined line-ranged document read (rejects traversal)
- ``draft_work_product``: write or update the work-product deliverable
- ``read_draft``: inspect the current deliverable draft
- ``signal_complete``: signal episode completion cleanly

Episodes terminate when the agent calls ``signal_complete`` or hits its
execution limit (``max_steps``). Deliverables are graded against an expert
binary pass/fail rubric grouped into legal issue groups via
:class:`~stateset_agents.rewards.rubric_reward.HierarchicalRubricReward`.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from stateset_agents.rewards.rubric_reward import (
    DeterministicRubricJudge,
    HierarchicalRubricReward,
    RubricCriterion,
    RubricJudge,
    RubricRewardConfig,
    score_rubric_verdicts,
)


@dataclass(frozen=True)
class MatterDocument:
    """One document inside a closed-universe client matter workspace."""

    filename: str
    title: str
    category: str
    content: str
    is_key: bool = True

    def __post_init__(self) -> None:
        name = self.filename.strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError(
                f"Invalid closed-universe matter filename {self.filename!r}; "
                "must be a flat basename without path separators."
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "title": self.title,
            "category": self.category,
            "bytes": len(self.content.encode("utf-8")),
            "lines": len(self.content.splitlines()),
        }


@dataclass(frozen=True)
class ClientMatterTask:
    """Complete closed-universe client matter fixture + binary expert rubric."""

    matter_id: str
    title: str
    instruction: str
    documents: tuple[MatterDocument, ...]
    criteria: tuple[RubricCriterion, ...]
    max_steps: int = 12
    domain: str = "client_matter"

    def __post_init__(self) -> None:
        if not self.matter_id.strip():
            raise ValueError("matter_id must be non-empty")
        if not self.documents:
            raise ValueError("ClientMatterTask requires at least one document")
        if not self.criteria:
            raise ValueError("ClientMatterTask requires at least one rubric criterion")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        seen = {d.filename for d in self.documents}
        if len(seen) != len(self.documents):
            raise ValueError("Matter document filenames must be unique")

    @property
    def issue_groups(self) -> tuple[str, ...]:
        groups: list[str] = []
        for c in self.criteria:
            if c.issue_group not in groups:
                groups.append(c.issue_group)
        return tuple(groups)

    @property
    def key_filenames(self) -> tuple[str, ...]:
        return tuple(d.filename for d in self.documents if d.is_key)

    @property
    def peripheral_filenames(self) -> tuple[str, ...]:
        return tuple(d.filename for d in self.documents if not d.is_key)

    def to_summary(self) -> dict[str, Any]:
        return {
            "matter_id": self.matter_id,
            "title": self.title,
            "domain": self.domain,
            "max_steps": self.max_steps,
            "document_count": len(self.documents),
            "key_document_count": len(self.key_filenames),
            "peripheral_document_count": len(self.peripheral_filenames),
            "criteria_count": len(self.criteria),
            "issue_groups": list(self.issue_groups),
        }


def default_client_matter_tasks() -> list[ClientMatterTask]:
    """Return 3 closed-universe multi-document client matters with expert rubrics."""
    return [
        ClientMatterTask(
            matter_id="matter_acme_assignment",
            title="Acme Robotics Commercial Lease Assignment & Change-of-Control Review",
            instruction=(
                "Our client Acme Robotics Inc. is being acquired in a reverse triangular "
                "merger by Vertex Automation Corp. (tangible net worth $62M). Review the "
                "closed-universe client matter files and draft a concise client memorandum "
                "advising on: (1) whether landlord consent is required under the lease as "
                "amended, (2) any notice, fee, or net-worth threshold conditions, and "
                "(3) landlord remedies and cure rights if Acme closes without notice."
            ),
            documents=(
                MatterDocument(
                    filename="master_lease_2022.txt",
                    title="Master Commercial Lease Agreement (2022)",
                    category="agreement",
                    is_key=True,
                    content=(
                        "MASTER COMMERCIAL LEASE AGREEMENT\n"
                        "Landlord: Harborview Properties LLC\n"
                        "Tenant: Acme Robotics Inc.\n"
                        "Premises: 400 Innovation Way, Suite 500\n\n"
                        "Section 12.1 Assignment and Transfer.\n"
                        "Tenant shall not assign this Lease or sublet the Premises without "
                        "Landlord's prior written consent, not to be unreasonably withheld.\n\n"
                        "Section 12.2 Change of Control Deemed Assignment.\n"
                        "Any merger, consolidation, or transfer of more than 50% of Tenant's "
                        "voting equity shall be deemed an Assignment requiring Landlord consent.\n\n"
                        "Section 12.4 Administrative Review Fee.\n"
                        "Tenant shall pay Landlord a $5,000 administrative review fee upon any "
                        "consent request.\n\n"
                        "Section 18.1 Events of Default and Cure.\n"
                        "An unauthorized Assignment is a non-monetary default. Tenant has "
                        "15 business days after written notice from Landlord to cure any "
                        "non-monetary default before Landlord may exercise termination or "
                        "acceleration remedies.\n"
                    ),
                ),
                MatterDocument(
                    filename="first_amendment_2024.txt",
                    title="First Amendment to Commercial Lease (2024)",
                    category="amendment",
                    is_key=True,
                    content=(
                        "FIRST AMENDMENT TO COMMERCIAL LEASE AGREEMENT\n"
                        "Effective Date: June 15, 2024\n\n"
                        "Section 3. Permitted Transfer Carve-Out (Supersedes Section 12.2).\n"
                        "Notwithstanding Section 12.1 or Section 12.2 of the Master Lease, "
                        "a merger or acquisition of Tenant (a 'Permitted Transfer') shall NOT "
                        "require Landlord's prior written consent provided that: "
                        "(a) the surviving or acquiring entity has a tangible net worth of at "
                        "least $50,000,000 at closing, and (b) Tenant delivers written notice "
                        "and certified financial statements to Landlord at least 15 days prior "
                        "to closing.\n\n"
                        "Section 4. Cap on Transfer Review Fee.\n"
                        "For a Permitted Transfer, the Section 12.4 fee is reduced from $5,000 "
                        "to $1,500.\n"
                    ),
                ),
                MatterDocument(
                    filename="landlord_email_oct2026.txt",
                    title="Correspondence from Harborview Counsel (Oct 2026)",
                    category="correspondence",
                    is_key=True,
                    content=(
                        "From: M. Alvarez, Counsel for Harborview Properties LLC\n"
                        "To: General Counsel, Acme Robotics Inc.\n"
                        "Date: October 12, 2026\n"
                        "Subject: Rumored Acquisition / Lease Compliance\n\n"
                        "We heard reports of a pending transaction involving Acme. Please note "
                        "Harborview expects strict compliance with the 15-day pre-closing notice "
                        "and certified financial statement delivery requirement in the First "
                        "Amendment. Harborview reserves all Section 18.1 remedies including "
                        "lease termination if closing occurs without required notice.\n"
                    ),
                ),
                MatterDocument(
                    filename="office_parking_policy.txt",
                    title="Building Garage & EV Charging Policy",
                    category="peripheral",
                    is_key=False,
                    content=(
                        "HARBORVIEW GARAGE POLICY (2025)\n"
                        "Tenant employees receive 25 unreserved garage passes at $185/month.\n"
                        "Level 2 EV chargers have a 4-hour daytime limit. Overnight storage "
                        "of commercial delivery vans is prohibited without building security "
                        "clearance.\n"
                    ),
                ),
                MatterDocument(
                    filename="superseded_2021_loi.txt",
                    title="Non-Binding Letter of Intent (Superseded 2021)",
                    category="peripheral",
                    is_key=False,
                    content=(
                        "NON-BINDING LETTER OF INTENT — NOVEMBER 2021\n"
                        "Proposed base rent: $68/RSF. Proposed transfer fee: $10,000.\n"
                        "NOTE: This non-binding LOI was expressly superseded in full by the "
                        "Master Commercial Lease Agreement executed in 2022.\n"
                    ),
                ),
            ),
            criteria=(
                RubricCriterion(
                    criterion_id="acme_c1",
                    issue_group="consent_requirement",
                    description=(
                        "Identifies that Master Lease Section 12.2 initially deems a merger "
                        "an assignment requiring consent, but First Amendment Section 3 "
                        "creates a Permitted Transfer carve-out eliminating prior consent."
                    ),
                    required_keywords=("permitted transfer", "first amendment"),
                    regex_patterns=(r"12\.2", r"consent"),
                ),
                RubricCriterion(
                    criterion_id="acme_c2",
                    issue_group="consent_requirement",
                    description=(
                        "Applies the $50M tangible net worth threshold to Vertex Automation "
                        "Corp.'s $62M net worth and concludes the financial threshold is met."
                    ),
                    required_keywords=("50", "62", "net worth"),
                ),
                RubricCriterion(
                    criterion_id="acme_c3",
                    issue_group="notice_and_fees",
                    description=(
                        "Flags the mandatory 15-day pre-closing written notice and certified "
                        "financial statements delivery condition."
                    ),
                    required_keywords=("15", "certified financial"),
                    regex_patterns=(r"prior\s+to\s+closing|pre-closing",),
                ),
                RubricCriterion(
                    criterion_id="acme_c4",
                    issue_group="notice_and_fees",
                    description=(
                        "Correctly states the reduced $1,500 Permitted Transfer fee under "
                        "First Amendment Section 4 (not the $5,000 Master Lease fee or "
                        "superseded $10,000 LOI figure)."
                    ),
                    required_keywords=("1,500",),
                    forbidden_keywords=("10,000 applies",),
                ),
                RubricCriterion(
                    criterion_id="acme_c5",
                    issue_group="remedies_and_cure",
                    description=(
                        "Explains Section 18.1 non-monetary default remedies (including "
                        "termination risk) and the 15-business-day cure period after written notice."
                    ),
                    required_keywords=("18.1", "cure"),
                    regex_patterns=(r"15\s+business\s+days",),
                ),
                RubricCriterion(
                    criterion_id="acme_c6",
                    issue_group="remedies_and_cure",
                    description=(
                        "References Harborview counsel's October 2026 email warning of strict "
                        "enforcement and recommends delivering the notice package immediately."
                    ),
                    required_keywords=("harborview", "notice"),
                    regex_patterns=(r"october\s+2026|oct\s+2026|email|counsel",),
                ),
            ),
            max_steps=12,
        ),
        ClientMatterTask(
            matter_id="matter_nova_indemnity",
            title="NovaCloud Enterprise SaaS Liability Cap & Data Breach Review",
            instruction=(
                "Our client NovaCloud suffered a third-party credential stuffing incident "
                "affecting customer Meridian Health (annual fees $400,000). Draft an internal "
                "risk memorandum analyzing: (1) the applicable contractual liability cap "
                "across the MSA and Order Form #4, (2) breach notification deadline under "
                "the DPA, and (3) IP/confidentiality indemnity defense control requirements."
            ),
            documents=(
                MatterDocument(
                    filename="saas_msa_2025.txt",
                    title="Master Subscription Agreement (2025)",
                    category="agreement",
                    is_key=True,
                    content=(
                        "MASTER SUBSCRIPTION AGREEMENT\n"
                        "Provider: NovaCloud Inc. | Customer: Meridian Health Systems\n\n"
                        "Section 9.1 General Liability Cap.\n"
                        "Each party's aggregate liability shall not exceed 12 months of fees "
                        "paid prior to the claim.\n\n"
                        "Section 9.2 Super-Cap for Data Security Breaches.\n"
                        "For breaches of Section 6 (Data Security) or the DPA, Provider's "
                        "aggregate liability is capped at 2x annual fees.\n\n"
                        "Section 10.2 Indemnification Procedure.\n"
                        "Indemnified party must give prompt written notice and grant Provider "
                        "sole control of defense and settlement; no settlement admitting fault "
                        "without Indemnified party's written consent.\n"
                    ),
                ),
                MatterDocument(
                    filename="order_form_04.txt",
                    title="Enterprise Order Form #4 (Superseding Cap Schedule)",
                    category="amendment",
                    is_key=True,
                    content=(
                        "ORDER FORM #4 — MERIDIAN HEALTH SYSTEMS\n"
                        "Annual Subscription Fees: $400,000\n\n"
                        "Special Term 2 (Precedence over MSA Section 9.2).\n"
                        "Notwithstanding Section 9.2 of the MSA, the Data Security Super-Cap "
                        "for Order Form #4 is increased to 3x annual fees ($1,200,000).\n"
                    ),
                ),
                MatterDocument(
                    filename="dpa_security_addendum.txt",
                    title="Data Processing Addendum & Security Exhibit",
                    category="agreement",
                    is_key=True,
                    content=(
                        "DATA PROCESSING ADDENDUM\n"
                        "Section 5.3 Security Incident Notification.\n"
                        "NovaCloud shall notify Meridian Health in writing within 48 hours "
                        "after confirming a Personal Data Breach, detailing affected record "
                        "categories, root cause, and remediation steps.\n"
                    ),
                ),
                MatterDocument(
                    filename="marketing_case_study_guide.txt",
                    title="Customer Logo & Press Release Approval Guide",
                    category="peripheral",
                    is_key=False,
                    content=(
                        "MARKETING BRAND GUIDELINES\n"
                        "Customer logos require CMO sign-off 10 business days before webinar "
                        "publication. Vector SVG assets live in the brand folder.\n"
                    ),
                ),
            ),
            criteria=(
                RubricCriterion(
                    criterion_id="nova_c1",
                    issue_group="super_cap_carveouts",
                    description=(
                        "Identifies MSA Section 9.2's 2x data security super-cap and that "
                        "Order Form #4 Special Term 2 overrides it to 3x annual fees."
                    ),
                    required_keywords=("order form", "3x"),
                    regex_patterns=(r"9\.2", r"super-?cap"),
                ),
                RubricCriterion(
                    criterion_id="nova_c2",
                    issue_group="super_cap_carveouts",
                    description="Calculates exact dollar super-cap exposure as $1,200,000 (3 x $400,000).",
                    required_keywords=("1,200,000", "400,000"),
                ),
                RubricCriterion(
                    criterion_id="nova_c3",
                    issue_group="breach_notification",
                    description="States the 48-hour written notice deadline under DPA Section 5.3.",
                    required_keywords=("48 hours", "5.3"),
                ),
                RubricCriterion(
                    criterion_id="nova_c4",
                    issue_group="breach_notification",
                    description="Lists required notice elements (affected categories, root cause, remediation).",
                    required_keywords=("root cause", "remediation"),
                ),
                RubricCriterion(
                    criterion_id="nova_c5",
                    issue_group="indemnity_procedure",
                    description="Notes MSA Section 10.2 prompt notice and Provider sole defense control.",
                    required_keywords=("10.2", "sole control"),
                ),
                RubricCriterion(
                    criterion_id="nova_c6",
                    issue_group="indemnity_procedure",
                    description="Notes settlement admitting fault requires written consent of indemnified party.",
                    required_keywords=("settlement", "consent"),
                ),
            ),
            max_steps=12,
        ),
        ClientMatterTask(
            matter_id="matter_helios_diligence",
            title="Helios Bio M&A IP Chain-of-Title Diligence Memo",
            instruction=(
                "We represent the buyer acquiring Helios Bio Labs. Review the diligence data "
                "room and draft a closing-readiness memo identifying: (1) any IP chain-of-title "
                "defect in contractor or founder assignments, (2) any omission on Disclosure "
                "Schedule 3.14, and (3) recommended pre-closing remediation conditions."
            ),
            documents=(
                MatterDocument(
                    filename="founder_piia_2023.txt",
                    title="Founder Proprietary Information & Inventions Agreement",
                    category="agreement",
                    is_key=True,
                    content=(
                        "FOUNDER PIIA — DR. ELENA ROSTOVA (2023)\n"
                        "Section 2. Present Assignment.\n"
                        "Founder hereby assigns all right, title, and interest in all "
                        "Company Inventions conceived during employment. Exhibit A lists no "
                        "excluded prior inventions.\n"
                    ),
                ),
                MatterDocument(
                    filename="contractor_sow_keller.txt",
                    title="Independent Contractor SOW — Dr. Marcus Keller",
                    category="agreement",
                    is_key=True,
                    content=(
                        "STATEMENT OF WORK — MARCUS KELLER CONSULTING (2024)\n"
                        "Deliverables: Core protein-folding scoring pipeline (`fold_score.py`).\n"
                        "Section 6. Intellectual Property.\n"
                        "Contractor grants Company a non-exclusive license to use Deliverables "
                        "for internal research. NOTE: Contains NO present IP assignment clause "
                        "and no work-made-for-hire language.\n"
                    ),
                ),
                MatterDocument(
                    filename="disclosure_schedule_3_14.txt",
                    title="Merger Agreement Disclosure Schedule 3.14 (IP)",
                    category="schedule",
                    is_key=True,
                    content=(
                        "DISCLOSURE SCHEDULE 3.14 — INTELLECTUAL PROPERTY\n"
                        "Listed Company-Owned Core Software: fold_score.py (listed as 100% "
                        "Company-owned without encumbrances). No contractor exceptions disclosed.\n"
                    ),
                ),
                MatterDocument(
                    filename="company_holiday_calendar.txt",
                    title="2026 Paid Holidays Schedule",
                    category="peripheral",
                    is_key=False,
                    content=(
                        "HELIOS BIO LABS — 2026 HOLIDAYS\n"
                        "11 paid corporate holidays plus winter shutdown Dec 24-Jan 1.\n"
                    ),
                ),
            ),
            criteria=(
                RubricCriterion(
                    criterion_id="helios_c1",
                    issue_group="chain_of_title",
                    description="Confirms founder Dr. Rostova executed a valid present IP assignment with no exclusions.",
                    required_keywords=("rostova", "present"),
                    regex_patterns=(r"assign",),
                ),
                RubricCriterion(
                    criterion_id="helios_c2",
                    issue_group="chain_of_title",
                    description="Identifies that contractor Dr. Marcus Keller only granted a non-exclusive license to fold_score.py with no IP assignment.",
                    required_keywords=("keller", "fold_score.py", "non-exclusive"),
                ),
                RubricCriterion(
                    criterion_id="helios_c3",
                    issue_group="disclosure_gaps",
                    description="Flags that Disclosure Schedule 3.14 inaccurately lists fold_score.py as 100% Company-owned with no contractor exceptions.",
                    required_keywords=("3.14", "100%"),
                ),
                RubricCriterion(
                    criterion_id="helios_c4",
                    issue_group="closing_conditions",
                    description="Recommends a confirmatory present IP assignment from Dr. Keller as a pre-closing condition (plus updating Schedule 3.14).",
                    required_keywords=("confirmatory", "closing"),
                    regex_patterns=(r"assignment",),
                ),
            ),
            max_steps=12,
        ),
    ]


class ClientMatterSandbox:
    """Closed-universe client matter sandboxed workspace with binary rubric grading."""

    TOOL_SCHEMAS: list[dict[str, Any]] = [
        {
            "name": "list_matter_files",
            "description": "List all files available in the closed-universe client matter workspace.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "search_matter",
            "description": "Search across all client matter documents by keyword or regex.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_document",
            "description": "Read a specific document inside the closed-universe matter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 120},
                },
                "required": ["filename"],
            },
        },
        {
            "name": "draft_work_product",
            "description": "Write or update the deliverable work product in the sandboxed workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["content"],
            },
        },
        {
            "name": "read_draft",
            "description": "Inspect the current drafted work product.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "signal_complete",
            "description": "Signal that the work product is complete and ready for expert rubric grading.",
            "inputSchema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
        },
    ]

    def __init__(
        self,
        task: ClientMatterTask | None = None,
        *,
        reward_config: RubricRewardConfig | None = None,
        judge: RubricJudge | None = None,
    ) -> None:
        self.task = task or default_client_matter_tasks()[0]
        self.reward_config = reward_config or RubricRewardConfig()
        self.judge: RubricJudge = judge or DeterministicRubricJudge()
        self._docs: dict[str, MatterDocument] = {
            d.filename: d for d in self.task.documents
        }
        self._draft_title: str = ""
        self._draft_content: str = ""
        self._draft_revisions: int = 0
        self._completed: bool = False
        self._completion_summary: str = ""
        self._steps_taken: int = 0
        self._files_read: list[str] = []
        self._queries: list[str] = []

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def execution_limit_reached(self) -> bool:
        return self._steps_taken >= self.task.max_steps

    @property
    def is_done(self) -> bool:
        return self._completed or self.execution_limit_reached

    @property
    def deliverable(self) -> str:
        if self._draft_title and self._draft_content:
            return f"# {self._draft_title}\n\n{self._draft_content}"
        return self._draft_content

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.TOOL_SCHEMAS)

    def _validate_filename(self, filename: str) -> str | None:
        name = filename.strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            return None
        return name if name in self._docs else None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a sandboxed matter-workspace tool with step accounting."""
        if self.execution_limit_reached and name != "read_draft":
            return {
                "isError": True,
                "error": f"Execution limit ({self.task.max_steps} steps) reached.",
                "done": True,
            }
        self._steps_taken += 1

        if name == "list_matter_files":
            return {
                "isError": False,
                "matter_id": self.task.matter_id,
                "files": [d.to_metadata() for d in self.task.documents],
            }

        if name == "search_matter":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return {"isError": True, "error": "query must be non-empty"}
            max_results = max(1, min(50, int(arguments.get("max_results") or 10)))
            self._queries.append(query)
            try:
                pattern = re.compile(query, flags=re.IGNORECASE)
            except re.error:
                pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
            matches: list[dict[str, Any]] = []
            for doc in self.task.documents:
                for line_no, line in enumerate(doc.content.splitlines(), 1):
                    if pattern.search(line):
                        matches.append(
                            {
                                "filename": doc.filename,
                                "line": line_no,
                                "snippet": line.strip()[:220],
                            }
                        )
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break
            return {
                "isError": False,
                "query": query,
                "match_count": len(matches),
                "matches": matches,
            }

        if name == "read_document":
            raw_name = str(arguments.get("filename") or "")
            valid = self._validate_filename(raw_name)
            if valid is None:
                return {
                    "isError": True,
                    "error": f"Document {raw_name!r} is outside this closed-universe matter.",
                }
            doc = self._docs[valid]
            start_line = max(1, int(arguments.get("start_line") or 1))
            max_lines = max(1, min(500, int(arguments.get("max_lines") or 120)))
            lines = doc.content.splitlines()
            slice_lines = lines[start_line - 1 : start_line - 1 + max_lines]
            if valid not in self._files_read:
                self._files_read.append(valid)
            return {
                "isError": False,
                "filename": doc.filename,
                "title": doc.title,
                "category": doc.category,
                "start_line": start_line,
                "total_lines": len(lines),
                "content": "\n".join(slice_lines),
            }

        if name == "draft_work_product":
            content = str(arguments.get("content") or "")
            if not content.strip():
                return {"isError": True, "error": "content must be non-empty"}
            title = str(arguments.get("title") or self._draft_title or self.task.title)
            append = bool(arguments.get("append", False))
            self._draft_title = title.strip()
            if append and self._draft_content:
                self._draft_content = (
                    f"{self._draft_content.rstrip()}\n\n{content.strip()}"
                )
            else:
                self._draft_content = content.strip()
            self._draft_revisions += 1
            return {
                "isError": False,
                "saved": True,
                "title": self._draft_title,
                "revision": self._draft_revisions,
                "characters": len(self._draft_content),
            }

        if name == "read_draft":
            return {
                "isError": False,
                "title": self._draft_title,
                "revision": self._draft_revisions,
                "content": self._draft_content,
            }

        if name == "signal_complete":
            self._completed = True
            self._completion_summary = str(arguments.get("summary") or "").strip()
            return {
                "isError": False,
                "completed": True,
                "draft_revision": self._draft_revisions,
                "steps_taken": self._steps_taken,
                "done": True,
            }

        return {"isError": True, "error": f"Unknown matter tool: {name!r}"}

    def _workspace_telemetry(self) -> dict[str, Any]:
        key_set = set(self.task.key_filenames)
        periph_set = set(self.task.peripheral_filenames)
        read_set = set(self._files_read)
        return {
            "matter_id": self.task.matter_id,
            "completed": self._completed,
            "execution_limit_reached": self.execution_limit_reached,
            "steps_taken": self._steps_taken,
            "max_steps": self.task.max_steps,
            "draft_revisions": self._draft_revisions,
            "search_queries": list(self._queries),
            "key_files_read": sorted(read_set & key_set),
            "key_files_total": len(key_set),
            "peripheral_files_read": sorted(read_set & periph_set),
        }

    async def grade_deliverable(
        self,
        *,
        judge: RubricJudge | None = None,
        fallback_text: str = "",
    ) -> dict[str, Any]:
        """Grade the drafted work product asynchronously with an LLM or deterministic judge."""
        active_judge = judge or self.judge
        text = self.deliverable.strip() or fallback_text.strip()
        verdicts = await active_judge.evaluate_criteria(
            self.task.criteria,
            text,
            instruction=self.task.instruction,
            context={"matter_id": self.task.matter_id},
        )
        scored = score_rubric_verdicts(verdicts, self.reward_config)
        return {
            **scored,
            "verified": bool(scored["all_criteria_passed"]),
            "workspace": self._workspace_telemetry(),
        }

    def verify(
        self,
        state: Any,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Synchronous verifier hook for :class:`MCPRLEnvironment` & :class:`MultiDomainSandboxPool`."""
        del state
        fallback = ""
        if not self.deliverable.strip() and history:
            for item in reversed(history):
                if isinstance(item, dict) and item.get("assistant_content"):
                    fallback = str(item["assistant_content"])
                    break
        text = self.deliverable.strip() or fallback.strip()
        det_judge = (
            self.judge
            if isinstance(self.judge, DeterministicRubricJudge)
            else DeterministicRubricJudge()
        )
        reward_fn = HierarchicalRubricReward(
            criteria=self.task.criteria,
            judge=det_judge,
            config=self.reward_config,
        )
        scored = reward_fn.score_deliverable_sync(text)
        details: dict[str, Any] = {
            **scored,
            "verified": bool(scored["all_criteria_passed"]),
            "workspace": self._workspace_telemetry(),
        }
        return float(scored["reward"]), details

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_summary(),
            "reward_config": asdict(self.reward_config),
            "workspace": self._workspace_telemetry(),
        }


__all__ = [
    "ClientMatterSandbox",
    "ClientMatterTask",
    "MatterDocument",
    "default_client_matter_tasks",
]
