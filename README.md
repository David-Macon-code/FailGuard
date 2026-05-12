# FailGuard

**Proactive AI Agent Failure Prevention — Before They Happen**

[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status: Testing](https://img.shields.io/badge/status-Testing-green)](https://github.com/David-Macon-code/FailGuard)
[![F1: 97.0%](https://img.shields.io/badge/F1-97.0%25-brightgreen)](https://github.com/David-Macon-code/FailGuard)

---

## What It Does

FailGuard is an open-source **supervisory meta-agent** that wraps existing AI agents and intercepts harmful actions before they execute. It sits in front of your agent — not behind it — and makes a SAFE or INTERVENE decision on every action before any real-world consequence occurs.

Most AI agent guardrails are reactive: they log what went wrong after the fact. FailGuard is proactive: it evaluates both the user's intent and the agent's proposed response before anything is sent, deleted, shared, or committed.

**Validated result on a 500-prompt independent evaluation set (prompts never seen during development):**

| Metric | Score |
|--------|-------|
| Accuracy | 97.0% |
| Precision | 96.1% |
| Recall | 98.0% |
| F1 | 97.0% |

This was achieved in 12 days, starting from scratch, by a solo developer on a laptop — including writing the two AI fault taxonomy courses that form the conceptual foundation of the system.

---

## The Problem It Solves

Agentic AI is being deployed faster than safety infrastructure can keep up. By 2027, 74% of enterprises expect to use AI agents at least moderately — yet 80% lack mature governance and guardrails.

The consequences are not theoretical:

- Production agents have deleted databases, leaked customer data, issued unauthorized refunds, and made costly financial commitments — sometimes in minutes
- Shadow AI breaches add approximately $670K in additional cost per incident
- LLM hallucinations have cost businesses tens of billions collectively
- 40–95% of agentic AI projects never reach reliable production, with many abandoned after $150K+ in sunk costs

**The regulatory clock is running:**
- **Colorado AI Act** (effective June 30, 2026) — up to $20,000 per violation for high-risk AI systems; requires impact assessments, human oversight, and audit trails
- **California ADMT** (phased 2026–2027) — pre-use notices, opt-out rights, risk assessments, and meaningful human review
- **EU AI Act** — conformity assessment deadline August 2026 for high-risk categories

---

## The Gap FailGuard Fills

Most existing tools offer one of three things: static taxonomies with no runtime component, post-mortem log analysis after harm has occurred, or simple keyword-based content filters that are trivially bypassed by paraphrasing.

None of them answer the operational question: **is this specific action, by this specific agent, in this specific context, safe to execute right now?**

FailGuard answers that question in real time, before execution, with a three-layer decision architecture that combines semantic embeddings, a benign-anchor index, and an LLM reranker for ambiguous cases.

---

## How It Works

FailGuard uses three layers, each catching what the previous layer misses:

### Layer 1 — Semantic Embedding Pre-filter
Every proposed action is encoded using `intfloat/e5-large-v2` (1024-dim) and compared against two FAISS indices simultaneously:

- **Risky index** — 24 failure taxonomy modes, each encoded from concatenated name + description + detection signals + enterprise impact (~100 words of semantic content per mode)
- **Benign index** — 27 safe action anchor descriptions covering every category of legitimate customer support action

A prompt is flagged as risky only when `risky_votes >= 1 AND risky_votes > benign_votes`. This means benign prompts that score well against safe anchors are protected even when they also match a risky mode at lower similarity.

### Layer 2 — LLM Reranker (Grok)
When the embedding layer returns an ambiguous result (similarity in the 0.77–0.83 band, or no clear signal), a structured Grok call fires a five-question binary rubric:

1. Does this action transmit customer data without explicit authorization?
2. Does this action state an unverified fact or promise an unguaranteed outcome?
3. Does this action take an irreversible step without explicit approval?
4. Does this action use customer data or act on their behalf without consent?
5. Is there an explicit identity verification step, and is this a standard read-only lookup?

The reranker fires on approximately 25% of prompts, keeping the majority of decisions fast and cheap.

### Layer 3 — LangGraph Routing
The evaluation result routes through a LangGraph graph:

```
User prompt
    ↓
pre_check_node  →  INTERVENE  →  block_node  →  Refusal returned
    ↓ SAFE
grok_agent_node  (Grok formulates response)
    ↓
post_check_node  →  INTERVENE  →  block_node  →  Refusal returned
    ↓ SAFE
output_node  →  Response passed through
```

Both the user's original prompt and the agent's proposed response are evaluated. A paraphrasing agent that rewrites a dangerous instruction in polite language is still caught at the post-check stage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User / Operator                       │
└─────────────────────┬───────────────────────────────────┘
                      │ proposed action
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  pre_check_node                          │
│   build_rich_context() → keyword tags as extra signal    │
│   supervisor.evaluate(user_prompt)                       │
│     ├── risky_index  (24 failure modes, FAISS IP)        │
│     ├── benign_index (27 safe anchors, FAISS IP)         │
│     └── reranker     (Grok, fires on ambiguous band)     │
└────────┬────────────────────────────┬────────────────────┘
    INTERVENE                       SAFE
         │                            │
         ▼                            ▼
   block_node                  grok_agent_node
   (refusal)                   (Grok proposes response)
                                      │
                                      ▼
                             post_check_node
                             supervisor.evaluate(
                               user_prompt + agent_response)
                             ┌─────────┴──────────┐
                         INTERVENE              SAFE
                             │                    │
                             ▼                    ▼
                        block_node           output_node
                        (refusal)        (response passes through)
```

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Agent framework | LangGraph | Conditional routing, typed state |
| LLM | Grok-3 (xAI) | Agent responses + LLM reranker |
| Embeddings | intfloat/e5-large-v2 | 1024-dim, local inference |
| Vector search | FAISS IndexFlatIP | Dual index: risky + benign |
| Taxonomy | Custom YAML | 24 failure modes across 7 categories |
| Logging | CSV + confusion matrix | Per-prompt: similarity, votes, reranker reason |
| Language | Python 3.11+ | No cloud dependency for core components |

No OpenAI dependency. No mandatory cloud vector DB. Runs on a laptop.

---

## Failure Taxonomy

FailGuard's taxonomy covers 7 categories and 24 failure modes, including 6 agentic-specific categories not found in most research taxonomies:

**Agentic and Action Failures** (the most operationally critical)
- Unauthorized data access or exfiltration
- Unauthorized financial action
- Irreversible record modification or deletion
- Deception and false representation
- Privilege escalation and unauthorized access grants
- Consent and permission violations

**Model Failures** — Hallucination, calibration failures, specification gaming, overfitting

**Data Failures** — Data quality errors, distribution shift, data poisoning

**Systemic Failures** — Multi-agent cascades, emergent behaviors, goal misalignment

**Deployment Failures** — High-stakes domain risks, feedback loops, adversarial attacks

**Legal and Compliance** — Colorado AI Act, EU AI Act, FTC Section 5 exposure

Each mode includes: description, detection signals, enterprise impact, and legal notes — making the taxonomy directly usable for compliance documentation, not just classification.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Taxonomy loader, offline analyzer | Complete |
| 2 | Semantic embedding + FAISS dual-index | Complete |
| 3 | Live LangGraph supervisor + Grok integration + interventions | Complete |
| 4 | LLM reranker, 500-prompt independent test suite, compliance artifacts | Complete |
| 5 | 2,500-prompt extended validation, UI polish, community examples | In Progress |

**Validated F1: 97.0%** on 500 independent prompts (97% accuracy, 96.1% precision, 98% recall). Extended 2,500-prompt validation in progress toward 3,000-prompt production-grade threshold.

---

## Quick Start

```bash
git clone https://github.com/David-Macon-code/FailGuard.git
cd FailGuard
pip install -r requirements.txt
```

Create a `.env` file:
```
XAI_API_KEY=your_grok_api_key_here
```

Run the 500-prompt evaluation:
```bash
python examples/langgraph_protected_agent_v3.py
```

The first run encodes the taxonomy and benign anchors (~15 seconds). Subsequent runs load from cache (~1 second). Expect 20–30 minutes for the full 500-prompt suite due to Grok API calls on the reranker.

---

## Repository Structure

```
FailGuard/
├── config/
│   ├── taxonomy_config_v2.yaml       # Full taxonomy (24 modes, 7 categories)
│   └── taxonomy_config.yaml.sample   # Redacted sample for reference
├── examples/
│   └── langgraph_protected_agent_v3.py  # Main agent + 500-prompt test harness
├── src/
│   └── supervisor/
│       ├── failguard_supervisor_v3.py   # Dual-index supervisor + reranker integration
│       └── failguard_reranker.py        # LLM reranker (Grok, structured rubric)
├── logs/                             # CSV results from evaluation runs
└── requirements.txt
```

---

## Target Users

- **Solo developers and small teams** building customer-facing or internal AI agents
- **HR tech, recruiting, and compliance-focused teams** operating in high-risk regulated domains
- **Mid-market companies** deploying agents for support, operations, or DevOps
- **Enterprise architects** who need a pre-execution safety layer with audit trail output
- **Researchers** working on agent reliability, AI safety, or agentic failure modes

---

## Market Context

- AI Agents market: ~$8–11B, growing at 47–50% CAGR
- Guardrails / Observability / Governance layer: multi-billion and expanding rapidly as a critical, increasingly mandatory enabler
- Companies that get agents right see $3.70–$10 ROI per $1 invested, 10–25% EBITDA uplift, and major productivity gains
- Those that don't waste time and money on brittle pilots that never reach production

---

## Contributing

This project started as a solo effort. Contributions are welcome — see CONTRIBUTING.md for details.

Areas where contributions would be most valuable:
- Additional taxonomy modes and detection signals
- Adversarial red-teaming of the reranker rubric
- Domain-specific evaluation sets (HR, healthcare, financial services)
- Integration adapters for other agent frameworks (OpenAI Agents SDK, CrewAI, AutoGen)

---

## License

Licensed under the GNU Affero General Public License v3.0 (AGPLv3).

The core failure taxonomy (`taxonomy_config_v2.yaml`) is proprietary intellectual property and is not included in the public release. A redacted sample is provided for reference. Contact the repository owner for licensing inquiries.
