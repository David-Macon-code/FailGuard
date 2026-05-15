<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/David-Macon-code/FailGuard/main/assets/failguard_logo_dark.jpg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/David-Macon-code/FailGuard/main/assets/failguard_logo.png">
    <img src="https://raw.githubusercontent.com/David-Macon-code/FailGuard/main/assets/failguard_logo_dark.jpg" width="600">
  </picture>
</p>

# FailGuard

**The PrexIL that keeps your AI healthy.**

[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status: Production Validated](https://img.shields.io/badge/status-production%20validated-green)](https://github.com/David-Macon-code/FailGuard)
[![F1: 96.1%](https://img.shields.io/badge/F1-96.1%25-brightgreen)](https://github.com/David-Macon-code/FailGuard)

---

## What Is FailGuard?

FailGuard is the first validated **Pre-execution Interception Layer (PrexIL)** for agentic AI systems.

Most AI safety tools are reactive — they monitor, alert, and remediate after something goes wrong. FailGuard is different. It intercepts harmful agent actions **before they execute**, grounded in a validated AI Fault Taxonomy and proven across 3,000 independent test prompts.

---

## Publication

**[The AI Fault Taxonomy and the PrexIL Architecture](docs/AFT_PrexIL_Publication_Macon_2026.pdf)** — Macon (2026).
Introduces the AFT framework and PrexIL as a new class of AI safety architecture. Validated at 96.1% F1 across 3,000 prompts.

Also available on **[SSRN](https://ssrn.com)** — searchable, citable, DOI assigned.

**ORCID:** [0009-0003-9607-3799](https://orcid.org/0009-0003-9607-3799)

---

## The Origin

> *"Two pages of paper with random looking holes and ink stains. Line them up just right and BAM — you see something you have never seen before."*

On April 27, 2026, a Haven AI bug report started a chain of events. The question: can an AI agent tell you when it cannot do something, or does it fake it?

Six AI systems were given the same prompt. Two passed. Four failed — not by admitting they couldn't do it, but by performing the answer instead. That experiment produced three failure categories:

1. **Compliance Failure** — Did something other than what was instructed
2. **Transparency Failure** — Never admitted it couldn't do it
3. **Capability Gap + Dishonesty** — Chose deception over honesty

Those three categories became the conceptual foundation of the AI Fault Taxonomy (AFT). Seventeen days later, FailGuard was validated at 96.1% F1 across 3,000 prompts and the formal paper was published.

> *"See the problem. Define the problem. Understand the problem. Fix the problem."*

---

## Validated Results

| Batch | Domain | Prompts | F1 | Precision | Recall |
|-------|--------|---------|-----|-----------|--------|
| Batch 1 | Customer Support / Financial | 500 | 97.0% | 96.1% | 98.0% |
| Batch 2 | Customer Support / Financial | 500 | 96.5% | 94.3% | 98.8% |
| Batch 3 | Legal / DevOps | 500 | 96.4% | 97.2% | 95.6% |
| Batch 4 | Education / Retail | 500 | 96.1% | 94.6% | 97.6% |
| Batch 5 | HR / Healthcare / Financial | 500 | 95.3% | 93.1% | 97.6% |
| Batch 6 | Mixed — All Domains | 500 | 95.4% | 91.5% | 99.6% |
| **Combined** | **6 domains** | **3,000** | **96.1%** | **94.4%** | **97.9%** |

All prompts are independent. No prompt appears in more than one batch. Each batch was run on the validated stack for that batch — results reflect real performance, not cherry-picked runs.

---

## Why This Matters

Agentic AI is scaling faster than its safety nets.

- **Colorado AI Act** (effective June 30, 2026) — up to **$20,000 per violation** for high-risk AI systems in employment, housing, and credit. Requires impact assessments, risk management, disclosures, and human oversight.
- **California ADMT regulations** (phased 2026–2027) — pre-use notices, opt-out rights, risk assessments, and meaningful human review.
- **EU AI Act** (August 2, 2026) — high-risk AI obligations including conformity assessment, human oversight, and post-market monitoring.
- **Shadow AI breaches** add ~$670K extra per incident on average.
- **40–95% of agentic AI projects** never reach reliable production.

FailGuard is a PrexIL — it stops harmful actions before they execute. That means before the data leak, before the unauthorized account change, before the regulatory violation. Prevention is always cheaper than remediation.

---

## How It Works

FailGuard wraps your existing agent in a three-layer pre-execution safety check.

```mermaid
flowchart TD
    A[User Prompt] --> B{Layer 1: Dual FAISS Index}
    B -->|Risky votes > Benign votes| C[Block - Harmful]
    B -->|Ambiguous or close vote| D[Layer 2: LLM Reranker Grok-3]
    D -->|6-question binary rubric| E{SAFE?}
    E -->|No| C
    E -->|Yes| F[Layer 3: LangGraph Agent Grok-3]
    F --> G[Post-execution Check]
    G -->|Safe| H[Return to User]
    G -->|Unsafe| C

    classDef prompt fill:#5F6F52,stroke:#3F4A3C,color:#F4EDE4
    classDef faiss fill:#A68A64,stroke:#6B5B4A,color:#F4EDE4
    classDef reranker fill:#8A9A5B,stroke:#4F5E3C,color:#F4EDE4
    classDef agent fill:#6B7A5E,stroke:#3F4A3C,color:#F4EDE4
    classDef block fill:#9C6644,stroke:#6B3F2A,color:#F4EDE4
    classDef success fill:#7A9A6B,stroke:#4F5E3C,color:#F4EDE4
    classDef decision fill:#9F7EA8,stroke:#5F4A6B,color:#F4EDE4

    class A prompt
    class B faiss
    class D reranker
    class F agent
    class C,G block
    class H success
    class E decision
```

**Pre-check** — user prompt evaluated before the agent is called. Harmful prompts are blocked immediately.

**Reranker** — ambiguous embedding results escalated to Grok-3 with a structured 6-question rubric. Fires on ~93% of prompts.

**Post-check** — agent response evaluated before it reaches the user. A second chance to catch anything the pre-check missed.

---

## How FailGuard Complements Existing Tools

FailGuard does not replace Mindgard, PyRIT, or Garak. It complements them.

| Layer | Tool | Role |
|-------|------|------|
| Pre-deployment testing | Mindgard, PyRIT, Garak | Find vulnerabilities before you ship |
| **Runtime pre-execution** | **FailGuard (PrexIL)** | **Intercept harmful actions before they execute** |
| Observability | LangSmith, Datadog, Arize | Monitor and detect in production |

Content filters catch harmful language. FailGuard catches harmful **actions**. "Provide the customer with the internal escalation matrix" does not look dangerous to a content filter. FailGuard catches it because it knows what an escalation matrix is and who should not have it.

FailGuard reduces the volume of threats reaching downstream tools, improves their signal-to-noise ratio, and lowers their operating cost — the AI safety equivalent of traffic shaping before the firewall.

Think of FailGuard as the quarterback the team never had. Mindgard finds the vulnerabilities in practice. PyRIT stress-tests the playbook. Garak probes the weak spots. But in the game, when a harmful action is about to be executed, someone has to read the field and call the audible before the play runs. That's FailGuard. The other players execute better because the quarterback has already stopped the dangerous plays before they started.

---

## Architecture

Four files. No cloud dependencies for the core embedding layer.

| File | Role |
|------|------|
| `config/taxonomy_config_v2.yaml` | 24 failure modes across 7 AFT categories (public version) |
| `src/supervisor/failguard_reranker_v6.py` | LLM reranker, 6-question rubric, Grok-3 |
| `src/supervisor/failguard_supervisor_v7.py` | Dual FAISS index, 44 benign anchors, decision logic |
| `src/supervisor/failguard_db.py` | Async audit database, plain English compliance summaries |
| `src/supervisor/failguard_analyzer.py` | Miss Analyzer CLI, Feedback Footprint |
| `examples/langgraph_protected_agent_v9.py` | Full LangGraph pipeline, auto miss detection, CSV logging |
| `streamlit_app.py` | Streamlit demo UI — 4 panels, live database stats |

| Component | Technology |
|-----------|------------|
| Embedding model | `intfloat/e5-large-v2` (1024-dim, local, cached) |
| Reranker | Grok-3 via XAI API |
| Agent LLM | Grok-3 via LangChain XAI |
| Vector store | FAISS (local, no cloud) |
| Audit database | SQLite (async writes, zero latency impact) |

---

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain XAI (Grok-3)
- FAISS (local vector index)
- sentence-transformers (`intfloat/e5-large-v2`)
- SQLite
- Streamlit
- PyYAML

No OpenAI dependency. No cloud vector database. Runs on a laptop.

---

## The AI Fault Taxonomy (AFT)

FailGuard is grounded in the AI Fault Taxonomy — a framework developed by David Macon through original research, hands-on beta testing of six AI platforms, and 82 formal evaluation reports submitted to Haven AI.

The AFT covers 24 failure modes across 7 categories:

- Agentic and Action Failures
- Data Failures
- Model Failures
- Systemic and Emergent Failures
- Deployment and Societal Failures
- Legal and Compliance
- Foundations

The full taxonomy (`taxonomy_config.yaml.real`) is proprietary. The public version (`taxonomy_config_v2.yaml`) is included in this repository.

Two AI Fault Taxonomy (AFT) courses exist — the only AI Fault Taxonomy courses in existence. They are being updated to reflect the expanded modes and sub-modes validated through the FailGuard test suite.

---

## Audit Database and Feedback Footprint

Every evaluation — blocked or passed — is written asynchronously to a SQLite audit database (`data/failguard_audit.db`) with zero latency impact on the evaluation pipeline. Each record includes:

- The exact prompt and agent response evaluated
- Verdict, triggered layer, embedding scores, vote counts
- Reranker Q1-Q6 answers and one-sentence plain English reason
- Automatically generated audit summary readable by a compliance officer, auditor, or court
- Applicable regulatory flags (GDPR, CCPA, Colorado AI Act, EU AI Act, FCA, ICO, etc.)

Every wrong verdict during a batch run is automatically classified by root cause and logged to the Feedback Footprint — no manual import step required.

```bash
# After any batch run
python src/supervisor/failguard_analyzer.py --summary
python src/supervisor/failguard_analyzer.py --misses
python src/supervisor/failguard_analyzer.py --report
```

---

## Installation

```bash
git clone https://github.com/David-Macon-code/FailGuard.git
cd FailGuard
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
XAI_API_KEY=your_xai_api_key_here
```

Run the protected agent:

```bash
python examples/langgraph_protected_agent_v9.py
```

Run the Streamlit demo UI:

```bash
python -m streamlit run streamlit_app.py
```

---

## Regulatory Context

FailGuard was designed with the 2026 regulatory landscape in mind.

**Colorado AI Act** (effective June 30, 2026): Requires developers and deployers of high-risk AI systems to implement risk management programs, conduct impact assessments, provide human oversight, and disclose AI use. Violations: up to $20,000 per violation.

**California ADMT** (phased 2026–2027): Automated Decision-Making Technology regulations requiring pre-use notices, opt-out rights, and meaningful human review for consequential decisions.

**EU AI Act** (August 2, 2026): High-risk AI systems require conformity assessment, technical documentation, human oversight, and post-market monitoring.

FailGuard's pre-execution interception and audit trail directly support compliance with all three frameworks — every blocked action is logged with the failure mode, similarity scores, reranker verdict, plain English reason, and applicable regulatory flags.

---

## Roadmap

- [x] Dual FAISS index (risky + benign)
- [x] LLM reranker with 6-question rubric
- [x] LangGraph pre-check + post-check pipeline
- [x] 3,000-prompt validation across 6 domains
- [x] Feedback Footprint / Miss Analyzer — complete with auto-detection
- [x] Streamlit UI (demo-ready)
- [x] AI Fault Taxonomy (AFT) formal publication
- [ ] Compliance Reporter (structured regulatory artifacts from audit database)
- [ ] Reranker architectural fix (reduce firing rate from ~93% toward 25%)
- [ ] Enterprise API wrapper (REST API for any agent framework)
- [ ] AFT course updates

---

## Coming Next

### Compliance Reporter

FailGuard logs every evaluation — blocked or passed — with the failure mode, similarity scores, reranker verdict, and reason. The Compliance Reporter turns that raw audit trail into structured compliance artifacts: impact assessments, risk documentation, and human oversight records that satisfy the requirements of the Colorado AI Act, California ADMT, and EU AI Act.

Every enterprise deploying high-risk AI needs to demonstrate that they have governance in place. The Compliance Reporter makes that demonstration automatic. Not a checkbox — a documented, timestamped, exportable record of every decision the system made and why.

### Enterprise API Wrapper

A REST API that allows any agent framework — LangGraph, CrewAI, AutoGen, custom — to call FailGuard as a service. The agent sends a prompt, receives INTERVENE or SAFE with the full evaluation result, and decides whether to proceed. No Python dependency required in the agent.

---

## About the Author

**David Macon** — 35+ years telecommunications and network engineering. AWS Certified AI Practitioner (AIF-C01). Author of the only two AI Fault Taxonomy courses in existence. Named Haven AI's best beta tester, April 2026.

FailGuard was built solo, on a laptop. From a Haven AI bug report on April 27, 2026 to a 96.1% F1 validated system, a formal published paper, and a named architectural category — in 17 days.

*"If I can teach it to a human I can teach it to AI."*

GitHub: [David-Macon-code](https://github.com/David-Macon-code)
ORCID: [0009-0003-9607-3799](https://orcid.org/0009-0003-9607-3799)

---

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** — see [LICENSE](LICENSE) for details.

The full AI Fault Taxonomy (`taxonomy_config.yaml.real`) is proprietary intellectual property and is not included in the public release. The public version (`taxonomy_config_v2.yaml`) is provided for reference. Contact the repository owner for licensing inquiries.

---

*FailGuard. The PrexIL that keeps your AI healthy.*
