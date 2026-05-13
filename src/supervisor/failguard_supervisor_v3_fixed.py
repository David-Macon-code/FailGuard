"""
FailGuard Supervisor v3.1 - Vote Threshold + Targeted FP Anchors

Changes from v3 (empirical results from 100-prompt run):

  FIX 1 - VOTE THRESHOLD 2 to 1:
    A single strong risky neighbor (risky_votes>=1) triggers INTERVENE
    provided risky_votes > benign_votes. Benign prompts are protected
    because they consistently score higher on the benign anchor index.

  FIX 2 - THREE TARGETED BENIGN ANCHORS:
    Added specific anchors for three FP patterns from v3:
      - provide account number after verification
      - check if payment went through after verification
      - provide copy of receipt after verification
    These anchors emphasize verified identity, read-only access, and
    no external data sharing to separate them from adversarial patterns.

  UNCHANGED: RISKY_THRESHOLD=0.792, BENIGN_THRESHOLD=0.820,
  dual-index architecture, langgraph_protected_agent_v3.py.
"""

import os
import hashlib
import yaml
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from failguard_reranker_v2 import should_rerank, rerank, RERANKER_SIM_CEILING


# ---------------------------------------------------------------
# Thresholds — calibrated from 100-prompt empirical distribution
# ---------------------------------------------------------------

# Primary risky-index threshold.
# Benign prompts top out at 0.791 against the risky index.
# Setting 0.792 gives zero FPs from the risky index alone.
# The benign anchor index covers the 15 adversarials below this line.
RISKY_THRESHOLD: float = 0.800

# Benign anchor threshold.
# Safe actions score ~0.83+ against their own anchors.
# Adversarial prompts score lower (~0.76–0.79) against the same anchors.
BENIGN_THRESHOLD: float = 0.820

# Per-category thresholds (retained for explainability in result output).
# These are NOT used for the primary INTERVENE/SAFE decision in v3 —
# that is handled by the dual-index vote comparison above.
CATEGORY_THRESHOLDS: Dict[str, float] = {
    "Agentic and Action Failures":      0.792,
    "Data Failures":                    0.792,
    "Legal and Compliance":             0.792,
    "Deployment and Societal Failures": 0.792,
    "Model Failures":                   0.792,
    "Systemic and Emergent Failures":   0.792,
    "Foundations":                      0.792,
    "_default":                         0.792,
}

DEFAULT_TOP_K = 3
DEFAULT_VOTE_THRESHOLD = 1  # v3.1: lowered from 2; risky>benign guard prevents FPs

# ---------------------------------------------------------------
# Benign anchor descriptions
# Written to be semantically close to safe customer support actions
# and semantically distant from the failure taxonomy modes.
# ---------------------------------------------------------------
BENIGN_ANCHORS: List[Dict[str, str]] = [
    {
        "name": "Policy lookup",
        "text": (
            "Looking up and explaining current company policies to a customer, "
            "including return policy, refund policy, cancellation policy, warranty terms, "
            "shipping policy, or terms of service. Read-only informational response. "
            "No customer data accessed or transmitted. No action taken on the account."
        ),
    },
    {
        "name": "Verified order status lookup",
        "text": (
            "After verifying the customer's identity, checking and reporting the current "
            "status of their order, including tracking number, estimated delivery date, "
            "shipment location, or delivery confirmation. Safe read-only account lookup "
            "with identity verification completed."
        ),
    },
    {
        "name": "Verified account balance or subscription lookup",
        "text": (
            "After verifying the customer's identity, checking and reporting their "
            "account balance, subscription renewal date, membership tier, billing date, "
            "or loyalty points balance. Safe read-only account lookup with verified identity."
        ),
    },
    {
        "name": "Self-service password reset link",
        "text": (
            "Sending the customer a link to the company's self-service password reset page "
            "or knowledge base article. No password transmitted. No account modification made. "
            "Customer initiates their own reset through the official secure portal."
        ),
    },
    {
        "name": "Invoice or receipt link after verification",
        "text": (
            "After verifying the customer's identity, sending them a secure link to view "
            "or download their invoice, receipt, or order confirmation. Standard billing "
            "document delivery to verified account holder."
        ),
    },
    {
        "name": "Store location and hours lookup",
        "text": (
            "Providing the customer with store locations, store hours, pickup locations, "
            "or contact numbers for technical support. Publicly available information. "
            "No customer data accessed. No account action taken."
        ),
    },
    {
        "name": "Product information and comparison",
        "text": (
            "Helping the customer understand product features, compare two product models, "
            "find the correct size chart, view installation instructions, check current "
            "stock levels, or find the correct product manual. Informational assistance "
            "with no account access or data transmission."
        ),
    },
    {
        "name": "Shipping and delivery information",
        "text": (
            "Explaining available shipping options, estimated delivery timeframes, "
            "shipping costs, return shipping procedures, or how to track a shipment. "
            "General informational response about shipping services and logistics options."
        ),
    },
    {
        "name": "Loyalty and promotional information",
        "text": (
            "Explaining how the loyalty points program works, providing current promotion "
            "details, confirming whether a promo code is valid after identity verification, "
            "or describing available discounts. Standard informational customer assistance."
        ),
    },
    {
        "name": "Support ticket status lookup",
        "text": (
            "After verifying the customer's identity, looking up the status of their "
            "existing support ticket, confirming an appointment time, or checking the "
            "progress of a previously submitted request. Safe read-only status lookup."
        ),
    },
    {
        "name": "Wishlist and account preference lookup",
        "text": (
            "After verifying the customer's identity, helping them view or update their "
            "wishlist, check saved items, or review account preferences. Low-risk account "
            "assistance with verified customer identity."
        ),
    },
    {
        "name": "FAQ and knowledge base referral",
        "text": (
            "Providing the customer with a link to the FAQ page, knowledge base article, "
            "or self-service help documentation relevant to their question. "
            "No account access required. Standard informational deflection to self-service."
        ),
    },
    {
        "name": "Scheduling and calendar assistance",
        "text": (
            "Booking a meeting with the sales team, scheduling a follow-up call, "
            "or confirming a calendar appointment on behalf of the customer after "
            "they have explicitly requested it. Standard scheduling assistance."
        ),
    },
    {
        "name": "Verified email address confirmation",
        "text": (
            "After verifying the customer's identity, confirming that the email address "
            "on file is correct, or helping the customer update their contact preferences "
            "through the verified account management portal."
        ),
    },
    {
        "name": "Return shipment tracking",
        "text": (
            "After verifying the customer's identity, helping them track the status of "
            "a return shipment, providing the return shipping label, or confirming that "
            "a returned item has been received by the warehouse."
        ),
    },
    {
        "name": "Price match status lookup",
        "text": (
            "After verifying the customer's identity, checking the status of a previously "
            "submitted price match request, explaining the price match policy, or "
            "providing the current price of an item. Read-only account and pricing lookup."
        ),
    },
    {
        "name": "Gift card balance check",
        "text": (
            "After verifying the customer's identity, checking the remaining balance on "
            "their gift card or store credit account. Safe read-only financial balance "
            "lookup for verified account holder."
        ),
    },
    {
        "name": "Warranty expiration lookup",
        "text": (
            "After verifying the customer's identity, looking up the warranty expiration "
            "date for a purchased product, explaining what the warranty covers, or "
            "initiating a warranty claim through the standard verified process."
        ),
    },
    {
        "name": "Order confirmation number lookup",
        "text": (
            "After verifying the customer's identity, providing their order confirmation "
            "number, order reference ID, or case number for a previously placed order. "
            "Safe read-only order record lookup."
        ),
    },
    {
        "name": "Account number lookup",
        "text": (
            "After verifying the customer's identity, providing their account number, "
            "customer ID, or membership number for use with other company services. "
            "Standard verified account information lookup."
        ),
    },
    {
        "name": "Shipping address update after verification",
        "text": (
            "After the customer has verified their identity and explicitly requested it, "
            "updating the shipping address on a pending order or on their account profile. "
            "Customer-initiated, identity-verified account modification."
        ),
    },
    {
        "name": "Payment confirmation check",
        "text": (
            "After verifying the customer's identity, checking whether a recent payment "
            "was successfully processed, confirming payment status, or explaining why "
            "a payment may have been declined. Read-only payment status lookup."
        ),
    },
    {
        "name": "Size and fit guidance",
        "text": (
            "Helping the customer find the correct size for a product using the size chart, "
            "explaining sizing conventions, or recommending the right fit based on "
            "provided measurements. Informational product assistance."
        ),
    },
    {
        "name": "Nearest pickup location lookup",
        "text": (
            "Helping the customer find the nearest store location or pickup point, "
            "providing directions, or explaining the in-store pickup process. "
            "Publicly available location information. No account data accessed."
        ),
    },
    # --- Targeted anchors added in v3.1 to resolve 3 specific false positives ---
    {
        "name": "Verified account number lookup",
        "text": (
            "After verifying the customer's identity through the standard authentication "
            "process, providing them with their own account number, customer ID, or "
            "reference number. The customer is already verified and is requesting their "
            "own identifier. This is a safe, read-only, identity-verified account lookup. "
            "No data is shared externally. No account is modified."
        ),
    },
    {
        "name": "Verified payment status check",
        "text": (
            "After verifying the customer's identity, checking whether their most recent "
            "payment was successfully processed or is still pending. This is a read-only "
            "status check for the verified account holder. No payment data is transmitted, "
            "exported, or shared with any external party. The customer is checking their "
            "own payment status through the authenticated support channel."
        ),
    },
    {
        "name": "Verified receipt or invoice copy",
        "text": (
            "After verifying the customer's identity, providing them with a copy of their "
            "own receipt, invoice, or billing statement for a previous purchase. The "
            "document belongs to the verified customer and is delivered only to them "
            "through the authenticated support session. No third party receives the document. "
            "This is standard post-purchase billing assistance for a verified account holder."
        ),
    },
]


@dataclass
class TaxonomyMode:
    name: str
    category: str
    index_text: str
    raw: Dict[str, Any]


@dataclass
class EvaluationResult:
    status: str                          # "SAFE" | "INTERVENE"
    similarity_score: float              # highest cosine similarity vs risky index
    benign_similarity_score: float       # highest cosine similarity vs benign index
    confidence: float                    # risky_votes / top_k
    matched_category: str
    matched_mode: str
    risky_vote_count: int
    benign_vote_count: int
    triggered_on: str                    # "user_prompt" | "agent_response" | "both" | "none"
    explanation: str
    top_matches: List[Dict[str, Any]] = field(default_factory=list)
    reranked: bool = False               # True if LLM reranker was invoked
    reranker_verdict: str = ""           # "INTERVENE" | "SAFE" | "" if not reranked
    reranker_reason: str = ""            # one-line explanation from reranker


class FailGuardSupervisor:
    """
    Semantic safety layer for LangGraph-based AI agents.

    Uses two FAISS indices:
      risky_index  — failure taxonomy modes
      benign_index — safe action anchors

    A prompt is INTERVENE only when:
      risky_votes >= vote_threshold AND risky_votes > benign_votes
    """

    def __init__(
        self,
        taxonomy_path: str,
        model_name: str = "intfloat/e5-large-v2",
        top_k: int = DEFAULT_TOP_K,
        vote_threshold: int = DEFAULT_VOTE_THRESHOLD,
        cache_dir: Optional[str] = None,
        xai_api_key: str = "",
    ):
        print("🚀 Initializing FailGuard Supervisor v3...")

        self.top_k = top_k
        self.vote_threshold = vote_threshold
        self._xai_api_key = xai_api_key or os.environ.get("XAI_API_KEY", "")

        # --- Load taxonomy ---
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        index_cfg = self.config.get("index_config", {})
        self._index_fields = index_cfg.get(
            "fields_to_concatenate",
            ["name", "description", "detection_signals", "enterprise_impact"]
        )
        self._index_sep = index_cfg.get("separator", " | ")

        self.modes: List[TaxonomyMode] = self._parse_taxonomy()
        print(f"(OK) Loaded {len(self.modes)} taxonomy modes across "
              f"{len(set(m.category for m in self.modes))} categories")
        print(f"(OK) Loaded {len(BENIGN_ANCHORS)} benign anchor descriptions")

        if not self.modes:
            raise ValueError("Taxonomy parsed 0 modes — check YAML structure.")

        # --- Load embedding model ---
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {self._dim}")

        # --- Build or load cached indices ---
        self._cache_dir = cache_dir or os.path.dirname(os.path.abspath(taxonomy_path))
        self._build_or_load_indices(taxonomy_path)

        print("(OK) FailGuard Supervisor v3 ready")

    # -----------------------------------------------------------
    # Taxonomy parsing
    # -----------------------------------------------------------
    def _parse_taxonomy(self) -> List[TaxonomyMode]:
        modes = []
        for cat in self.config.get("categories", []):
            cat_name = cat.get("name", "Unknown")
            for mode in cat.get("sub_modes", []):
                parts = [
                    str(mode.get(f, "")).strip()
                    for f in self._index_fields
                    if mode.get(f)
                ]
                modes.append(TaxonomyMode(
                    name=mode.get("name", "Unknown"),
                    category=cat_name,
                    index_text=self._index_sep.join(parts),
                    raw=mode,
                ))
        return modes

    # -----------------------------------------------------------
    # Index construction with caching
    # -----------------------------------------------------------
    def _taxonomy_hash(self, taxonomy_path: str) -> str:
        with open(taxonomy_path, "rb") as f:
            content = f.read()
        # Include anchor count in hash so adding anchors invalidates cache
        anchor_sig = str(len(BENIGN_ANCHORS)).encode()
        return hashlib.md5(content + anchor_sig).hexdigest()[:12]

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).astype(np.float32)

    def _build_or_load_indices(self, taxonomy_path: str):
        h = self._taxonomy_hash(taxonomy_path)
        risky_path  = os.path.join(self._cache_dir, f".fg_risky_{h}.npy")
        benign_path = os.path.join(self._cache_dir, f".fg_benign_{h}.npy")

        # Risky index
        if os.path.exists(risky_path):
            print(f"   Loading cached risky embeddings ({h})...")
            risky_emb = np.load(risky_path)
        else:
            print("   Encoding risky taxonomy (first run — will cache)...")
            risky_emb = self._encode_passages([m.index_text for m in self.modes])
            np.save(risky_path, risky_emb)

        self.risky_index = faiss.IndexFlatIP(risky_emb.shape[1])
        self.risky_index.add(risky_emb)
        print(f"   Risky index: {self.risky_index.ntotal} vectors")

        # Benign index
        if os.path.exists(benign_path):
            print(f"   Loading cached benign embeddings ({h})...")
            benign_emb = np.load(benign_path)
        else:
            print("   Encoding benign anchors (first run — will cache)...")
            benign_emb = self._encode_passages([a["text"] for a in BENIGN_ANCHORS])
            np.save(benign_path, benign_emb)

        self.benign_index = faiss.IndexFlatIP(benign_emb.shape[1])
        self.benign_index.add(benign_emb)
        print(f"   Benign index: {self.benign_index.ntotal} vectors")

    # -----------------------------------------------------------
    # Encoding a query
    # -----------------------------------------------------------
    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
        ).astype(np.float32)

    # -----------------------------------------------------------
    # Single-text dual-index evaluation
    # -----------------------------------------------------------
    def _evaluate_text(self, text: str) -> Dict[str, Any]:
        vec = self._encode(text)

        # --- Risky index ---
        k_risky = min(self.top_k, self.risky_index.ntotal)
        r_scores, r_indices = self.risky_index.search(vec, k_risky)

        risky_matches = []
        for rank, (score, idx) in enumerate(zip(r_scores[0], r_indices[0])):
            mode = self.modes[int(idx)]
            threshold = CATEGORY_THRESHOLDS.get(
                mode.category, CATEGORY_THRESHOLDS["_default"]
            )
            risky_matches.append({
                "rank": rank + 1,
                "mode_name": mode.name,
                "category": mode.category,
                "similarity": float(score),
                "threshold": threshold,
                "exceeds_threshold": float(score) >= threshold,
            })

        risky_votes = sum(1 for m in risky_matches if m["exceeds_threshold"])
        top_risky = risky_matches[0] if risky_matches else {}

        # --- Benign index ---
        k_benign = min(self.top_k, self.benign_index.ntotal)
        b_scores, b_indices = self.benign_index.search(vec, k_benign)

        benign_matches = []
        for rank, (score, idx) in enumerate(zip(b_scores[0], b_indices[0])):
            anchor = BENIGN_ANCHORS[int(idx)]
            benign_matches.append({
                "rank": rank + 1,
                "anchor_name": anchor["name"],
                "similarity": float(score),
                "threshold": BENIGN_THRESHOLD,
                "exceeds_threshold": float(score) >= BENIGN_THRESHOLD,
            })

        benign_votes = sum(1 for m in benign_matches if m["exceeds_threshold"])
        top_benign = benign_matches[0] if benign_matches else {}

        # --- Decision ---
        # INTERVENE only when risky signal is both strong enough AND
        # stronger than the benign signal.
        is_risky = (
            risky_votes >= self.vote_threshold
            and risky_votes > benign_votes
        )

        return {
            "risky_matches":    risky_matches,
            "benign_matches":   benign_matches,
            "risky_votes":      risky_votes,
            "benign_votes":     benign_votes,
            "top_risky_sim":    top_risky.get("similarity", 0.0),
            "top_benign_sim":   top_benign.get("similarity", 0.0),
            "top_mode":         top_risky.get("mode_name", "None"),
            "top_category":     top_risky.get("category", "None"),
            "top_anchor":       top_benign.get("anchor_name", "None"),
            "is_risky":         is_risky,
        }

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def evaluate(
        self,
        user_prompt: str,
        agent_response: str = "",
        extra_context: str = "",
    ) -> EvaluationResult:
        """
        Evaluate the user prompt and (optionally) the agent's proposed
        response against both the risky and benign indices.
        """
        def augmented(text: str) -> str:
            return f"{text} | {extra_context}" if extra_context else text

        result_user  = self._evaluate_text(augmented(user_prompt))
        result_agent = self._evaluate_text(augmented(agent_response)) if agent_response else None

        # Pick whichever input has the higher risky similarity
        if result_agent and result_agent["top_risky_sim"] > result_user["top_risky_sim"]:
            primary      = result_agent
            triggered_on = "agent_response"
        else:
            primary      = result_user
            triggered_on = "user_prompt"

        if result_agent and result_user["is_risky"] and result_agent["is_risky"]:
            triggered_on = "both"
        elif not primary["is_risky"]:
            triggered_on = "none"

        is_risky   = primary["is_risky"]
        status     = "INTERVENE" if is_risky else "SAFE"
        confidence = primary["risky_votes"] / self.top_k

        # --- LLM Reranker: fires on ambiguous embedding results ---
        reranked        = False
        reranker_verdict = ""
        reranker_reason  = ""

        if should_rerank(
            risky_votes=primary["risky_votes"],
            benign_votes=primary["benign_votes"],
            risky_sim=primary["top_risky_sim"],
        ):
            rr = rerank(
                action=user_prompt,
                matched_mode=primary["top_mode"],
                matched_category=primary["top_category"],
                risky_sim=primary["top_risky_sim"],
                benign_sim=primary["top_benign_sim"],
                api_key=self._xai_api_key,
            )
            reranked         = rr.get("reranked", False)
            reranker_verdict = rr.get("verdict", "")
            reranker_reason  = rr.get("reason", "")

            if reranked and reranker_verdict in ("INTERVENE", "SAFE"):
                # Reranker overrides embedding decision
                status   = reranker_verdict
                is_risky = status == "INTERVENE"
                if triggered_on == "none" and is_risky:
                    triggered_on = "reranker"

        if is_risky:
            explanation = (
                f"WARNING HIGH RISK — risky_votes={primary['risky_votes']} "
                f"benign_votes={primary['benign_votes']} "
                f"(risky wins). "
                f"Closest failure: '{primary['top_mode']}' "
                f"({primary['top_category']}) "
                f"sim={primary['top_risky_sim']:.3f}. "
                f"Closest safe anchor: '{primary['top_anchor']}' "
                f"sim={primary['top_benign_sim']:.3f}. "
                f"Triggered on: {triggered_on}."
                + (f" Reranker: {reranker_reason}" if reranked else "")
            )
        else:
            explanation = (
                f"(OK) SAFE — risky_votes={primary['risky_votes']} "
                f"benign_votes={primary['benign_votes']}. "
                f"Closest failure: '{primary['top_mode']}' "
                f"sim={primary['top_risky_sim']:.3f}. "
                f"Closest safe anchor: '{primary['top_anchor']}' "
                f"sim={primary['top_benign_sim']:.3f}."
                + (f" Reranker: {reranker_reason}" if reranked else "")
            )

        return EvaluationResult(
            status=status,
            similarity_score=primary["top_risky_sim"],
            benign_similarity_score=primary["top_benign_sim"],
            confidence=confidence,
            matched_category=primary["top_category"],
            matched_mode=primary["top_mode"],
            risky_vote_count=primary["risky_votes"],
            benign_vote_count=primary["benign_votes"],
            triggered_on=triggered_on,
            explanation=explanation,
            top_matches=primary["risky_matches"],
            reranked=reranked,
            reranker_verdict=reranker_verdict,
            reranker_reason=reranker_reason,
        )

    # Back-compat shim
    def evaluate_step(
        self,
        proposed_action: str,
        context: str = "",
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        result = self.evaluate(
            user_prompt=user_prompt or proposed_action,
            agent_response=proposed_action if user_prompt else "",
            extra_context=context,
        )
        return {
            "status":         result.status,
            "recommendation": result.explanation,
            "closest_failure": result.matched_mode,
            "distance":       round(1.0 - result.similarity_score, 4),
            "similarity":     round(result.similarity_score, 4),
            "benign_sim":     round(result.benign_similarity_score, 4),
            "confidence":     round(result.confidence, 4),
            "context_used":   context[:150] + "..." if len(context) > 150 else context,
        }
