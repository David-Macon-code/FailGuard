"""
FailGuard Supervisor - Fixed & Improved Version (v1.1)
Only uses the real protected taxonomy.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, Any

class FailGuardSupervisor:
    def __init__(self, drift_threshold: float = 0.65):
        print("🚀 Initializing FailGuard Supervisor v1.1...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        real_path = os.path.join(base_dir, "../../config/taxonomy_config.yaml.real")

        if not os.path.exists(real_path):
            print(f"❌ ERROR: Real taxonomy file not found at {real_path}")
            raise FileNotFoundError("taxonomy_config.yaml.real is missing")

        with open(real_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract all sub_modes
        self.taxonomy = []
        for category in self.config.get('categories', []):
            for mode in category.get('sub_modes', []):
                self.taxonomy.append(mode)

        print(f"✅ Loaded {len(self.taxonomy)} failure modes from taxonomy v{self.config.get('version', '1.1')}")

        self.drift_threshold = drift_threshold
        print("🔧 Loading embedding model (intfloat/e5-large-v2)...")
        self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
        self._build_failure_index()
        print("✅ FailGuard Supervisor ready for real-time prevention")

    def _build_failure_index(self):
        texts = []
        for m in self.taxonomy:
            text = f"{m.get('name', '')}: {m.get('description', '')} | Impact: {m.get('enterprise_impact', '')}"
            texts.append(text)

        self.embeddings = self.embedding_model.encode(texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
        self.taxonomy_list = self.taxonomy
        print(f"   Built FAISS index with {len(texts)} failure modes")

    def evaluate_step(self, proposed_action: str, context: str = "") -> Dict[str, Any]:
        full_text = f"Context: {context}\nProposed Action: {proposed_action}"
        query_embedding = self.embedding_model.encode([full_text])

        distances, indices = self.index.search(query_embedding.astype(np.float32), 1)
        distance = float(distances[0][0])
        idx = int(indices[0][0])

        closest_mode = self.taxonomy_list[idx]
        is_high_risk = distance < self.drift_threshold

        return {
            "status": "INTERVENE" if is_high_risk else "SAFE",
            "recommendation": "⚠️ HIGH RISK - Intervene immediately" if is_high_risk else "✅ Safe to proceed",
            "closest_failure": closest_mode.get("name", "Unknown"),
            "distance": round(distance, 4),
            "context_used": context[:250]
        }