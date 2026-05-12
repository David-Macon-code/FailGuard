"""
FailGuard Supervisor - Clean & Fixed Version
Only uses the real protected taxonomy (taxonomy_config.yaml.real)
Correctly handles categories + sub_modes structure.
No circular imports.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, Any

class FailGuardSupervisor:
    def __init__(self, drift_threshold: float = 1.3651):
        print("🚀 Initializing FailGuard Supervisor...")

        # STRICT: Only load the real protected taxonomy
        base_dir = os.path.dirname(os.path.abspath(__file__))
        real_path = os.path.join(base_dir, "../../config/taxonomy_config.yaml.real")

        if not os.path.exists(real_path):
            print("❌ ERROR: Real taxonomy file not found!")
            print(f"   Expected: {real_path}")
            raise FileNotFoundError("taxonomy_config.yaml.real is missing")

        print("✅ Loaded protected real taxonomy (full version)")

        with open(real_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract all failure modes from categories -> sub_modes
        self.taxonomy = []
        if isinstance(self.config, dict) and 'categories' in self.config:
            for category in self.config['categories']:
                if isinstance(category, dict) and 'sub_modes' in category:
                    self.taxonomy.extend(category['sub_modes'])

        print(f"✅ Loaded FailGuard Taxonomy v1.0")
        print(f"📊 Loaded {len(self.taxonomy)} failure modes across categories")

        if len(self.taxonomy) == 0:
            print("❌ ERROR: No failure modes found in real taxonomy file.")
            raise ValueError("Taxonomy file appears to be empty or wrong structure.")

        self.drift_threshold = drift_threshold

        print("🔧 Loading upgraded local embedding model (all-mpnet-base-v2)...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        self._build_failure_index()

        print("✅ FailGuard Supervisor ready for real-time prevention")

    def _build_failure_index(self):
        texts = []
        for mode in self.taxonomy:
            if isinstance(mode, dict):
                text = mode.get("description") or mode.get("name") or str(mode)
            else:
                text = str(mode)
            texts.append(text)

        self.embeddings = self.embedding_model.encode(texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
        self.failure_texts = texts
        print(f"   Built index with {len(texts)} failure modes")

    def evaluate_step(self, proposed_action: str, context: str = "") -> Dict[str, Any]:
        full_text = f"Context: {context}\nAction: {proposed_action}"
        query_embedding = self.embedding_model.encode([full_text])

        distances, indices = self.index.search(query_embedding.astype(np.float32), 1)
        distance = float(distances[0][0])
        closest_idx = int(indices[0][0])

        closest_mode = self.taxonomy[closest_idx]

        is_high_risk = distance < self.drift_threshold

        if is_high_risk:
            recommendation = "⚠️ HIGH RISK - Intervene immediately"
            status = "INTERVENE"
        else:
            recommendation = "✅ Safe to proceed"
            status = "SAFE"

        mode_name = closest_mode.get("name", "Unknown") if isinstance(closest_mode, dict) else str(closest_mode)

        return {
            "status": status,
            "recommendation": recommendation,
            "closest_failure": mode_name,
            "distance": round(distance, 4),
            "context_used": context[:150] + "..." if len(context) > 150 else context
        }