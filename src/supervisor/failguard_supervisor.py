"""
FailGuard Supervisor - Protected Version
Loads the real hidden taxonomy (taxonomy_config.yaml.real) first, then falls back to sample.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, List, Any

class FailGuardSupervisor:
    def __init__(self, drift_threshold: float = 1.3651):
        print("🚀 Initializing FailGuard Supervisor...")
        
        # === PROTECTED TAXONOMY LOADING ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        real_taxonomy_path = os.path.join(base_dir, "../../config/taxonomy_config.yaml.real")
        sample_taxonomy_path = os.path.join(base_dir, "../../config/taxonomy_config.yaml.sample")
        
        if os.path.exists(real_taxonomy_path):
            taxonomy_path = real_taxonomy_path
            print("✅ Loaded protected real taxonomy")
        else:
            taxonomy_path = sample_taxonomy_path
            print("⚠️  Loaded public sample taxonomy (real file not found on this machine)")
        
        # Load taxonomy
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.drift_threshold = drift_threshold
        self.taxonomy = self.config.get("failure_modes", [])
        
        print(f"✅ Loaded FailGuard Taxonomy v1.0")
        print(f"📊 Loaded {len(self.taxonomy)} failure modes across categories")
        
        # Load local embedding model
        print("🔧 Loading upgraded local embedding model (all-mpnet-base-v2)...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Build failure mode index
        self._build_failure_index()
        
        print("✅ FailGuard Supervisor ready for real-time prevention")

    def _build_failure_index(self):
        texts = [mode["description"] for mode in self.taxonomy]
        self.embeddings = self.embedding_model.encode(texts)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
        self.failure_texts = texts

    def evaluate_step(self, proposed_action: str, context: str = "") -> Dict[str, Any]:
        full_text = f"Context: {context}\nAction: {proposed_action}"
        query_embedding = self.embedding_model.encode([full_text])
        
        # Search for nearest failure mode
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
        
        return {
            "status": status,
            "recommendation": recommendation,
            "closest_failure": closest_mode["name"],
            "distance": round(distance, 4),
            "context_used": context[:150] + "..." if len(context) > 150 else context
        }

    # Optional helper methods
    def get_categories(self):
        return list(set(mode["category"] for mode in self.taxonomy))