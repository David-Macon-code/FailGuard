"""
FailGuard Multidimensional Mapper (Upgraded Local Model)
Now using all-mpnet-base-v2 (768 dimensions) for much better reliability.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

from src.core.taxonomy import FailGuardTaxonomy

class MultidimensionalMapper:
    def __init__(self):
        # Upgraded to a stronger local model for better reliability testing
        print("🔧 Loading upgraded local embedding model (all-mpnet-base-v2)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')   # 768 dimensions - much stronger
        self.taxonomy = FailGuardTaxonomy()
        self.dimension = 768
        self.index = None
        self.failure_vectors = []
        self.failure_metadata = []
        
        self._build_failure_index()

    def _build_failure_index(self):
        print("🔧 Building failure mode index in high-dimensional space...")
        all_modes = self.taxonomy.list_all_modes()
        texts = []
        
        for mode in all_modes:
            text = f"{mode['category']}: {mode['name']} - {mode.get('description', '')} - {mode.get('enterprise_impact', '')}"
            texts.append(text)
            self.failure_metadata.append(mode)
        
        vectors = self.model.encode(texts, convert_to_numpy=True)
        self.failure_vectors = vectors.astype('float32')
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.failure_vectors)
        
        print(f"✅ Indexed {len(texts)} failure modes in {self.dimension}-dimensional space (upgraded model)")

    def embed_trajectory(self, trajectory_text: str) -> np.ndarray:
        vector = self.model.encode([trajectory_text], convert_to_numpy=True)
        return vector.astype('float32')

    def find_nearest_failures(self, trajectory_text: str, k: int = 3) -> list:
        query_vector = self.embed_trajectory(trajectory_text)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            mode = self.failure_metadata[idx]
            results.append({
                "category": mode["category"],
                "name": mode["name"],
                "distance": float(distances[0][i]),
                "enterprise_impact": mode.get("enterprise_impact")
            })
        return results

    def detect_drift(self, trajectory_text: str, threshold: float = 0.42) -> dict:
        nearest = self.find_nearest_failures(trajectory_text, k=1)[0]
        is_drifting = nearest["distance"] < threshold
        return {
            "is_drifting": is_drifting,
            "closest_failure": nearest["name"],
            "category": nearest["category"],
            "distance": nearest["distance"],
            "recommendation": "INTERVENE" if is_drifting else "OK"
        }


# Quick test
if __name__ == "__main__":
    mapper = MultidimensionalMapper()
    
    test_trajectory = "The agent confidently cited a regulation that was repealed last year and proceeded with the action anyway."
    results = mapper.find_nearest_failures(test_trajectory)
    print("\n🔍 Test trajectory nearest failures:")
    for r in results:
        print(f"   • {r['category']} → {r['name']} (distance: {r['distance']:.4f})")
    
    drift = mapper.detect_drift(test_trajectory)
    print(f"\n🚨 Drift detected: {drift['is_drifting']} → {drift['recommendation']}")