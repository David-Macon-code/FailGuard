"""
FailGuard Quick Demo - Taxonomy Loader
Run this to see your full taxonomy in action.
"""
import sys
import os
# Fix for running from any folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.taxonomy import FailGuardTaxonomy

def main():
    print("🚀 FailGuard Taxonomy Demo")
    print("=" * 50)
    
    # Load the taxonomy
    taxonomy = FailGuardTaxonomy()
    
    print(f"\n📊 Loaded {len(taxonomy.list_all_modes())} failure modes across categories")
    print(f"Categories: {taxonomy.get_categories()}")
    
    # Example searches
    print("\n🔍 Search for 'hallucination':")
    results = taxonomy.search("hallucination")
    for r in results[:3]:  # Show first 3 matches
        print(f"   • {r['category']} → {r['mode']['name']}")
    
    print("\n🔍 Search for 'bias':")
    results = taxonomy.search("bias")
    for r in results[:3]:
        print(f"   • {r['category']} → {r['mode']['name']}")
    
    # Enterprise impact examples
    print("\n💼 Enterprise Impact Examples:")
    impact_examples = ["hallucination", "distribution shift", "multi-agent failure cascades"]
    for name in impact_examples:
        impact = taxonomy.get_enterprise_impact(name)
        if impact:
            print(f"   • {name}: {impact}")
    
    # Legal notes
    print("\n⚖️  Legal Notes Example:")
    legal = taxonomy.get_legal_notes("Colorado AI Act / California ADMT exposure")
    if legal:
        print(f"   • {legal}")
    
    print("\n✅ Demo complete! The taxonomy engine is working.")
    print("   Next step: Build the multidimensional mapper and supervisor.")

if __name__ == "__main__":
    main()