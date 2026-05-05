import os

def create_project_structure():
    folders = [
        "src/core",
        "src/supervisor",
        "src/utils",
        "config",
        "data/sample_traces",
        "examples",
        "tests",
        "notebooks",
        "logs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    # Create empty placeholder files
    placeholders = [
        "src/core/__init__.py",
        "src/supervisor/__init__.py",
        "src/utils/__init__.py",
        "config/taxonomy_config.yaml",
        "examples/simple_agent_protection.py",
        "README.md"  # just in case
    ]
    
    for file in placeholders:
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write("# Placeholder file\n")
            print(f"✅ Created placeholder: {file}")

if __name__ == "__main__":
    print("🚀 Setting up FailGuard project structure...\n")
    create_project_structure()
    print("\n✅ Project structure setup complete!")