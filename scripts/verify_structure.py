import os
from pathlib import Path

def check_structure():
    """Verify project structure and files"""
    required_files = [
        "pyproject.toml",
        "setup.py",
        "agent_framework/__init__.py",
        "agent_framework/agent.py",
        "agent_framework/models.py",
        "agent_framework/config.py",
        "agent_framework/example_agent.py",
        "agent_framework/llm/__init__.py",
        "agent_framework/llm/base.py",
        "agent_framework/llm/models.py",
        "agent_framework/llm/openai_provider.py",
        "examples/__init__.py",
        "examples/run_simple_agent.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("All required files present")
    return True

if __name__ == "__main__":
    check_structure() 