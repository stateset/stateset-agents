#!/usr/bin/env python3
"""
Type checking utility for the StateSet Agents framework.

This script runs mypy on the codebase and provides a summary of type issues.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_mypy(paths: List[str]) -> Tuple[int, str, str]:
    """Run mypy on the given paths."""
    cmd = ["mypy"] + paths
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    """Main entry point."""
    print("🔍 Running type checks on StateSet Agents...")
    
    # Define paths to check
    paths_to_check = [
        "stateset_agents",
        "core",
        "training",
        "utils",
        "rewards",
        "api",
        # Don't check tests and examples for strict typing
        # "tests",
        # "examples"
    ]
    
    # Filter out paths that don't exist
    existing_paths = [p for p in paths_to_check if Path(p).exists()]
    
    if not existing_paths:
        print("❌ No valid paths found to check")
        return 1
    
    print(f"📁 Checking paths: {', '.join(existing_paths)}")
    
    # Run mypy
    returncode, stdout, stderr = run_mypy(existing_paths)
    
    # Print results
    if stdout:
        print("\n📋 Type check results:")
        print(stdout)
    
    if stderr:
        print("\n⚠️  Errors/Warnings:")
        print(stderr)
    
    # Summary
    if returncode == 0:
        print("\n✅ All type checks passed!")
        return 0
    else:
        print(f"\n❌ Type checks failed with return code {returncode}")
        return returncode


if __name__ == "__main__":
    sys.exit(main())
