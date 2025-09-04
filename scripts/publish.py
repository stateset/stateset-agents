#!/usr/bin/env python3
"""
Publishing script for StateSet Agents framework.

This script handles the complete publishing workflow including:
- Version bumping
- Package building
- PyPI publishing
- GitHub release creation
- Docker image publishing
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import re
from datetime import datetime
import argparse


class Publisher:
    """Handles framework publishing operations."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.version_file = self.project_root / "stateset_agents" / "__init__.py"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        
    def get_current_version(self) -> str:
        """Get current version from __init__.py."""
        with open(self.version_file, 'r') as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        raise ValueError("Could not find version in __init__.py")
    
    def bump_version(self, new_version: str) -> None:
        """Bump version in all relevant files."""
        print(f"Bumping version to {new_version}")
        
        # Update __init__.py
        with open(self.version_file, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'__version__\s*=\s*["\']([^"\']+)["\']',
            f'__version__ = "{new_version}"',
            content
        )
        
        with open(self.version_file, 'w') as f:
            f.write(content)
        
        # Update pyproject.toml
        with open(self.pyproject_file, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'version\s*=\s*["\']([^"\']+)["\']',
            f'version = "{new_version}"',
            content
        )
        
        with open(self.pyproject_file, 'w') as f:
            f.write(content)
        
        print(f"Version bumped to {new_version}")
    
    def build_package(self) -> bool:
        """Build the package."""
        print("Building package...")
        
        try:
            # Clean previous builds
            subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
            subprocess.run([sys.executable, "-m", "build"], check=True, cwd=self.project_root)
            print("Package built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to build package: {e}")
            return False
    
    def test_package(self) -> bool:
        """Test the built package."""
        print("Testing package...")
        
        try:
            # Install in test environment
            dist_dir = self.project_root / "dist"
            wheel_files = list(dist_dir.glob("*.whl"))
            
            if not wheel_files:
                print("No wheel file found")
                return False
            
            latest_wheel = max(wheel_files, key=lambda x: x.stat().st_mtime)
            
            # Test installation
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--force-reinstall", str(latest_wheel)
            ], check=True)
            
            # Test import
            test_script = """
import stateset_agents
print(f"Version: {stateset_agents.__version__}")
print("Import test successful")
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Package test successful")
                print(result.stdout)
                return True
            else:
                print(f"Package test failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Package test failed: {e}")
            return False
    
    def publish_to_pypi(self, test: bool = True) -> bool:
        """Publish to PyPI or TestPyPI."""
        print(f"Publishing to {'TestPyPI' if test else 'PyPI'}...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "twine"], check=True)
            
            cmd = [sys.executable, "-m", "twine", "upload"]
            if test:
                cmd.extend(["--repository", "testpypi"])
            
            # Check for API token
            token_var = "TEST_PYPI_API_TOKEN" if test else "PYPI_API_TOKEN"
            if token_var not in os.environ:
                print(f"Warning: {token_var} not found in environment")
                print("Please set your PyPI API token or twine will prompt for credentials")
            
            cmd.extend(["dist/*"])
            subprocess.run(cmd, check=True, cwd=self.project_root)
            
            print(f"Successfully published to {'TestPyPI' if test else 'PyPI'}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to publish: {e}")
            return False
    
    def create_github_release(self, version: str, notes: str) -> bool:
        """Create GitHub release."""
        print(f"Creating GitHub release v{version}...")
        
        try:
            # This would typically use GitHub CLI or API
            # For now, we'll prepare the release information
            release_info = {
                "tag_name": f"v{version}",
                "name": f"Release v{version}",
                "body": notes,
                "draft": False,
                "prerelease": "rc" in version or "beta" in version or "alpha" in version
            }
            
            print(f"Release info prepared: {json.dumps(release_info, indent=2)}")
            print("Note: Manual GitHub release creation required")
            print("Use: gh release create v{version} --title 'Release v{version}' --notes '{notes}'")
            
            return True
            
        except Exception as e:
            print(f"Failed to create GitHub release: {e}")
            return False
    
    def update_changelog(self, version: str, changes: str) -> None:
        """Update CHANGELOG.md with new release."""
        print(f"Updating changelog for version {version}")
        
        changelog_entry = f"""## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

{changes}

"""
        
        if self.changelog_file.exists():
            with open(self.changelog_file, 'r') as f:
                content = f.read()
        else:
            content = "# Changelog\n\nAll notable changes will be documented in this file.\n\n"
        
        # Insert after header
        lines = content.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('## ['):
                insert_index = i
                break
        
        lines.insert(insert_index, changelog_entry.strip())
        
        with open(self.changelog_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print("Changelog updated")
    
    def run_full_publish(
        self, 
        new_version: Optional[str] = None,
        test_pypi: bool = True,
        create_release: bool = True,
        changes: str = ""
    ) -> bool:
        """Run the complete publishing workflow."""
        print("🚀 Starting StateSet Agents publishing workflow...")
        
        try:
            # Get or validate version
            if new_version:
                self.bump_version(new_version)
                version = new_version
            else:
                version = self.get_current_version()
            
            print(f"📦 Publishing version {version}")
            
            # Update changelog
            if changes:
                self.update_changelog(version, changes)
            
            # Build package
            if not self.build_package():
                return False
            
            # Test package
            if not self.test_package():
                return False
            
            # Publish to PyPI/TestPyPI
            if not self.publish_to_pypi(test=test_pypi):
                return False
            
            # Create GitHub release
            if create_release and changes:
                self.create_github_release(version, changes)
            
            print("✅ Publishing workflow completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Publishing failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Publish StateSet Agents framework")
    parser.add_argument("--version", help="New version to publish")
    parser.add_argument("--test", action="store_true", help="Publish to TestPyPI")
    parser.add_argument("--production", action="store_true", help="Publish to production PyPI")
    parser.add_argument("--skip-release", action="store_true", help="Skip GitHub release creation")
    parser.add_argument("--changes", help="Release notes/changes")
    
    args = parser.parse_args()
    
    publisher = Publisher()
    
    # Determine PyPI target
    test_pypi = not args.production and not args.test
    
    success = publisher.run_full_publish(
        new_version=args.version,
        test_pypi=test_pypi,
        create_release=not args.skip_release,
        changes=args.changes or "Release notes to be added..."
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
