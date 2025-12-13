#!/usr/bin/env python3
"""
Publishing script for StateSet Agents framework.

This script handles the publishing workflow including:
- Version bumping (explicit or semver bump)
- Package building + metadata validation
- PyPI/TestPyPI publishing
- Optional changelog + GitHub release prep
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


class Publisher:
    """Handles framework publishing operations."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.version_file = self.project_root / "stateset_agents" / "__init__.py"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        self.setup_py_file = self.project_root / "setup.py"

    def get_current_version(self) -> str:
        """Get current version from __init__.py."""
        with open(self.version_file, "r") as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        raise ValueError("Could not find version in __init__.py")

    def _parse_version_base(self, version: str) -> Tuple[int, int, int]:
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if not match:
            raise ValueError(f"Unsupported version format: {version!r}")
        major, minor, patch = match.groups()
        return int(major), int(minor), int(patch)

    def _validate_version(self, version: str) -> None:
        # Accept semver-like versions, optionally with prerelease/build suffix.
        if not re.match(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$", version):
            raise ValueError(
                f"Invalid version format: {version!r} (expected MAJOR.MINOR.PATCH)"
            )

    def _compute_bumped_version(self, current_version: str, bump: str) -> str:
        major, minor, patch = self._parse_version_base(current_version)
        if bump == "patch":
            patch += 1
        elif bump == "minor":
            minor += 1
            patch = 0
        elif bump == "major":
            major += 1
            minor = 0
            patch = 0
        else:
            raise ValueError(f"Unknown bump type: {bump!r}")
        return f"{major}.{minor}.{patch}"

    def resolve_version(self, requested: str) -> str:
        """Resolve a requested version or bump spec to a concrete version string."""
        requested = requested.strip()
        if requested in {"patch", "minor", "major"}:
            return self._compute_bumped_version(self.get_current_version(), requested)
        self._validate_version(requested)
        return requested

    def _replace_project_version_in_pyproject(
        self, pyproject_content: str, new_version: str
    ) -> str:
        lines = pyproject_content.splitlines(keepends=True)
        in_project = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_project = stripped == "[project]"
                continue
            if not in_project:
                continue
            if re.match(r'^version\s*=\s*["\']', stripped):
                newline = "\n" if line.endswith("\n") else ""
                lines[idx] = re.sub(
                    r'^(\s*version\s*=\s*)["\'][^"\']+["\'](\s*)$',
                    rf'\1"{new_version}"\2',
                    line.rstrip("\n"),
                ) + newline
                return "".join(lines)
        raise ValueError("Could not find [project].version in pyproject.toml")

    def _replace_version_in_setup_py(self, setup_py_content: str, new_version: str) -> str:
        # Replace the first `version="x.y.z"` occurrence (expected in setup()).
        new_content, count = re.subn(
            r'(\bversion\s*=\s*["\'])([^"\']+)(["\'])',
            rf'\g<1>{new_version}\3',
            setup_py_content,
            count=1,
        )
        if count == 0:
            raise ValueError("Could not find version=... in setup.py")
        return new_content

    def bump_version(self, new_version: str) -> None:
        """Bump version in all relevant files."""
        self._validate_version(new_version)
        print(f"Bumping version to {new_version}")

        # Update __init__.py
        with open(self.version_file, "r") as f:
            content = f.read()

        content = re.sub(
            r'__version__\s*=\s*["\']([^"\']+)["\']',
            f'__version__ = "{new_version}"',
            content,
        )

        with open(self.version_file, "w") as f:
            f.write(content)

        # Update pyproject.toml
        with open(self.pyproject_file, "r") as f:
            pyproject_content = f.read()

        pyproject_content = self._replace_project_version_in_pyproject(
            pyproject_content, new_version
        )

        with open(self.pyproject_file, "w") as f:
            f.write(pyproject_content)

        # Update setup.py (if present)
        if self.setup_py_file.exists():
            with open(self.setup_py_file, "r", encoding="utf-8") as f:
                setup_py_content = f.read()
            setup_py_content = self._replace_version_in_setup_py(
                setup_py_content, new_version
            )
            with open(self.setup_py_file, "w", encoding="utf-8") as f:
                f.write(setup_py_content)

        print(f"Version bumped to {new_version}")

    def _clean_build_artifacts(self) -> None:
        for path in [
            self.project_root / "dist",
            self.project_root / "build",
            self.project_root / "stateset_agents.egg-info",
        ]:
            shutil.rmtree(path, ignore_errors=True)

    def build_package(self) -> bool:
        """Build the package."""
        print("Building package...")

        try:
            build_commands = [
                [sys.executable, "-m", "build", "--no-isolation"],
                [sys.executable, "-m", "build"],
            ]
            if self.setup_py_file.exists():
                build_commands.append(
                    [sys.executable, str(self.setup_py_file), "sdist", "bdist_wheel"]
                )

            last_error: Optional[subprocess.CalledProcessError] = None
            for cmd in build_commands:
                self._clean_build_artifacts()
                try:
                    subprocess.run(cmd, check=True, cwd=self.project_root)
                    break
                except subprocess.CalledProcessError as e:
                    last_error = e
            else:
                raise last_error or subprocess.CalledProcessError(
                    returncode=1, cmd=build_commands[-1]
                )

            subprocess.run(
                [sys.executable, "-m", "twine", "check", "dist/*"],
                check=True,
                cwd=self.project_root,
            )
            print("Package built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to build package: {e}")
            print("Ensure build + twine are installed: pip install build twine")
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
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    str(latest_wheel),
                ],
                check=True,
            )

            # Test import
            test_script = """
import stateset_agents
print(f"Version: {stateset_agents.__version__}")
print("Import test successful")
"""

            result = subprocess.run(
                [sys.executable, "-c", test_script], capture_output=True, text=True
            )

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

    def _dist_files_for_version(self, version: str) -> Iterable[Path]:
        dist_dir = self.project_root / "dist"
        candidates = sorted(dist_dir.glob(f"stateset_agents-{version}*"))
        if candidates:
            return candidates
        return sorted(dist_dir.glob(f"*{version}*"))

    def publish_to_pypi(
        self, version: str, test: bool = True, skip_existing: bool = False
    ) -> bool:
        """Publish to PyPI or TestPyPI."""
        print(f"Publishing to {'TestPyPI' if test else 'PyPI'}...")

        try:
            files = list(self._dist_files_for_version(version))
            if not files:
                print(f"No dist artifacts found for version {version} in dist/")
                return False

            cmd = [sys.executable, "-m", "twine", "upload"]
            if test:
                cmd.extend(["--repository-url", "https://test.pypi.org/legacy/"])
            if skip_existing:
                cmd.append("--skip-existing")

            env = os.environ.copy()
            token_var = "TEST_PYPI_API_TOKEN" if test else "PYPI_API_TOKEN"
            token = os.environ.get(token_var)
            if not token and not env.get("TWINE_PASSWORD"):
                print(f"Warning: {token_var} not found in environment")
                print(
                    "Please set your PyPI API token or twine will prompt for credentials"
                )
            if token:
                env.setdefault("TWINE_USERNAME", "__token__")
                env.setdefault("TWINE_PASSWORD", token)

            cmd.extend([str(path) for path in files])
            subprocess.run(cmd, check=True, cwd=self.project_root, env=env)

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
                "prerelease": "rc" in version
                or "beta" in version
                or "alpha" in version,
            }

            print(f"Release info prepared: {json.dumps(release_info, indent=2)}")
            print("Note: Manual GitHub release creation required")
            print(
                "Use: gh release create v{version} --title 'Release v{version}' --notes '{notes}'"
            )

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
            with open(self.changelog_file, "r") as f:
                content = f.read()
        else:
            content = "# Changelog\n\nAll notable changes will be documented in this file.\n\n"

        # Insert after header
        lines = content.split("\n")
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("## ["):
                insert_index = i
                break

        lines.insert(insert_index, changelog_entry.strip())

        with open(self.changelog_file, "w") as f:
            f.write("\n".join(lines))

        print("Changelog updated")

    def run_full_publish(
        self,
        new_version: Optional[str] = None,
        test_pypi: bool = True,
        create_release: bool = True,
        changes: str = "",
        skip_existing: bool = False,
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
            if not self.publish_to_pypi(
                version=version, test=test_pypi, skip_existing=skip_existing
            ):
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
    parser.add_argument(
        "--version",
        help="Version to publish (e.g. 1.2.3) or bump type (patch|minor|major)",
    )
    parser.add_argument("--test", action="store_true", help="Publish to TestPyPI")
    parser.add_argument(
        "--production", action="store_true", help="Publish to production PyPI"
    )
    parser.add_argument(
        "--skip-release", action="store_true", help="Skip GitHub release creation"
    )
    parser.add_argument("--changes", help="Release notes/changes")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already-uploaded distributions (idempotent uploads)",
    )

    args = parser.parse_args()

    publisher = Publisher()

    # Determine PyPI target
    test_pypi = not args.production

    resolved_version: Optional[str] = None
    if args.version:
        resolved_version = publisher.resolve_version(args.version)

    success = publisher.run_full_publish(
        new_version=resolved_version,
        test_pypi=test_pypi,
        create_release=not args.skip_release,
        changes=args.changes or "",
        skip_existing=args.skip_existing,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
