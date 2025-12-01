# StateSet Agents Governance

This document describes the governance model for the StateSet Agents project.

## Project Structure

### Roles

#### Maintainers
Maintainers have full commit access and are responsible for:
- Reviewing and merging pull requests
- Managing releases
- Setting project direction
- Enforcing code of conduct

Current maintainers:
- StateSet Core Team (@stateset/core)

#### Contributors
Anyone who contributes code, documentation, or other improvements to the project. Contributors are recognized in release notes and the contributors file.

#### Community Members
Anyone who uses StateSet Agents, reports issues, or participates in discussions.

---

## Decision Making

### Technical Decisions

**Minor decisions** (bug fixes, small features):
- Can be merged by any maintainer after one approval
- Should follow existing patterns and conventions

**Moderate decisions** (new features, refactoring):
- Require review from at least one maintainer
- Should have associated issue or discussion
- May require RFC for complex changes

**Major decisions** (architecture changes, breaking changes):
- Require RFC (Request for Comments) document
- Need consensus among maintainers
- Should have community feedback period (minimum 1 week)

### RFC Process

For significant changes:

1. **Create RFC**: Open an issue with `[RFC]` prefix describing:
   - Motivation and use cases
   - Proposed solution
   - Alternatives considered
   - Migration path (if breaking)

2. **Discussion**: Community feedback period (1-2 weeks)

3. **Decision**: Maintainers review and decide:
   - Accept: Proceed with implementation
   - Revise: Request changes to proposal
   - Reject: Close with explanation

4. **Implementation**: Follow normal PR process

---

## Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking API changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Schedule
- **Patch releases**: As needed for bug fixes
- **Minor releases**: Approximately monthly
- **Major releases**: When significant breaking changes accumulate

### Release Checklist
1. Update CHANGELOG.md
2. Update version in pyproject.toml
3. Run full test suite
4. Create release notes
5. Tag release in git
6. Publish to PyPI
7. Announce in appropriate channels

---

## Code of Conduct

All participants must follow our [Code of Conduct](../CODE_OF_CONDUCT.md).

### Enforcement
1. **Warning**: First-time minor violations
2. **Temporary ban**: Repeated or moderate violations
3. **Permanent ban**: Severe violations or repeated moderate violations

Reports should be sent to: conduct@stateset.io

---

## Communication Channels

### Official Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Q&A, general discussion
- **Discord**: Real-time community chat (link in README)

### Meeting Schedule
- **Maintainer sync**: Weekly (internal)
- **Community call**: Monthly (open to all)

---

## Intellectual Property

### Licensing
StateSet Agents is licensed under the Business Source License 1.1:
- Free for non-production use
- Converts to Apache 2.0 after September 2029
- Commercial licenses available

### Contributions
By contributing, you agree that:
- Your contributions are your original work
- You have the right to submit them
- They are licensed under the project's license

### CLA
Contributors may be asked to sign a Contributor License Agreement (CLA) for significant contributions.

---

## Amendments

This governance document may be amended through the RFC process. Changes require:
- RFC with proposed changes
- 2-week community feedback period
- Consensus among maintainers

---

*Last updated: December 2024*
