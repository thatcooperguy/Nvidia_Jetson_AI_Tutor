# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in EdgeTutor AI, please report it
responsibly.

### How to Report

1. **Email**: Send a description to the repository maintainer via GitHub
   (open a private security advisory on this repository).
2. **Do not** open a public GitHub issue for security vulnerabilities.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix/patch**: Depends on severity, typically within 2 weeks for critical issues

## Scope

### In Scope
- Safety filter bypasses (kid-safe guardrails)
- Prompt injection that could produce harmful content
- Local file access vulnerabilities
- Authentication/authorization issues (if added)
- Data leakage from the device

### Out of Scope
- Vulnerabilities in upstream dependencies (report to those projects)
- Physical access attacks (the device owner has full control)
- Denial of service on a local device
- Issues only reproducible with custom/modified models

## Design Principles

EdgeTutor AI is designed with these security principles:

1. **Offline-first**: No data leaves the device. No telemetry. No cloud calls.
2. **Local-only**: The web UI is served on localhost/LAN only. No public endpoints.
3. **Kid-safe**: Content filtering is applied at input and output layers.
4. **No secrets in code**: Configuration via `.env` file (gitignored).
5. **Minimal attack surface**: No user accounts, no databases, no external APIs.

## Known Limitations

- The safety filter uses keyword/regex matching and is not comprehensive.
  Determined users may find ways to bypass it. The LLM's own training also
  provides a layer of safety.
- The LLM may occasionally produce inaccurate information. It is a tutor
  assistant, not a definitive source of truth.
- The system is designed for trusted local network use. Do not expose it
  to the public internet without additional security measures.
