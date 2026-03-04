# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Do NOT open a public issue for security vulnerabilities.**

Instead, please report security vulnerabilities by emailing:

📧 **soham.dutta.devops@gmail.com**

Include the following information:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if you have one)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days

### Security Considerations

When using Cognitive Memory, be aware of:

#### Data Storage

- Memories may contain sensitive user data
- Ensure proper access controls on storage backends
- Use encryption at rest for production deployments

#### API Keys

- Never commit API keys (OpenAI, etc.) to version control
- Use environment variables or secret management
- Rotate keys regularly

#### LLM Interactions

- Consolidation uses LLM calls which may send data externally
- Review your LLM provider's data handling policies
- Consider self-hosted models for sensitive data

#### Network Security

- Use TLS for all database connections
- Restrict network access to storage backends
- Use authentication for the REST API

### Security Best Practices

```python
# Use environment variables for secrets
import os

config = MemoryConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    database_url=os.environ["DATABASE_URL"],
)

# Enable encryption for sensitive deployments
config = MemoryConfig(
    encryption_at_rest=True,
    encryption_key=os.environ["ENCRYPTION_KEY"],
)
```

### Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with their permission).

## Security Updates

Security updates will be released as patch versions and announced via:

- GitHub Security Advisories
- Release notes
- Discussions announcements

Subscribe to releases to stay informed:

```bash
gh repo subscribe NP-compete/cognitive-memory --releases
```
