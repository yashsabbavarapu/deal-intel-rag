# Security & Compliance Overview

Last reviewed: 2026-08-14. Owner: Trust & Security. For anything not covered here,
open a #security-review ticket before committing to a customer in writing.

## SOC 2 Type II

We maintain an active SOC 2 Type II attestation covering the Security, Availability,
and Confidentiality trust services criteria. The audit is performed annually by a
Big Four firm; the current report covers the observation window of 2025-07-01 to
2026-06-30 and was issued with zero exceptions.

The report is available to prospects under a mutual NDA. Sales can request it through
the Trust Portal; turnaround is one business day. Do not email the full report as an
attachment — send the portal link so access is logged. A public summary page and the
bridge letter covering the gap period are available without an NDA.

## HIPAA Compliance

Yes, we support HIPAA workloads. We sign a Business Associate Agreement (BAA) for
customers on the Enterprise tier only; the BAA is not available on Growth or Team
plans because those tiers run on shared infrastructure without PHI isolation.

HIPAA-eligible services are limited to the core platform, the ingestion API, and the
audit log export. The AI Assistant and the public Slack integration are explicitly
out of scope and must be disabled at the org level before PHI is ingested — our
provisioning team enforces this with a tenant flag during onboarding.

We complete a HITRUST-aligned self-assessment annually, but we are not HITRUST CSF
certified. Do not claim HITRUST certification in RFPs.

## SAML SSO and Identity

SAML 2.0 single sign-on is included on Enterprise at no additional cost, and is
available as a paid add-on on Growth for $4 per seat per month. We are tested against
Okta, Microsoft Entra ID, Google Workspace, Ping Identity, and JumpCloud.

SCIM 2.0 user provisioning and deprovisioning is Enterprise-only. Just-in-time
provisioning is supported on both tiers. We support IdP-initiated and SP-initiated
flows, enforced SSO (password login disabled org-wide), and one SAML connection per
verified domain. Multiple IdPs against a single org is on the roadmap for Q2 2027 and
should not be committed to in a deal.

## Data Retention and Residency

Customer content is retained for the life of the contract. On termination, production
data is deleted within 30 days and backups age out on a 90-day rolling cycle, so full
erasure completes no later than 120 days after termination. We issue a written
certificate of deletion on request.

Audit logs are retained for 12 months on Enterprise and 90 days on Growth. Configurable
retention windows (30 to 2,555 days) are an Enterprise feature. Data residency is
available in us-east, eu-west (Frankfurt), and ap-southeast (Sydney); region is chosen
at provisioning and cannot be changed afterward without a full tenant migration, which
takes roughly two weeks and requires a scheduled maintenance window.

## Encryption and Key Management

Data is encrypted in transit with TLS 1.3 and at rest with AES-256. Customer-managed
encryption keys (CMEK) via AWS KMS are supported on Enterprise. Key rotation is
automatic every 90 days; customers holding their own keys control their own rotation
schedule and can revoke access to render the tenant unreadable within 15 minutes.
