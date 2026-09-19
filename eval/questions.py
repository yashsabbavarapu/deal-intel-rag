"""Labelled question sets. Edit these when the corpus changes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    question: str
    category: str  # in-domain | false-premise | near-domain | out-of-domain
    section: str | None = None  # expected heading, for retrieval scoring
    expect_refusal: bool = False
    note: str = ""


# Questions the corpus genuinely answers. `section` is the heading that should
# reach the model's context; retrieval is scored on top-k recall of it.
IN_DOMAIN: list[Case] = [
    Case("Do we support HIPAA compliance?", "in-domain", "HIPAA Compliance"),
    Case("What is the Enterprise platform fee?", "in-domain", "Platform Fee", note="$24,000/yr"),
    Case("What are Competitor X's weaknesses?", "in-domain", "Competitor X Weaknesses"),
    Case("What are Competitor X's strengths?", "in-domain", "Competitor X Strengths"),
    Case("How do we displace Competitor X?", "in-domain", "Displacement Tactics"),
    Case("Where is Competitor X vulnerable in a security review?", "in-domain", "Competitor X Weaknesses"),
    Case("Can we offer a migration credit?", "in-domain", "Displacement Tactics"),
    Case("What discount can a VP of Sales approve?", "in-domain", "Discount Authority", note="30%"),
    Case("What is the minimum contract value?", "in-domain", "Minimum Commitments"),
    Case("What is our SOC 2 status?", "in-domain", "SOC 2 Type II"),
    Case("Is SAML SSO included on Growth?", "in-domain", "SAML SSO and Identity"),
    Case("How long until customer data is fully erased?", "in-domain", "Data Retention and Residency", note="120 days"),
    Case("What encryption do we use at rest?", "in-domain", "Encryption and Key Management"),
    Case("What ROI do our enterprise customers report?", "in-domain", "Aggregate ROI Metrics"),
    Case("Do we have a healthcare reference customer?", "in-domain", "Meridian Health — HIPAA Deployment"),
]

# The corpus contradicts or does not cover the premise. Answering from the docs
# is correct where they contradict it; refusing is correct where they are silent.
FALSE_PREMISE: list[Case] = [
    Case("Are we HITRUST CSF certified?", "false-premise", "HIPAA Compliance",
         expect_refusal=False, note="docs say explicitly NOT certified"),
    Case("Can we sign a BAA on the Growth plan?", "false-premise", "HIPAA Compliance",
         expect_refusal=False, note="docs say Enterprise only"),
    Case("What is our FedRAMP authorization status?", "false-premise",
         expect_refusal=True, note="absent from corpus"),
    Case("What is the discount on a 5-year term?", "false-premise",
         expect_refusal=True, note="only 12/24/36-month terms exist"),
]

# Plausible internal questions this corpus does not cover. These are the hard
# ones: they sit close enough to pass the distance gate.
NEAR_DOMAIN: list[Case] = [
    Case("What is our parental leave policy?", "near-domain", expect_refusal=True),
    Case("What is our employee stock option vesting schedule?", "near-domain", expect_refusal=True),
    Case("How many engineers are on the platform team?", "near-domain", expect_refusal=True),
]

OUT_OF_DOMAIN: list[Case] = [
    Case("What is the weather in Paris?", "out-of-domain", expect_refusal=True),
    Case("How do I bake sourdough bread?", "out-of-domain", expect_refusal=True),
    Case("What is the capital of Mongolia?", "out-of-domain", expect_refusal=True),
    Case("Who won the World Cup in 2022?", "out-of-domain", expect_refusal=True),
    Case("How do I fix a flat bicycle tire?", "out-of-domain", expect_refusal=True),
]

ALL: list[Case] = [*IN_DOMAIN, *FALSE_PREMISE, *NEAR_DOMAIN, *OUT_OF_DOMAIN]
REFUSAL_SET: list[Case] = [*NEAR_DOMAIN, *OUT_OF_DOMAIN]
