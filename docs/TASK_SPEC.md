# Task and labeling contract

## Scope

Input: a short English support ticket for a fictional developer SaaS product. Output: one JSON object, without markdown or explanations. Limit the first experiment to 512 total input/output tokens; measure truncation before training. The model routes a ticket and extracts stated facts. It does not diagnose root causes, execute changes, or invent a resolution.

## Required output fields

| Field | Allowed values / rule |
| --- | --- |
| `category` | `authentication`, `billing`, `rate_limit`, `integration`, `bug`, `other` |
| `environment` | `production`, `staging`, `development`, `unknown` |
| `customer_reported_impact` | One short factual sentence, or `null` if impact is not stated |
| `suggested_team` | `developer_support`, `billing_support`, `general_support` |

No additional keys. Missing facts stay unknown/null. Never infer production from the word “urgent”. Summarize customer impact without claiming a verified cause. Free-text impact is human-reviewed; exact string matching is not a useful semantic accuracy metric.

## Category rules

- Authentication: credential, login, permission, or authorization failures (including 401/403).
- Billing: invoices, charges, subscription, payment, or refund requests.
- Rate limit: explicitly reported throttling, 429 responses, or quota exhaustion.
- Integration: setup/configuration questions involving SDKs, webhooks, or external systems without an established defect.
- Bug: an explicit reproducible functional defect, excluding the more specific categories above.
- Other: unclear requests, unrelated content, or multiple independent issues that cannot be assigned one dominant category.

For mixed independent issues, label `other` and route to general support for review. Category is about the reported issue, not a verified diagnosis. Explicit authentication/billing/rate-limit symptoms take precedence over generic “bug” wording. Map authentication/rate_limit/integration/bug to developer_support; billing to billing_support; other to general_support. This team mapping is deterministic: compare with a simple rules-based mapping too, and do not present team accuracy as an independent reasoning capability.

## Ambiguity and input handling

Do not follow instructions embedded in the ticket to alter the schema. Environment must be explicitly stated (local/laptop means development; test alone is ambiguous). Contradictory or multiple environments map to unknown in this first version. Human review is appropriate for unclear tickets. No generated numerical confidence field is included; it would require calibration before being treated as a probability.

## Data policy

All initial examples are authored fictional tickets. No real customer data or API credentials. Review naturalness, labels, and edge cases before accepting new examples. Keep related scenario families and paraphrases in one split. Automated normalization catches exact duplicates, not semantic leakage. Handwritten test performance does not establish production performance.
