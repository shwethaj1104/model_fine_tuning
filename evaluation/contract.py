"""Shared task contract; no third-party dependencies."""

CATEGORIES = ("authentication", "billing", "rate_limit", "integration", "bug", "other")
ENVIRONMENTS = ("production", "staging", "development", "unknown")
TEAMS = ("developer_support", "billing_support", "general_support")
FIELDS = {"category", "environment", "customer_reported_impact", "suggested_team"}


def output_errors(value):
    if not isinstance(value, dict):
        return ["output must be an object"]
    errors = []
    if set(value) != FIELDS:
        errors.append("output keys must match the contract exactly")
    for field, allowed in (("category", CATEGORIES), ("environment", ENVIRONMENTS), ("suggested_team", TEAMS)):
        if value.get(field) not in allowed:
            errors.append(f"invalid {field}")
    impact = value.get("customer_reported_impact")
    if impact is not None and (not isinstance(impact, str) or not impact.strip()):
        errors.append("customer_reported_impact must be nonempty text or null")
    return errors


def expected_team(category):
    return {"billing": "billing_support", "other": "general_support"}.get(category, "developer_support")
