"""Published constants of the TraceForge specification."""
from __future__ import annotations

from fractions import Fraction

SCHEMA_VERSION = "1.0"

# Canonical issue types (finding["type"] is normalised with strip().lower()).
DUPLICATE, ALLOCATION, FX, CUTOFF = "duplicate", "allocation", "fx", "cutoff"
ISSUE_TYPES = (DUPLICATE, ALLOCATION, FX, CUTOFF)
SCENARIOS = ("cross_border_partial", "period_end_duplicate")

# Environment limits
MAX_STEPS = 100            # including submit
MAX_PAGE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_AFFECTED = 20
MAX_EXPLANATION = 1000
MAX_ID_LEN = 64
MAX_ISSUES = 12

# Scoring
AMOUNT_TOLERANCE_MINOR = 1          # |claimed - expected| <= 1 minor unit
STEP_COST = Fraction(1, 500)        # 0.002 per inspection step
INVALID_REWARD = -1.0

TABLES = ("ledger", "invoices", "payments", "fx")
ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")

# Chart of accounts
ACC_CASH, ACC_AR, ACC_UNAPPLIED, ACC_REVENUE = "1000", "1200", "2300", "4000"
ACCOUNTS = {
    ACC_CASH: "Cash",
    ACC_AR: "Accounts receivable",
    ACC_UNAPPLIED: "Unapplied customer receipts",
    ACC_REVENUE: "Revenue",
}

LEDGER_FIELDS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
                 "currency", "entity_id", "invoice_id", "payment_id")
FX_FIELDS = ("date", "source_currency", "target_currency", "rate")

# ISO-4217 minor-unit exponents (rounding_rules in manifest)
CURRENCY_DECIMALS = {"EUR": 2, "USD": 2, "GBP": 2, "CHF": 2, "SEK": 2, "CAD": 2,
                     "JPY": 0, "KWD": 3}

PUBLIC_FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                "fx.csv", "policy.json")
PRIVATE_FILES = ("truth.json", "provenance.json", "clean_baseline.json")
