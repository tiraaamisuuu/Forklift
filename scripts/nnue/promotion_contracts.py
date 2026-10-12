"""Versioned, predeclared promotion contracts; changes require a new ID."""

ORIGINAL_CONTRACT_ID = "shared-nnue-20260914"
RECOVERY_CONTRACT_ID = "shared-nnue-long-recovery-20261012"

PROMOTION_CONTRACT = {
    "network": {"games": 800, "timeControl": "30+0.3", "seed": 202609140,
                "threads": 1, "hashMb": 128, "concurrency": 6},
    "classical": {"games": 1200, "timeControl": "90+0.9", "seed": 202609141,
                  "threads": 1, "hashMb": 128, "concurrency": 6},
}
RECOVERY_CONTRACT = {
    "network": dict(PROMOTION_CONTRACT["network"]),
    "classical": {**PROMOTION_CONTRACT["classical"], "seed": 202610120},
}
CONTRACTS = {
    ORIGINAL_CONTRACT_ID: PROMOTION_CONTRACT,
    RECOVERY_CONTRACT_ID: RECOVERY_CONTRACT,
}
