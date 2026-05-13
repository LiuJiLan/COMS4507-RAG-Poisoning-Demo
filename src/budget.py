"""
OpenRouter balance check + per-operation budget gating.
OpenRouter 余额查询 + 单次操作预算检查。

OpenRouter exposes:
    GET https://openrouter.ai/api/v1/credits
    -> {"data": {"total_credits": <float>, "total_usage": <float>}}
    balance = total_credits - total_usage
OpenRouter 端点返回 total_credits / total_usage 两个 float,余额 = total - usage。

CLI:
    python -m src.budget                 # show balance only / 仅查询余额
    python -m src.budget --op pilot      # estimate whether pilot is affordable / 估算 pilot 是否够
"""
import argparse
import logging
import sys

import requests

import config

logger = logging.getLogger(__name__)


CREDITS_ENDPOINT = "https://openrouter.ai/api/v1/credits"
REQUEST_TIMEOUT = 15

# Conservative USD cost estimates per operation. Recalibrate before each run.
# 各操作的保守美元成本估算,跑前再校准。
ESTIMATED_COSTS = {
    "generate_poisons_all":   1.00,
    "generate_poisons_one":   0.20,
    "generate_poisons_pilot": 0.10,
    "variants_precompute":    0.05,
    "experiment_full":        7.00,
    "experiment_pilot":       1.50,
}


def get_openrouter_balance() -> float:
    """
    Return remaining USD balance on OpenRouter. Raises if the key is missing
    or the network call fails.
    查 OpenRouter 余额(USD)。key 缺失或网络失败会抛异常。
    """
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    r = requests.get(
        CREDITS_ENDPOINT,
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return float(data["total_credits"]) - float(data["total_usage"])


def check_budget_or_warn(operation: str, assume_yes: bool = False) -> bool:
    """
    Check the balance. Plenty → True; clearly insufficient → False; tight → interactive prompt.
    检查余额:够 → True;明显不够 → False;紧张 → 交互式询问。

    Args:
        operation:  key from ESTIMATED_COSTS.
        assume_yes: True skips the interactive prompt (CLI --yes / CI use).

    Returns:
        True to continue, False to abort.
    """
    try:
        balance = get_openrouter_balance()
    except Exception as e:
        print(f"WARNING: Could not check OpenRouter balance: {e}")
        print("Proceed at your own risk.")
        if assume_yes:
            return True
        return input("Continue anyway? [y/N]: ").strip().lower() == "y"

    estimated = ESTIMATED_COSTS.get(operation, 1.0)
    print(f"  Operation: {operation}")
    print(f"  OpenRouter balance: ${balance:.4f}")
    print(f"  Estimated cost:     ${estimated:.4f}")

    if balance < estimated:
        print(
            f"\nBALANCE ${balance:.2f} IS BELOW ESTIMATED COST ${estimated:.2f}.\n"
            "Please top up: https://openrouter.ai/credits"
        )
        return False

    if balance < estimated * 1.5:
        print(
            f"\nWARNING: balance is tight (< 1.5x estimated cost). "
            "Recommend topping up before this run."
        )
        if assume_yes:
            print("--yes passed; proceeding anyway.")
            return True
        return input("Continue anyway? [y/N]: ").strip().lower() == "y"

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--op", default=None,
        help=f"Operation name to estimate. One of: {', '.join(ESTIMATED_COSTS)}",
    )
    args = parser.parse_args()

    try:
        balance = get_openrouter_balance()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"OpenRouter balance: ${balance:.4f}")
    if args.op:
        est = ESTIMATED_COSTS.get(args.op)
        if est is None:
            print(f"  (unknown op {args.op!r})")
            sys.exit(2)
        print(f"  Estimated cost of {args.op}: ${est:.4f}")
        print(f"  Balance / estimate ratio: {balance / est:.2f}x")


if __name__ == "__main__":
    main()
