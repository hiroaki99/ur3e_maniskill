#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CONFIG="configs/ur3e_pick_lift.yaml"
echo "[1/2] 数値仕様を検証"
python scripts/validate_pick_lift_config.py --config "${CONFIG}" --output-dir reports
echo
echo "[2/2] 既存UR3e環境を検査"
python scripts/day1_inspect_ur3e.py --config "${CONFIG}" --env-id UR3eReach-v0 --register-module envs.ur3e_reach --steps 20
echo
echo "Day 1 checks completed."
echo "reports/day1_pick_lift_spec.md"
echo "reports/day1_pick_lift_spec.json"
echo "reports/day1_ur3e_inspection.json"
