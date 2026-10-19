#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

wait_for_live() {
  local attempts=60
  local i
  for i in $(seq 1 "$attempts"); do
    code="$(curl -s -o /tmp/live.json -w "%{http_code}" "${BASE_URL}/health/live" || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "live endpoint did not become ready in time"
  return 1
}

assert_json_field_nonempty() {
  local file="$1"
  local field="$2"
  python - "$file" "$field" <<'PY'
import json, sys
path = sys.argv[2].split(".")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
cur = data
for p in path:
    if p not in cur:
        raise SystemExit(f"missing field: {sys.argv[2]}")
    cur = cur[p]
if cur in ("", None):
    raise SystemExit(f"empty field: {sys.argv[2]}")
PY
}

wait_for_live

# 1) live must be 200
code="$(curl -s -o /tmp/live.json -w "%{http_code}" "${BASE_URL}/health/live")"
[[ "$code" == "200" ]]
assert_json_field_nonempty /tmp/live.json request_id

# 2) ready should be 503 in default smoke (model not loaded)
code="$(curl -s -o /tmp/ready.json -w "%{http_code}" "${BASE_URL}/health/ready" || true)"
[[ "$code" == "503" ]]
assert_json_field_nonempty /tmp/ready.json detail.request_id

# 3) generate should fail with 503 and include request_id in error
code="$(curl -s -o /tmp/gen.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -X POST "${BASE_URL}/v1/generate" \
  -d '{"prompt":"smoke test"}' || true)"
[[ "$code" == "503" ]]
assert_json_field_nonempty /tmp/gen.json detail.request_id

echo "MVP API smoke test passed"
