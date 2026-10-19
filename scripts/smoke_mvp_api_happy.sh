#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"

wait_for_live() {
  local attempts=60
  local i
  for i in $(seq 1 "$attempts"); do
    code="$(curl -s -o /tmp/live_happy.json -w "%{http_code}" "${BASE_URL}/health/live" || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "live endpoint did not become ready in time (${BASE_URL})"
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
code="$(curl -s -o /tmp/live_happy.json -w "%{http_code}" "${BASE_URL}/health/live")"
[[ "$code" == "200" ]]
assert_json_field_nonempty /tmp/live_happy.json request_id

# 2) ready should be 200 for fake backend
code="$(curl -s -o /tmp/ready_happy.json -w "%{http_code}" "${BASE_URL}/health/ready")"
[[ "$code" == "200" ]]
assert_json_field_nonempty /tmp/ready_happy.json request_id

# 3) generate should be 200 with stable response shape
TRACE_ID="smoke-trace-happy"
REQUEST_ID="smoke-req-happy"
code="$(curl -s -o /tmp/gen_happy.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -H "X-Trace-Id: ${TRACE_ID}" \
  -H "X-Request-Id: ${REQUEST_ID}" \
  -X POST "${BASE_URL}/v1/generate" \
  -d '{"prompt":"smoke happy path","max_new_tokens":8,"temperature":0.2}')"
[[ "$code" == "200" ]]
assert_json_field_nonempty /tmp/gen_happy.json request_id
assert_json_field_nonempty /tmp/gen_happy.json text

# Validate returned request id is propagated
python - <<'PY'
import json
with open("/tmp/gen_happy.json", "r", encoding="utf-8") as f:
    payload = json.load(f)
assert payload["request_id"] == "smoke-req-happy", payload
assert isinstance(payload["usage"]["input_tokens"], int)
assert isinstance(payload["usage"]["output_tokens"], int)
PY

echo "MVP API happy-path smoke test passed"
