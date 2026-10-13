#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

cases="${1:-cases.jsonl}"
url="${URL:-http://127.0.0.1:8001/v1/decisions}"
[[ -s "$cases" ]] || { printf 'Missing or empty cases file: %s\n' "$cases" >&2; exit 2; }
count=$(jq -s 'length' "$cases")

request() {
  local payload
  payload=$(jq -ce '{state, questions: {q: ({type, instructions} +
    (if has("criteria") then {criteria} else {} end))}}' <<< "$1")
  curl -fsS --connect-timeout 5 --max-time 60 \
    -H 'Content-Type: application/json' --data-binary "$payload" \
    -w '\n%{time_total}' "$url"
}

# Warm all request shapes before measuring latency.
while IFS= read -r case || [[ -n "$case" ]]; do
  [[ -n "$case" ]] || continue
  request "$case" >/dev/null
done < "$cases"

while IFS= read -r case || [[ -n "$case" ]]; do
  [[ -n "$case" ]] || continue
  response=$(request "$case")
  elapsed="${response##*$'\n'}"
  body="${response%$'\n'*}"
  jq -ce --argjson case "$case" --arg elapsed "$elapsed" '
    .model as $model | .answers.q as $answer |
    if $answer.type != $case.type then error("Unexpected answer type") else
    (if $case.type == "noul" then ($answer.noul >= 0.5)
     elif $case.type == "choice" then $answer.choice
     else ($answer.probabilities | to_entries | max_by(.value) | .key | tonumber)
     end) as $prediction |
    {model: $model, type: $case.type, expected: $case.expected, prediction: $prediction,
     correct: ($prediction == $case.expected), latency_ms: (($elapsed | tonumber) * 1000),
     score_error: (if $case.type == "score" then (($answer.score - $case.expected) | fabs) else null end)}
    end' <<< "$body"
done < "$cases" | jq -s --argjson count "$count" '
  if length == 0 or length != $count then error("Benchmark incomplete") else
  (map(.latency_ms) | sort) as $times |
  (map(select(.type == "score") | .score_error)) as $errors |
  {model: .[0].model, requests: length, correct: (map(select(.correct)) | length),
   accuracy: ((map(select(.correct)) | length) / length),
   by_type: (group_by(.type) | map({type: .[0].type, count: length,
     correct: (map(select(.correct)) | length), accuracy: ((map(select(.correct)) | length) / length)})),
   score_mae: (if ($errors | length) > 0 then ($errors | add / length) else null end),
   latency_ms: {average: ($times | add / length), p50: $times[(($times | length) * 0.5 | ceil) - 1],
     p95: $times[(($times | length) * 0.95 | ceil) - 1], maximum: $times[-1]}}
  end'
