#!/usr/bin/env bash
# Test inference routing through the policy-proxy for curated models.
# Requires: NVIDIA_API_KEY set, policy-proxy running on port 18990.

PROXY_URL="${PROXY_URL:-http://127.0.0.1:18990}"

echo "=== Test 1: Kimi K2.5 (curated — chat_template_kwargs.thinking) ==="
curl -s "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  -d '{"model":"moonshotai/kimi-k2.5","messages":[{"role":"user","content":"say hello"}]}'
echo -e "\n"

echo "=== Test 2: GLM 5 (curated — chat_template_kwargs.enable_thinking) ==="
curl -s "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  -d '{"model":"z-ai/glm5","messages":[{"role":"user","content":"say hello"}]}'
echo -e "\n"

echo "=== Test 3: GPT-OSS 120B (curated — reasoning_effort high) ==="
curl -s "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"say hello"}]}'
echo -e "\n"

echo "=== Test 4: Non-curated pass-through ==="
curl -s "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  -d '{"model":"nvidia/nemotron-3-super-120b-a12b","messages":[{"role":"user","content":"say hello"}]}'
echo -e "\n"

echo "=== Test 5: Health check ==="
curl -s "${PROXY_URL}/health"
echo -e "\n"
