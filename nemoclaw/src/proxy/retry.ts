// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const RETRY_MAX_TOKENS_MULTIPLIER = 4;

interface Choice {
  finish_reason?: string;
}

interface CompletionResponse {
  choices?: Choice[];
}

/**
 * Check whether a non-streaming completion response was truncated due to
 * token length, indicating the client should retry with a larger max_tokens.
 */
export function shouldRetry(response: Record<string, unknown>): boolean {
  const parsed = response as unknown as CompletionResponse;
  if (!Array.isArray(parsed.choices)) return false;
  return parsed.choices.some((c) => c.finish_reason === "length");
}

/**
 * Scan an SSE data line for `finish_reason: "length"`.
 * Returns true when the chunk signals truncation.
 */
export function shouldRetryStreamChunk(dataLine: string): boolean {
  try {
    const parsed = JSON.parse(dataLine) as CompletionResponse;
    if (!Array.isArray(parsed.choices)) return false;
    return parsed.choices.some((c) => c.finish_reason === "length");
  } catch {
    return false;
  }
}

/**
 * Build a retry body with max_tokens multiplied by the retry factor (4x).
 * Caps at the model's context window when provided.
 */
export function buildRetryBody(
  originalBody: Record<string, unknown>,
  contextWindow?: number,
): Record<string, unknown> {
  const current = typeof originalBody["max_tokens"] === "number"
    ? (originalBody["max_tokens"] as number)
    : 8192;

  let next = current * RETRY_MAX_TOKENS_MULTIPLIER;
  if (contextWindow !== undefined && next > contextWindow) {
    next = contextWindow;
  }

  return { ...originalBody, max_tokens: next };
}
