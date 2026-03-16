// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CURATED_MODELS,
  DEFAULT_MAX_TOKENS,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  PROXY_HEADERS,
  type CuratedModel,
} from "./models.js";

export interface TransformResult {
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

/**
 * Resolve a model field (as sent by the client) to a curated model definition.
 * Returns null when the model is not in the curated catalog.
 */
export function resolveModel(modelField: string): CuratedModel | null {
  return CURATED_MODELS.get(modelField) ?? null;
}

/**
 * Transform a client request body for a curated model.
 *
 * - Rewrites `model` to the `private/openshell/…` prefixed form
 * - Merges default sampling params when not supplied by the client
 * - Sets `stream: true` unless the client explicitly sent `false`
 * - Spreads per-model extra body fields
 * - Returns the extra upstream headers to inject
 */
export function transformRequest(
  body: Record<string, unknown>,
  model: CuratedModel,
): TransformResult {
  const merged: Record<string, unknown> = { ...body };

  merged["model"] = model.prefixedId;

  if (merged["temperature"] === undefined) {
    merged["temperature"] = DEFAULT_TEMPERATURE;
  }
  if (merged["top_p"] === undefined) {
    merged["top_p"] = DEFAULT_TOP_P;
  }
  if (merged["max_tokens"] === undefined) {
    merged["max_tokens"] = DEFAULT_MAX_TOKENS;
  }
  if (merged["stream"] !== false) {
    merged["stream"] = true;
  }

  for (const [key, value] of Object.entries(model.extraBody)) {
    merged[key] = value;
  }

  return {
    headers: { ...PROXY_HEADERS },
    body: merged,
  };
}
