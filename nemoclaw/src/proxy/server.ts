// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import http from "node:http";
import https from "node:https";
import { resolveModel, transformRequest } from "./transform.js";
import { PROXY_HEADERS } from "./models.js";
import { shouldRetry, shouldRetryStreamChunk, buildRetryBody } from "./retry.js";

export interface ProxyServerOptions {
  port: number;
  upstreamUrl: string;
  apiKey: string;
  logger?: { info(msg: string): void; warn(msg: string): void; error(msg: string): void };
}

const COMPLETION_PATHS = new Set(["/v1/chat/completions", "/v1/completions"]);

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: string[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk.toString()));
    req.on("end", () => resolve(chunks.join("")));
    req.on("error", reject);
  });
}

function forwardRequest(
  upstreamUrl: string,
  apiKey: string,
  bodyStr: string,
  extraHeaders: Record<string, string>,
  res: http.ServerResponse,
  isStreaming: boolean,
  originalBody: Record<string, unknown>,
  contextWindow: number | undefined,
  logger?: ProxyServerOptions["logger"],
): void {
  const url = new URL(`${upstreamUrl.replace(/\/+$/, "")}/chat/completions`);
  const client = url.protocol === "https:" ? https : http;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "Content-Length": String(Buffer.byteLength(bodyStr)),
    Authorization: `Bearer ${apiKey}`,
    ...extraHeaders,
  };

  const options: https.RequestOptions = {
    hostname: url.hostname,
    port: url.port || (url.protocol === "https:" ? 443 : 80),
    path: url.pathname,
    method: "POST",
    headers,
  };

  const proxyReq = client.request(options, (proxyRes) => {
    if (proxyRes.statusCode !== 200) {
      let errBody = "";
      proxyRes.on("data", (c: Buffer) => (errBody += c.toString()));
      proxyRes.on("end", () => {
        res.writeHead(proxyRes.statusCode ?? 502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: errBody }));
      });
      return;
    }

    if (isStreaming) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      let detectedTruncation = false;
      proxyRes.on("data", (chunk: Buffer) => {
        const text = chunk.toString();
        res.write(chunk);

        // Scan SSE lines for finish_reason: "length"
        for (const line of text.split("\n")) {
          if (line.startsWith("data: ") && line !== "data: [DONE]") {
            if (shouldRetryStreamChunk(line.slice(6))) {
              detectedTruncation = true;
            }
          }
        }
      });

      proxyRes.on("end", () => {
        if (detectedTruncation) {
          res.write(`\ndata: ${JSON.stringify({ x_retry_suggested: true, message: "Response truncated due to max_tokens. Retry with max_tokens multiplied by 4." })}\n\n`);
        }
        res.end();
      });
    } else {
      // Non-streaming: buffer entire response, check for retry
      let respBody = "";
      proxyRes.on("data", (c: Buffer) => (respBody += c.toString()));
      proxyRes.on("end", () => {
        try {
          const parsed = JSON.parse(respBody) as Record<string, unknown>;
          if (shouldRetry(parsed)) {
            logger?.info("Detected finish_reason: length — retrying with 4x max_tokens");
            const retryBodyObj = buildRetryBody(originalBody, contextWindow);
            const retryStr = JSON.stringify(retryBodyObj);
            forwardRequest(
              upstreamUrl, apiKey, retryStr, extraHeaders, res,
              false, retryBodyObj, contextWindow, logger,
            );
            return;
          }
        } catch {
          // Response wasn't valid JSON — pass through as-is
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(respBody);
      });
    }
  });

  proxyReq.on("error", (err) => {
    logger?.error(`Proxy upstream error: ${err.message}`);
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: err.message }));
  });

  proxyReq.write(bodyStr);
  proxyReq.end();
}

/**
 * Start the policy-proxy HTTP server.
 *
 * Intercepts completion requests, applies curated-model transformations
 * (header injection, body merge, model rewrite), and forwards upstream.
 * Non-curated models are forwarded with only the standard proxy headers.
 */
export function startProxyServer(opts: ProxyServerOptions): http.Server {
  const { port, upstreamUrl, apiKey, logger } = opts;

  const server = http.createServer(async (req, res) => {
    // CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // Health check
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
      return;
    }

    // Only proxy completion endpoints
    if (req.method !== "POST" || !COMPLETION_PATHS.has(req.url ?? "")) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
      return;
    }

    try {
      const rawBody = await readBody(req);
      const body = JSON.parse(rawBody) as Record<string, unknown>;
      const modelField = typeof body["model"] === "string" ? (body["model"] as string) : "";

      const curatedModel = resolveModel(modelField);

      let finalBodyStr: string;
      let extraHeaders: Record<string, string>;
      let isStreaming: boolean;
      let contextWindow: number | undefined;

      if (curatedModel) {
        const result = transformRequest(body, curatedModel);
        finalBodyStr = JSON.stringify(result.body);
        extraHeaders = result.headers;
        isStreaming = result.body["stream"] !== false;
        contextWindow = curatedModel.contextWindow;
        logger?.info(`[policy-proxy] Curated model: ${modelField} -> ${curatedModel.prefixedId}`);
      } else {
        // Non-curated: pass through with standard proxy headers only
        extraHeaders = { ...PROXY_HEADERS };
        finalBodyStr = rawBody;
        isStreaming = body["stream"] !== false;
        contextWindow = undefined;
        logger?.info(`[policy-proxy] Pass-through model: ${modelField}`);
      }

      // Use the client's Authorization header if present, fall back to configured key
      const clientAuth = req.headers["authorization"];
      const effectiveKey = clientAuth
        ? clientAuth.replace(/^Bearer\s+/i, "")
        : apiKey;

      forwardRequest(
        upstreamUrl, effectiveKey, finalBodyStr, extraHeaders,
        res, isStreaming, curatedModel ? JSON.parse(finalBodyStr) as Record<string, unknown> : body,
        contextWindow, logger,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger?.error(`[policy-proxy] Request error: ${message}`);
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: `Invalid request: ${message}` }));
    }
  });

  server.listen(port, "127.0.0.1", () => {
    logger?.info(`[policy-proxy] Listening on http://127.0.0.1:${String(port)}`);
  });

  return server;
}
