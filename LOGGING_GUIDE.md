# Structured Error & Operation Logging

## Overview

As of this commit, PaperMind has a comprehensive structured logging system that captures all critical operations (uploads, queries, billing, etc.) with context (request ID, user ID, operation duration) and automatically reports errors to Sentry.

## Key Components

### 1. Enhanced Logger (`api/logger.py`)

Two main APIs:

#### `generate_request_id() -> str`
Generates a unique 8-character request ID for tracing across logs and Sentry.

#### `log_operation(operation, status, req_id, user_id, duration_ms, error, context)`
Logs a structured operation with:
- **operation**: Operation name (e.g., "upload_pdf", "ingest_document", "query_paper")
- **status**: "success" or "error"
- **req_id**: Request ID for tracing
- **user_id**: User ID for ownership tracking
- **duration_ms**: Operation duration in milliseconds
- **error**: Exception or error message (optional)
- **context**: Extra context dict (paper_id, filename, etc.)

### 2. Request ID Middleware

Every HTTP request gets a unique request ID via middleware that:
1. Generates a request ID at entry
2. Attaches it to `request.state.req_id`
3. Tags all Sentry errors with this ID
4. Returns it in the `X-Request-ID` response header

This enables end-to-end tracing of a user action across logs and error tracking.

### 3. Log Files

Structured logs are written to:
- **`logs/operations.log`** — All operations (success and error) as JSON lines, rotating at 50 MB
- **`logs/queries.jsonl`** — Query telemetry (existing, unchanged)

Example operations.log entry:
```json
{
  "timestamp": "2026-06-18T12:30:45.123456+00:00",
  "operation": "upload_pdf",
  "status": "success",
  "req_id": "a1b2c3d4",
  "user_id": "user_12345",
  "duration_ms": 2150,
  "paper_id": "paper_abc123",
  "filename": "paper.pdf"
}
```

### 4. Sentry Integration

Every error operation with `status: "error"` and a real exception is automatically:
1. Sent to Sentry with full context (req_id, user_id, operation, duration)
2. Tagged with request_id for correlation
3. Guarded by SENTRY_DSN env var (fail-open if unset)

This means:
- Operations teams can tail `logs/operations.log` locally for debugging
- Production issues appear in Sentry with full context
- No logs are lost if Sentry is down

## Logging Coverage

### API Endpoints
- `/upload` — File upload, PDF validation, storage upload
- `/delete` — PDF deletion, registry cleanup, Chroma cleanup, cache invalidation
- `/query` — Single and multi-paper queries with timeout handling
- `/query/stream` — Streaming queries with progress updates
- `/papers/{id}/glossary` — Glossary extraction
- `/papers/{id}/recommendations` — Recommendation search
- `/rewrite` — Text rewriting

### Billing
- `POST /billing/checkout` — Stripe Checkout session creation
- `POST /billing/portal` — Customer portal
- `POST /billing/webhook` — Subscription lifecycle events

### Background Tasks
- `run_ingestion_from_storage` — Paper ingestion with temp file cleanup
- `regenerate_missing_collections` — Startup Chroma rebuild from R2

## Usage in Your Code

### Logging Success
```python
log_operation(
    "extract_glossary",
    "success",
    user_id=user_id,
    context={"paper_id": paper_id},
    duration_ms=round((time.monotonic() - t0) * 1000),
)
```

### Logging Errors
```python
log_operation(
    "extract_glossary",
    "error",
    user_id=user_id,
    error=exc,  # Pass the exception object
    context={"paper_id": paper_id},
    duration_ms=round((time.monotonic() - t0) * 1000),
)
```

### Accessing Request ID in Routes
```python
@app.post("/upload")
async def upload_paper(request: Request, ...):
    req_id = getattr(request.state, 'req_id', 'unknown')
    log_operation("upload_pdf", "success", req_id=req_id, user_id=user_id, ...)
```

## Deployment Notes

### Local Development
No configuration needed. Logs write to `logs/operations.log` (git-ignored). Sentry is a no-op without `SENTRY_DSN`.

### Production (HF Spaces)
Set environment variables:
- `SENTRY_DSN` — Your Sentry project DSN
- `SENTRY_ENVIRONMENT` — e.g., "production" (defaults to "development")

Logs rotate at 50 MB in `logs/operations.log` (keep 5 backups). For long-term retention, pipe logs to CloudWatch or your logging provider.

## Debugging with Logs

### Tail operations in real-time
```bash
tail -f logs/operations.log | grep -E '"status":"error"' | jq .
```

### Find all errors for a user
```bash
grep "user_12345" logs/operations.log | jq 'select(.status=="error")'
```

### Find a specific request
```bash
grep "a1b2c3d4" logs/operations.log | jq .
```

### Count operations by type
```bash
jq -r '.operation' logs/operations.log | sort | uniq -c | sort -rn
```

## Future Enhancements

1. **Metrics export** — Emit operation durations/counts to CloudWatch, Datadog, or Prometheus
2. **Correlation IDs** — Propagate request_id through service calls (e.g., to discovery module)
3. **Sampling** — Log only a % of successful operations to reduce disk usage
4. **Alerts** — Trigger alarms on error rates exceeding thresholds (e.g., >5% failures)
