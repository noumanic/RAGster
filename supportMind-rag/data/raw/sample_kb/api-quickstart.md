# API quickstart

The Helios REST API lets your applications read sensor data, push commands,
and subscribe to real-time events.

## Base URL and authentication

All requests go to `https://api.helios.example/v1`. Authenticate with a bearer
token created in **Account → API tokens**. Tokens are scoped: a `read` token
cannot push commands, and an `events` token cannot read historical data.

```
curl -H "Authorization: Bearer hlt_pk_..." \
     https://api.helios.example/v1/devices
```

## Rate limits

- 60 requests per minute per token.
- 5 simultaneous WebSocket connections per account.
- Bulk endpoints (`/devices:bulkUpdate`) cost 5 quota units per call.

When you exceed the limit, the API returns HTTP 429 with a `Retry-After`
header. Back off and retry — do not hammer the endpoint.

## Pagination

List endpoints return at most 100 items per page. To paginate, pass the
`next_page` token from the previous response:

```
GET /v1/events?limit=100
GET /v1/events?limit=100&page=eyJjdXJzb3IiOiI...
```

## Webhook events

Subscribe to events with:

```
POST /v1/webhooks
{ "url": "https://your-app.example/hook",
  "events": ["device.online", "device.alert"] }
```

We sign every webhook with the secret returned at creation time. Verify the
`X-Helios-Signature` header before processing — unsigned requests should be
discarded.

## SDKs

Official SDKs are available for Python, Node.js, and Go. For other languages,
use the OpenAPI spec at `https://api.helios.example/openapi.json`.

## Error codes

- `401` — invalid or expired token.
- `403` — token scope insufficient (e.g., a `read` token tried to issue a
  command).
- `409` — version conflict on optimistic concurrency.
- `429` — rate limit exceeded (back off).
- `5xx` — server error; safe to retry with exponential backoff.
