# Prompt-injection firewall

This is the prompt-injection feature. It sits in front of a model API, preserves
the application-owned policy, labels retrieved text as untrusted, and holds the
complete answer until a policy judge checks it. It returns an OpenAI Chat
Completions response, so supported clients can point at its local URL.

The older `demo` and `run` commands exercise a different component: a broker for
authorizing tool actions. They do not inspect ordinary answers. This firewall
endpoint currently handles text-only requests; it rejects tool calls. Tool
actions need an explicit app integration that routes only those calls through the
broker.

## Run the API firewall

Create the local service policy file from the checked-in example and start the
server. Keep this policy owned by the application operator; clients cannot
replace it in their requests.

```bash
python3 -m nps_gateway firewall-serve \
  --policy-file examples/gateway/firewall-policy.txt \
  --provider ollama --model qwen2.5:3b --judge-model qwen2.5:3b
```

The server listens on `http://127.0.0.1:8765` by default. Change the client’s
OpenAI-compatible base URL to `http://127.0.0.1:8765/v1`. The firewall selects the
upstream model at server startup; it ignores a client's `model` value. `/health`
reports readiness.

Send the ordinary chat request and the retrieved document as a separate
untrusted-context field:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/gateway/injected-context.json
```

The `nps_untrusted_context` field is an extension understood by this firewall.
Your retrieval layer must put tool results and external documents there. Previous
conversation turns are treated as untrusted context. The current user message is
the task. Client `system` and `developer` messages are rejected; trusted policy
comes from the service's policy file. On approval, the endpoint releases a
complete answer. On a block, the OpenAI-style response has `finish_reason` set to
`content_filter` and no answer content. Provider, judge and malformed-response
errors fail closed. `stream: true` and tool calls are rejected, so no partial
answer can escape before review.

To use another compatible API:

```bash
python3 -m nps_gateway firewall-serve \
  --policy-file examples/gateway/firewall-policy.txt \
  --provider compatible --base-url https://your-provider.example/v1 \
  --model YOUR_ANSWER_MODEL --judge-model YOUR_JUDGE_MODEL \
  --judge-base-url https://your-independent-judge.example/v1
```

For actual deployments, use a separately controlled judge model and provider
where possible. The smoke test uses Qwen 3B for both calls to keep it runnable
locally; a second call to the same model family is not an independent defense.
Each response takes at least two model calls, so expect extra latency and cost.

## Demo and integration boundary

```bash
python3 -m nps_gateway firewall-demo --provider offline
python3 -m nps_gateway firewall-demo --provider ollama --model qwen2.5:3b
```

The offline command tests the gate's control flow with scripted responses; it is
not model evidence. The live command checks a clean answer and a retrieved text
that tells the model to override the task and ask for a password. The verified
Qwen 3B run released the normal answer and withheld the injected case. This is one
task and one model. It does not estimate general attack resistance.

The input pattern matcher is only a signal for the judge; it does not block by
itself and it is not a general injection detector. The policy judge is another
model call and can miss attacks or block safe responses. The service does not
claim immunity. It does not identify attacker intent with certainty, protect
model activations, or make arbitrary business decisions. For higher-assurance
applications, supplement it with deterministic output constraints, canary/secret
egress checks, restricted retrieval and action-specific authorization. The action
broker protects only explicitly integrated tool effects; it cannot authorize
ordinary generated text.

This server binds only to loopback and has no client authentication. It is a local
integration surface, not a shared network service. Do not expose it remotely
without adding authenticated client identity, TLS, request limits and deployment
isolation. Do not put credentials or unapproved confidential records in context.

See the [historical evidence-to-design review](WORKING_SYSTEM_EVIDENCE.md) and
[action-broker implementation guide](WORKING_SYSTEM.md).

The current check ran one local Qwen 3B answer/judge pair plus a loopback HTTP
request smoke test; the API gate also has deterministic unit coverage. This does
not cover concurrent production load, remote authentication, or provider outages.

## Verify

```bash
python3 -m unittest discover -s tests/gateway -v
python3 -m compileall -q nps_gateway
```
