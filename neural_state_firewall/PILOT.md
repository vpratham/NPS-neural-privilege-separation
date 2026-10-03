# Authenticated local pilot

This is a single-principal, single-permission-realm model API backed by a separate inference process. The host owns the policy, documents and grants. The API accepts a task only (or empty context), authenticates every request, buffers output, and terminates its worker if the wall-clock generation deadline expires. No tool executes here. Tool proposals must go through the existing NPS action broker at the point of effect.

The supported security property is denied-document isolation. Readable prompt injections can still alter answers; the BIPIA review includes concrete counterexamples. This pilot is not approved for general production use or arbitrary natural-language policy enforcement.

## Start

Use Python 3.11 with the existing pinned Torch/Transformers environment and the cached Qwen2.5-0.5B model. From the repository root:

```sh
python3 - <<'PY'
import os, secrets
from pathlib import Path
path = Path('neural_state_firewall/artifacts/pilot_token')
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(secrets.token_urlsafe(32) + '\n')
PY

HF_HOME=neural_state_firewall/.cache/huggingface HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall.pilot \
  --config neural_state_firewall/examples/pilot_config.json \
  --token-file neural_state_firewall/artifacts/pilot_token --port 8765
```

The token file must be an owner-only regular file, not a symlink. Store real document/policy/grant files with permissions appropriate to their contents: the model boundary is not an OS filesystem access control. Replace example document/configuration files only through trusted host administration. Configuration file paths resolve relative to the configuration file. Configuration is snapshotted at startup, so restart after deliberate configuration or token rotation.

Call the service from another terminal without printing the token:

```sh
python3 - <<'PY'
import http.client, json
from pathlib import Path
token = Path('neural_state_firewall/artifacts/pilot_token').read_text().strip()
connection = http.client.HTTPConnection('127.0.0.1', 8765, timeout=70)
connection.request('POST', '/v1/respond',
    json.dumps({'task': 'What day is the workshop? Reply with only the weekday.'}),
    {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
response = connection.getresponse()
print(response.status, response.read().decode())
connection.close()
PY
```

The example returns `Friday`. `/health` also requires authorization and returns 503 when the worker is unavailable. Errors, incomplete generation and worker timeouts return no model text. After worker death, explicit service restart is required; there is no unbounded retry or automatic model reload. Stop with Ctrl-C; the parent terminates and joins its worker.

## Operating contract

- Localhost only; no public listening socket, TLS terminator or remote deployment is included. A remote pilot needs an authenticated ingress designed for that environment.
- One service/token/configuration is one permission realm. There is no shared multi-tenant model cache or principal-switching API. Use separate service instances and credentials for separate realms.
- One inference worker, sequential HTTP servicing, finite socket backlog, 10-second client read timeout and 256-KiB body limit. This is not a throughput service; benchmark actual load before changing concurrency.
- Parent-enforced generation deadline: once it expires, terminate the child and release no partial output. Process termination relies on the OS; it is not a sandbox for malicious model/runtime code.
- Audit entries contain request ID, configuration digest, outcome/reason and elapsed time; they omit request/document/output text and tokens. They go to stderr; provision an access-controlled log sink in a deployment.
- Host configuration fixes model revision, policy, source permissions, context and generation budget. Requests cannot change them. Runtime/model integrity checks remain active.
- The anomaly monitor is not enabled in this pilot: no permission-path-specific semantic detector has passed a release gate. Fail-closed document permissions stay enabled.

## Recorded verification

`tests/test_pilot.py` exercises authentication (including duplicate headers), forged grants, private token-file permissions, worker crash/protocol error, wall-clock timeout/termination, explicit restart, configuration drift and concurrency refusal. Existing tests cover attention masks, host grants, cached state and buffered output.

`validation/pilot_smoke_20261003.json` records six passing real-model HTTP checks: authentication rejection, authenticated readiness, permitted answer, unavailable private answer, forged grants and client context. `validation/boundary_stress_20261003.json` records 100 additional tiny-model denied-content checks and 100 public-input sensitivity checks. These engineering results do not establish semantic prompt-injection resistance.

`validation/release_decision_20261003.json` records the scope and remaining release gates. Production approval is false.
