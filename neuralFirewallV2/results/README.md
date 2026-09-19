# Neural Firewall results

This directory stores versioned experiment outputs and archived runs. A result is interpretable only with its matching experiment manifest, split definition, model revision, artifact hashes, and evaluation protocol.

Use distinct names for:

- request-level intent-monitor performance;
- target-model behavioral outcomes;
- firewall interruption and false-block rates;
- benign task utility; and
- latency or resource use.

Do not combine these into one undifferentiated safety score. Generated model weights, large activation caches, and raw bulk outputs should live in a versioned release or artifact store; commit their manifest and aggregate tables here.
