# Neural Runtime Monitor

Standalone research repository for activation-trajectory monitoring and buffered response release. The monitor compares each request's hidden-state trajectory with a model-specific benign profile and can withhold a completed response on an alarm.

This is experimental. An anomalous trajectory is not proof of an attack, a normal trajectory is not proof of safety, and current evidence does not justify enforcement for real users. Keep it in monitor mode until a fresh source-separated study meets the security, utility, false-block and latency gates described in [`neural_state_firewall/PRODUCTION_ROADMAP.md`](neural_state_firewall/PRODUCTION_ROADMAP.md).

The monitor does not implement document read permissions or execute tools. Use its pinned-model identity, calibration and paired evaluation workflow as documented in [`neural_state_firewall/PRODUCTION_ROADMAP.md`](neural_state_firewall/PRODUCTION_ROADMAP.md).

Run its tests with:

```sh
python -m unittest discover -s neural_state_firewall/tests -v
```
