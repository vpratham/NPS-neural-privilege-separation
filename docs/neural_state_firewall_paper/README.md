# Neural State Firewall paper draft

This folder contains an evidence-bounded manuscript draft, bibliography, figure generator, and sanitized summaries of the six-case paired smoke run. It contains no raw prompts or model completions.

## Compile

With a TeX installation that includes `natbib`, `graphicx`, `booktabs`, and `amsmath`:

```sh
cd docs/neural_state_firewall_paper
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

The current environment does not include a LaTeX compiler, so only source-level checks can be run here.

## Rebuild figures

The case and trajectory CSV files are sanitized derivatives of the private result artifact `neural_state_firewall/artifacts/paired-results-v4.json` (SHA-256 `7faa8363d320af768a94caa8dfe47cbdbea789bfec415f583910e63a045b455b`). They expose case class, observed guard status, and token-step CUSUM telemetry only; prompt and response text are excluded. Exact code, model, and split digests are recorded in `data/paired_run_manifest.txt`. `historical_predecessor_results.csv` contains aggregate historical figures transcribed from the repository audit and NFW-003 report. To regenerate both figures from the tracked CSV files:

```sh
python3 make_figures.py
```

To regenerate the sanitized CSVs from an available local raw artifact:

```sh
python3 make_figures.py --results /path/to/paired-results-v4.json --export-data
```

Matplotlib is required for figure generation. Generated PNGs are tracked so the manuscript compiles without Python.

## Evidence boundary

This is a research draft, not a submission-ready efficacy paper. The paired run has six author-constructed cases (three per manifest condition), only three source groups, no independent output labels, and one model/profile. Its observed block proportions are not attack-success rate or false-block rate. The software tests establish engineering behavior, not security efficacy. The manuscript states these limitations and does not claim a release-worthy firewall.
