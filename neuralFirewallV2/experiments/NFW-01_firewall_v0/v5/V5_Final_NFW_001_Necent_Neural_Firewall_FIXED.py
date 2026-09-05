#!/usr/bin/env python
# coding: utf-8

# # NFW-001 — Neural Activation Safety Interrupt (Necent, frozen Exp017 probes)
# ### AUDITED / CORRECTED VERSION
# 
# **Purpose.** Determine whether activation-space risk (frozen Exp017 probes, layers
# 19/20/21/22 of `Qwen/Qwen2.5-3B-Instruct`) predicts unsafe *generation*, and
# whether it can support a runtime safety interrupt ("neural firewall").
# 
# **This revision fixes 13 audit findings from the previous version.** See the
# `CORRECTIONS_APPLIED` list defined in Stage 1 and echoed in the Stage-18
# forensic report for the full list. In summary:
# 
# 1. Dataset revision/fingerprint now recorded in the manifest; split integrity
#    verification unchanged (hash, sizes, zero overlap, OOD separation) — still
#    **stops** rather than resampling if the locked split can't be verified.
# 2. `response_harmful` provenance is now **audited** (Stage 2d) before being
#    used as behavioral ground truth — the notebook records whether the labeled
#    response can be tied to Qwen or another model/source, and carries that
#    caveat into the final report rather than asserting it's Qwen-specific.
# 3. The activation location used by streaming (per-token) extraction is now
#    **empirically verified** against `hidden_states[L]` before any probe
#    scoring happens (Stage 6), fixing a real layer-index mismatch between the
#    prompt-level and generation-time extraction paths in the prior version.
# 4. Two explicit, separately-named threshold sets: **`PROMPT_THRESHOLDS`**
#    (calibrated against `prompt_harmful`) and **`BEHAVIORAL_THRESHOLDS`**
#    (calibrated against `response_harmful`). The runtime firewall uses
#    `BEHAVIORAL_THRESHOLDS` only.
# 5. Behavioral calibration uses **only** calibration-split rows that already
#    carry a `response_harmful` label — both classes are asserted present or the
#    notebook stops.
# 6. Persistence window `k` (1/2/3) is now selected **only** on the behavioral
#    calibration subset, via a documented rule, and locked as
#    `LOCKED_PERSISTENCE_K` before `behavioral_test` or `test_ood` are touched.
#    The previous version selected `k` from `behavioral_test` — a leak. That
#    leak is fixed.
# 7. Token/temporal semantics are now explicit: every trajectory records
#    `decision_step`, `predicted_token_id` — the activation at `decision_step t`
#    is the state used to *produce* token `t`, never a representation of an
#    already-generated token.
# 8. The invalid lead-time metric (`total_tokens_seen - block_token_index`,
#    which was ~1 by construction) is **removed** and replaced with
#    `remaining_unmonitored_tokens = unmonitored_generation_length -
#    firewall_block_decision_step`, explicitly *not* described as "tokens
#    before harmful content" (Necent has no token-level harm annotation).
# 9. Metrics are namespaced explicitly: `prompt_detection_rate` / `prompt_fpr`,
#    `behavioral_detection_rate` / `behavioral_fpr`, `firewall_detection_rate` /
#    `firewall_false_block_rate`, `remaining_unmonitored_tokens`.
# 10. An explicit, standalone assertion (Stage 3d) confirms `behavioral_test` is
#     disjoint from every other locked split, immediately before any evaluation
#     stage runs (in addition to the general pairwise check in Stage 3c).
# 11. Frozen-probe validation is unchanged in spirit (dimensions, layer IDs,
#     scaler validity, finite weights) — still stops rather than substituting.
# 12. The final report keeps **A) prompt-level detectability, B) behavioral
#     prediction, C) runtime firewall interruption, D) OOD generalization**
#     fully separate — no single collapsed "safety score", and interruption is
#     never described as proof of harm prevention.
# 13. All Drive checkpointing/resumability is preserved; new stages
#     (behavioral-calibration generation) get their own checkpoint directories
#     under `runs/<run_id>/`.
# 

# ## Stage 1 — Environment, Imports, Reproducibility

# In[ ]:


# Stage 1a: core imports.
import os, sys, json, time, hashlib, random, platform, subprocess, shutil, zipfile, gc
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Python {sys.version.split()[0]}  |  platform: {platform.platform()}")
print(f"numpy {np.__version__}  |  pandas {pd.__version__}")

CORRECTIONS_APPLIED = [
    "1. Dataset revision/fingerprint recorded in manifest; split verification unchanged (stops on failure).",
    "2. response_harmful provenance audited (model_name/source check) before being treated as behavioral ground truth.",
    "3. Activation location (hidden_states[L] vs decoder-layer hook output) empirically verified; streaming hook offset fixed.",
    "4. PROMPT_THRESHOLDS and BEHAVIORAL_THRESHOLDS are now separate, explicitly named threshold sets.",
    "5. Behavioral calibration restricted to calibration-split rows with an existing response_harmful label; both classes asserted present.",
    "6. Persistence k selected only on behavioral-calibration data via a documented rule; locked before touching behavioral_test/OOD.",
    "7. Trajectories store decision_step / predicted_token_id explicitly; activation-at-t is the state used to produce token t.",
    "8. Invalid lead-time metric removed; replaced with remaining_unmonitored_tokens (unmonitored_generation_length - firewall_block_decision_step).",
    "9. Metrics namespaced: prompt_detection_rate/prompt_fpr, behavioral_detection_rate/behavioral_fpr, firewall_detection_rate/firewall_false_block_rate.",
    "10. Explicit standalone behavioral_test immutability assertion added (Stage 3d), ahead of any evaluation stage.",
    "11. Frozen-probe validation unchanged: dimensions/layer IDs/scaler/finite-weight checks, stops rather than substituting.",
    "12. Final report keeps prompt-level / behavioral / firewall / OOD results in separate sections; no collapsed safety score; blocks are never claimed as harm prevention.",
    "13. Resumable Drive checkpointing preserved; new behavioral-calibration generation stage gets its own checkpoint directory.",
]
print(f"\n{len(CORRECTIONS_APPLIED)} corrections applied in this revision (see forensic report for full text).")


# In[ ]:


# Stage 1b: global seeding.
GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_all_seeds(GLOBAL_SEED)
print(f"Seeded RNGs with GLOBAL_SEED={GLOBAL_SEED}")


# In[ ]:


# Stage 1c: CONFIG.
CONFIG = {
    "run_id": None,
    "seed": GLOBAL_SEED,

    "drive_mount_point": "/content/drive",
    "project_root_rel": "MyDrive/NFW-001",
    "split_file_rel": "MyDrive/NFW-001/splits/necent_nfw_splits_v2.json",
    "probe_artifact_dir_rel": "MyDrive/NFW-001/artifacts/exp017",
    "runs_dir_rel": "MyDrive/NFW-001/runs",
    "dataset_cache_rel": "MyDrive/NFW-001/cache/necent_train.parquet",

    "hf_dataset_id": "Necent/llm-jailbreak-prompt-injection-dataset",
    "hf_split": "train",

    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    "model_revision": None,
    "probe_layers": [19, 20, 21, 22],
    "pooling": "last_token",
    "dtype": "bfloat16",

    "expected_split_sizes": {
        "calibration": 1000, "test_benign": 1000, "test_harmful": 1000,
        "test_jailbreak": 1000, "test_injection": 1000, "test_ood": 1000,
        "behavioral_test": 1000,
    },

    "target_fpr": 0.02,
    "persistence_sweep": [1, 2, 3],
    "voting": "k_of_n",
    "vote_k": 2,

    "max_new_tokens": 256,
    "generation_batch_checkpoint_every": 25,

    "force_recompute": False,
}
print("CONFIG keys:", list(CONFIG.keys()))


# In[ ]:


# Stage 1d: run_id.
if CONFIG["run_id"] is None:
    CONFIG["run_id"] = "nfw001_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print("run_id:", CONFIG["run_id"])


# In[ ]:


# Stage 1e: hashing / reproducibility utilities.
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_json_canonical(obj) -> str:
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canon.encode("utf-8"))

def capture_package_versions(pkgs=("torch", "transformers", "datasets", "huggingface_hub",
                                    "scikit-learn", "numpy", "pandas", "scipy")):
    versions = {}
    for pkg in pkgs:
        try:
            out = subprocess.run([sys.executable, "-m", "pip", "show", pkg],
                                  capture_output=True, text=True, timeout=30)
            ver = None
            for line in out.stdout.splitlines():
                if line.lower().startswith("version:"):
                    ver = line.split(":", 1)[1].strip()
            versions[pkg] = ver
        except Exception as e:
            versions[pkg] = f"ERROR:{e}"
    return versions

def now_iso():
    return datetime.now(timezone.utc).isoformat()

print("Reproducibility utilities defined.")


# In[ ]:


# Stage 1f: manifest / checkpoint helpers.
RUN_MANIFEST = {
    "run_id": CONFIG["run_id"],
    "created_at": now_iso(),
    "config": dict(CONFIG),
    "stages_completed": [],
    "artifact_hashes": {},
    "package_versions": capture_package_versions(),
    "notes": [],
    "corrections_applied": CORRECTIONS_APPLIED,
}

def mark_stage_done(stage_name: str, **extra):
    entry = {"stage": stage_name, "completed_at": now_iso(), **extra}
    RUN_MANIFEST["stages_completed"].append(entry)

def stage_is_done(stage_name: str) -> bool:
    return any(e["stage"] == stage_name for e in RUN_MANIFEST["stages_completed"])

def save_manifest(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(RUN_MANIFEST, f, indent=2, default=str)

def load_manifest_if_exists(run_dir):
    p = Path(run_dir) / "manifest.json"
    if not p.exists():
        return False
    with open(p) as f:
        prior = json.load(f)
    RUN_MANIFEST["stages_completed"] = prior.get("stages_completed", [])
    RUN_MANIFEST["artifact_hashes"] = prior.get("artifact_hashes", {})
    RUN_MANIFEST["notes"] = prior.get("notes", [])
    return True

print("Manifest helpers defined.")


# ## Stage 2 — Google Drive Mount + Necent Loading

# In[ ]:


# Stage 2a: mount Drive (idempotent).
from google.colab import drive  # noqa: E402

DRIVE_ROOT = Path(CONFIG["drive_mount_point"])
if not DRIVE_ROOT.exists() or not any(DRIVE_ROOT.iterdir()):
    drive.mount(str(DRIVE_ROOT), force_remount=False)
else:
    print("Drive already mounted.")

PROJECT_ROOT = DRIVE_ROOT / CONFIG["project_root_rel"]
SPLIT_FILE = DRIVE_ROOT / CONFIG["split_file_rel"]
PROBE_DIR = DRIVE_ROOT / CONFIG["probe_artifact_dir_rel"]
RUNS_DIR = DRIVE_ROOT / CONFIG["runs_dir_rel"]
RUN_DIR = RUNS_DIR / CONFIG["run_id"]
DATASET_CACHE = DRIVE_ROOT / CONFIG["dataset_cache_rel"]

for d in (PROJECT_ROOT, RUNS_DIR, RUN_DIR, DATASET_CACHE.parent):
    d.mkdir(parents=True, exist_ok=True)

resumed = load_manifest_if_exists(RUN_DIR)
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"SPLIT_FILE   = {SPLIT_FILE}  (exists={SPLIT_FILE.exists()})")
print(f"PROBE_DIR    = {PROBE_DIR}  (exists={PROBE_DIR.exists()})")
print(f"RUN_DIR      = {RUN_DIR}  (resumed_prior_manifest={resumed})")


# In[ ]:


# ============================================================
# Stage 2b — Authenticate + Load Necent with immutable cache
# ============================================================

from pathlib import Path
import json
import pandas as pd

from huggingface_hub import (
    login,
    get_token,
    dataset_info,
)
from datasets import load_dataset


DATASET_CACHE_META = DATASET_CACHE.with_suffix(
    ".metadata.json"
)


# ------------------------------------------------------------
# 1. Authenticate with Hugging Face
# ------------------------------------------------------------

print("=" * 80)
print("HUGGING FACE AUTHENTICATION")
print("=" * 80)

# SECURITY FIX: a live HF token was previously hardcoded here in plaintext.
# Never commit tokens to notebooks. Read from an environment variable first
# (falls through to the Colab Secret lookup below if unset).
token = os.environ.get("HF_TOKEN")

if token is None:

    print(
        """
No Hugging Face token is currently available.

Because Necent is a gated dataset, this Colab runtime
must have a Hugging Face token belonging to an account
that has been granted access to Necent.

If you stored HF_TOKEN as a Colab Secret, retrieve it
using the code below.
"""
    )

    try:
        from google.colab import userdata

        token = userdata.get(
            "HF_TOKEN"
        )

    except Exception as e:

        raise RuntimeError(
            "No Hugging Face token found.\n\n"
            "Add your Hugging Face token to Colab Secrets "
            "as HF_TOKEN, then rerun this cell.\n\n"
            f"Original error: {e}"
        )


if not token:

    raise RuntimeError(
        "HF_TOKEN is empty. "
        "Necent is gated and requires authentication."
    )


# Authenticate using the installed API.
# Do NOT use new_session=...
login(
    token=token,
    add_to_git_credential=False,
)

print(
    "Hugging Face authentication successful."
)


# ------------------------------------------------------------
# 2. Resolve exact Necent revision
# ------------------------------------------------------------

print()
print(
    "Resolving Necent dataset revision..."
)

try:

    info = dataset_info(
        CONFIG["hf_dataset_id"],
        revision=CONFIG["hf_split"],
    )

except Exception:

    # The dataset revision should be independent
    # of the split name; retry normally.
    info = dataset_info(
        CONFIG["hf_dataset_id"]
    )


DATASET_REVISION_SHA = getattr(
    info,
    "sha",
    None,
)

if not DATASET_REVISION_SHA:

    raise RuntimeError(
        "Could not resolve the Necent dataset commit SHA."
    )


print(
    "Necent revision:",
    DATASET_REVISION_SHA,
)


# ------------------------------------------------------------
# 3. Load verified Drive cache if available
# ------------------------------------------------------------

def load_necent(
    force_redownload=False
):

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if (
        DATASET_CACHE.exists()
        and not force_redownload
    ):

        print(
            "Found Necent cache:"
        )

        print(
            DATASET_CACHE
        )

        if not DATASET_CACHE_META.exists():

            raise RuntimeError(
                "Necent parquet cache exists but its metadata "
                "file is missing.\n\n"
                f"Cache:\n{DATASET_CACHE}\n\n"
                f"Metadata:\n{DATASET_CACHE_META}\n\n"
                "Refusing to use an unverifiable cache."
            )

        with open(
            DATASET_CACHE_META,
            "r",
        ) as f:

            cache_meta = json.load(f)

        cached_revision = (
            cache_meta.get(
                "dataset_revision"
            )
        )

        if (
            cached_revision
            != DATASET_REVISION_SHA
        ):

            raise RuntimeError(
                "NECENT CACHE REVISION MISMATCH.\n\n"
                f"Cached revision: "
                f"{cached_revision}\n"
                f"Current revision: "
                f"{DATASET_REVISION_SHA}\n\n"
                "Delete the stale cache and rerun."
            )

        df = pd.read_parquet(
            DATASET_CACHE
        )

        cached_rows = cache_meta.get(
            "rows"
        )

        if (
            cached_rows is not None
            and len(df) != cached_rows
        ):

            raise RuntimeError(
                "Necent cache row count does not "
                "match its metadata."
            )

        print(
            f"Verified cached Necent: "
            f"{len(df):,} rows"
        )

        return df


    # --------------------------------------------------------
    # Fresh gated download
    # --------------------------------------------------------

    print(
        f"Downloading gated dataset:\n"
        f"{CONFIG['hf_dataset_id']}"
    )

    print(
        f"Split: {CONFIG['hf_split']}"
    )

    print(
        f"Revision: {DATASET_REVISION_SHA}"
    )

    ds = load_dataset(
        CONFIG["hf_dataset_id"],
        split=CONFIG["hf_split"],
        revision=DATASET_REVISION_SHA,
        token=token,
    )

    df = ds.to_pandas()

    print(
        f"Downloaded: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Save permanent cache
    # --------------------------------------------------------

    DATASET_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        DATASET_CACHE
    )

    cache_metadata = {

        "dataset":
            CONFIG["hf_dataset_id"],

        "split":
            CONFIG["hf_split"],

        "dataset_revision":
            DATASET_REVISION_SHA,

        "rows":
            int(len(df)),

        "columns":
            list(df.columns),

        "created_utc":
            now_iso(),
    }

    with open(
        DATASET_CACHE_META,
        "w",
    ) as f:

        json.dump(
            cache_metadata,
            f,
            indent=2,
        )

    print(
        "Permanent Necent cache saved."
    )

    print(
        DATASET_CACHE
    )

    return df


# ------------------------------------------------------------
# 4. Load
# ------------------------------------------------------------

necent_df = load_necent(
    force_redownload=CONFIG[
        "force_recompute"
    ]
)

NECENT_ROW_COUNT = len(
    necent_df
)


# ------------------------------------------------------------
# 5. Record manifest
# ------------------------------------------------------------

RUN_MANIFEST[
    "artifact_hashes"
][
    "hf_dataset_commit_sha"
] = DATASET_REVISION_SHA

RUN_MANIFEST[
    "artifact_hashes"
][
    "necent_row_count_at_load"
] = NECENT_ROW_COUNT

RUN_MANIFEST[
    "artifact_hashes"
][
    "necent_cache_metadata"
] = str(
    DATASET_CACHE_META
)


print()
print("=" * 80)
print("NECENT LOAD SUCCESSFUL")
print("=" * 80)

print(
    f"Dataset: "
    f"{CONFIG['hf_dataset_id']}"
)

print(
    f"Revision: "
    f"{DATASET_REVISION_SHA}"
)

print(
    f"Rows: "
    f"{NECENT_ROW_COUNT:,}"
)

print(
    f"Cache: "
    f"{DATASET_CACHE}"
)

mark_stage_done(
    "stage2_load_necent",
    row_count=NECENT_ROW_COUNT,
    dataset_revision=DATASET_REVISION_SHA,
)

save_manifest(
    RUN_DIR
)


# In[ ]:


# Stage 2c: defensive column detection. Fails loudly on a REQUIRED column;
# degrades gracefully (None) on optional ones used only for provenance.
def _find_col(df, candidates, required=True, label=""):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Could not locate a '{label}' column in Necent among candidates "
            f"{candidates}. Available columns: {list(df.columns)}. STOPPING."
        )
    return None

NECENT_COLS = {
    "prompt": _find_col(necent_df, ["prompt", "text", "input", "user_prompt"], label="prompt text"),
    "prompt_harmful": _find_col(necent_df,
        ["prompt_harmful", "is_harmful", "harmful", "label"], label="prompt_harmful"),
    "jailbreak": _find_col(necent_df,
        ["jailbreak", "is_jailbreak", "jailbreak_flag"], required=False, label="jailbreak flag"),
    "injection": _find_col(necent_df,
        ["injection", "is_injection", "prompt_injection"], required=False, label="injection flag"),
    "response_harmful": _find_col(necent_df,
        ["response_harmful", "is_response_harmful", "behavior_harmful"],
        required=False, label="response_harmful (behavioral label)"),
    "source": _find_col(necent_df,
        ["source", "category", "origin", "dataset_source"], required=False, label="source/category"),
    "row_id": _find_col(necent_df,
        ["id", "row_id", "uid", "index"], required=False, label="row id"),
    "model_name": _find_col(necent_df,
        ["model_name", "model", "generator_model", "response_model"],
        required=False, label="response-generating model name"),
    "response_text": _find_col(necent_df,
        ["response", "output", "model_response", "completion"],
        required=False, label="response text"),
}
print("Resolved Necent columns:")
for k, v in NECENT_COLS.items():
    print(f"  {k:>18s} -> {v}")


# ### Stage 2d — Behavioral Label Provenance Audit (Issue #2)
# 
# Before `response_harmful` is used anywhere as behavioral ground truth, we
# check whether it can plausibly be tied to `Qwen2.5-3B-Instruct` specifically,
# via a `model_name`/`source` column if one exists. We record what we find —
# we never assert Qwen-specific provenance we can't support.

# In[ ]:


PROVENANCE = {
    "response_harmful_column": NECENT_COLS["response_harmful"],
    "model_name_column_found": NECENT_COLS["model_name"] is not None,
    "response_text_column_found": NECENT_COLS["response_text"] is not None,
    "model_name_values": None,
    "matches_target_model": None,
    "verdict": None,
}

if NECENT_COLS["response_harmful"] is None:
    PROVENANCE["verdict"] = (
        "No response_harmful column found in Necent at load time. Behavioral "
        "evaluation stages below will report 100% missing labels rather than "
        "fabricate a label."
    )
elif NECENT_COLS["model_name"] is not None:
    vals = necent_df[NECENT_COLS["model_name"]].dropna().unique().tolist()
    PROVENANCE["model_name_values"] = [str(v) for v in vals][:20]
    target = CONFIG["model_id"].split("/")[-1].lower()
    matches = any(target in str(v).lower() or "qwen" in str(v).lower() for v in vals)
    PROVENANCE["matches_target_model"] = bool(matches)
    if matches and len(vals) == 1:
        PROVENANCE["verdict"] = (
            "response_harmful labels responses from a model matching "
            f"CONFIG['model_id'] ({CONFIG['model_id']}) — single model_name "
            "value, Qwen-family match. Even so, Necent's stored response text "
            "(if present) is NOT the same text as this notebook's own "
            "generations (different prompt template / sampling / decoding "
            "settings) — response_harmful is evidence about the DATASET's "
            "response, not a direct label for this notebook's outputs."
        )
    elif matches:
        PROVENANCE["verdict"] = (
            "model_name values include a Qwen-family match among MULTIPLE "
            "represented models. response_harmful is model-specific per row; "
            "using it uniformly as ground truth mixes provenance across rows. "
            "Treat behavioral results with this caveat."
        )
    else:
        PROVENANCE["verdict"] = (
            f"model_name values do NOT match CONFIG['model_id'] "
            f"({CONFIG['model_id']}): {PROVENANCE['model_name_values']}. "
            f"response_harmful reflects a DIFFERENT model's response — it is "
            f"NOT Qwen-specific behavioral ground truth. This notebook still "
            f"uses it as the task's designated behavioral label, but it is, "
            f"at best, a proxy for 'can this prompt elicit a harmful response "
            f"from *some* model' rather than 'does Qwen2.5-3B-Instruct "
            f"produce a harmful response'. Carried verbatim into the final "
            f"report."
        )
else:
    PROVENANCE["verdict"] = (
        "No model_name/source column found alongside response_harmful. Its "
        "provenance (which model produced the labeled response, under what "
        "conditions) is UNVERIFIED from the available schema. Used as the "
        "task's designated behavioral label, but this limitation is reported "
        "rather than silently assumed away."
    )

print("Behavioral label provenance audit:")
print(json.dumps(PROVENANCE, indent=2))
RUN_MANIFEST["notes"].append(f"response_harmful provenance verdict: {PROVENANCE['verdict']}")
with open(RUN_DIR / "response_harmful_provenance.json", "w") as f:
    json.dump(PROVENANCE, f, indent=2)
mark_stage_done("stage2d_provenance_audit", matches_target_model=PROVENANCE["matches_target_model"])
save_manifest(RUN_DIR)


# ## Stage 3 — Locked Split Loading + Validation
# 
# Read-only ground truth. Hash-verified, size-verified, zero-overlap-verified,
# OOD-separation-verified. Stops (raises) rather than resampling if the file is
# missing or fails any check.

# In[ ]:


# Stage 3a: load + hash the split file. STOP if missing.
if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Locked split file not found at {SPLIT_FILE}. STOPPING rather than "
        f"creating a new split. Place the canonical necent_nfw_splits_v2.json "
        f"at this path and re-run."
    )

split_file_bytes = SPLIT_FILE.read_bytes()
SPLIT_FILE_SHA256 = sha256_bytes(split_file_bytes)
print(f"Split file found: {SPLIT_FILE}")
print(f"Split file SHA256: {SPLIT_FILE_SHA256}")

with open(SPLIT_FILE) as f:
    split_doc = json.load(f)

declared_hash = split_doc.get("sha256") or split_doc.get("expected_sha256")
if declared_hash:
    doc_without_hash = {k: v for k, v in split_doc.items()
                         if k not in ("sha256", "expected_sha256")}
    recomputed = sha256_json_canonical(doc_without_hash)
    if recomputed != declared_hash:
        raise ValueError(
            f"Split file self-declared hash mismatch! declared={declared_hash} "
            f"recomputed={recomputed}. STOPPING — file may be corrupted/edited."
        )
    print("Split file self-declared SHA256 verified OK.")
else:
    print("Split file has no self-declared sha256; recording file-bytes hash "
          "above as the reproducibility anchor.")

RUN_MANIFEST["artifact_hashes"]["split_file_sha256"] = SPLIT_FILE_SHA256
mark_stage_done("stage3a_load_split_file", sha256=SPLIT_FILE_SHA256)
save_manifest(RUN_DIR)


# In[ ]:


# Stage 3b: extract the seven named splits as row-index arrays.
SPLIT_NAMES = list(CONFIG["expected_split_sizes"].keys())

splits_container = split_doc.get("splits", split_doc)
locked_splits = {}
for name in SPLIT_NAMES:
    if name not in splits_container:
        raise KeyError(
            f"Split '{name}' not present in {SPLIT_FILE}. Found keys: "
            f"{list(splits_container.keys())}. STOPPING."
        )
    idx = splits_container[name]
    if isinstance(idx, dict):
        idx = idx.get("indices") or idx.get("row_indices")
        if idx is None:
            raise KeyError(f"Split '{name}' entry has no 'indices'/'row_indices' key.")
    locked_splits[name] = np.array(sorted(idx), dtype=np.int64)

for name, idx in locked_splits.items():
    print(f"{name:>16s}: n={len(idx):5d}  min={idx.min():7d}  max={idx.max():7d}")


# In[ ]:


# ============================================================
# Stage 3c — Locked split integrity + OOD attack-technique check
# ============================================================

errors = []

print("=" * 80)
print("LOCKED SPLIT INTEGRITY CHECK")
print("=" * 80)

# ------------------------------------------------------------
# Actual Necent columns
# ------------------------------------------------------------

ATTACK_TECHNIQUE_COL = "attack_technique"
SOURCE_COL = "source"

required_columns = [
    ATTACK_TECHNIQUE_COL,
    SOURCE_COL,
]

missing_columns = [
    c for c in required_columns
    if c not in necent_df.columns
]

if missing_columns:
    raise RuntimeError(
        "Required Necent columns are missing:\n"
        f"{missing_columns}\n\n"
        f"Available columns:\n"
        f"{list(necent_df.columns)}"
    )

print(
    "Verified required columns:",
    required_columns,
)


# ------------------------------------------------------------
# 1. Expected split sizes
# ------------------------------------------------------------

for name, expected_n in (
    CONFIG["expected_split_sizes"].items()
):

    actual_n = len(
        locked_splits[name]
    )

    if actual_n != expected_n:

        errors.append(
            f"Split '{name}' has {actual_n} rows; "
            f"expected {expected_n}."
        )


# ------------------------------------------------------------
# 2. Index range
# ------------------------------------------------------------

min_index_seen = min(
    int(idx.min())
    for idx in locked_splits.values()
)

max_index_seen = max(
    int(idx.max())
    for idx in locked_splits.values()
)

if min_index_seen < 0:

    errors.append(
        f"Negative row index detected: "
        f"{min_index_seen}"
    )

if max_index_seen >= len(necent_df):

    errors.append(
        f"Split references row {max_index_seen}, "
        f"but Necent contains only "
        f"{len(necent_df):,} rows."
    )


# ------------------------------------------------------------
# 3. Zero overlap between every split
# ------------------------------------------------------------

for i, a in enumerate(SPLIT_NAMES):

    for b in SPLIT_NAMES[i + 1:]:

        overlap = np.intersect1d(
            locked_splits[a],
            locked_splits[b],
        )

        if len(overlap) > 0:

            errors.append(
                f"Split overlap: "
                f"{a} ∩ {b} = "
                f"{len(overlap)} rows."
            )


# ------------------------------------------------------------
# 4. Locked OOD attack techniques
# ------------------------------------------------------------

OOD_ATTACKS = set(
    split_doc[
        "ood_attack_techniques"
    ]
)

print()
print(
    "Locked OOD attack techniques:"
)

for attack in sorted(OOD_ATTACKS):

    print(
        f"  - {attack}"
    )


# ------------------------------------------------------------
# 5. Check OOD techniques do NOT occur outside test_ood
# ------------------------------------------------------------

non_ood_splits = [
    s
    for s in SPLIT_NAMES
    if s != "test_ood"
]

non_ood_indices = np.concatenate(
    [
        locked_splits[s]
        for s in non_ood_splits
    ]
)

non_ood_attack_values = set(
    necent_df.iloc[
        non_ood_indices
    ][
        ATTACK_TECHNIQUE_COL
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

non_ood_attack_values.discard("")

leaked_ood_techniques = (
    OOD_ATTACKS
    &
    non_ood_attack_values
)

if leaked_ood_techniques:

    errors.append(
        "OOD ATTACK-TECHNIQUE LEAKAGE: "
        f"{sorted(leaked_ood_techniques)}"
    )

else:

    print()
    print(
        "PASS: OOD attack techniques do not "
        "appear in non-OOD splits."
    )


# ------------------------------------------------------------
# 6. Check all expected OOD techniques are represented
# ------------------------------------------------------------

ood_indices = locked_splits[
    "test_ood"
]

ood_attack_values = set(
    necent_df.iloc[
        ood_indices
    ][
        ATTACK_TECHNIQUE_COL
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

ood_attack_values.discard("")

missing_ood_techniques = (
    OOD_ATTACKS
    -
    ood_attack_values
)

if missing_ood_techniques:

    errors.append(
        "OOD techniques missing from test_ood: "
        f"{sorted(missing_ood_techniques)}"
    )

else:

    print(
        "PASS: all locked OOD attack techniques "
        "are represented in test_ood."
    )


# ------------------------------------------------------------
# 7. Source overlap — REPORT ONLY
#
# Source overlap is allowed.
# OOD is defined by attack technique.
# ------------------------------------------------------------

id_sources = set(
    necent_df.iloc[
        non_ood_indices
    ][
        SOURCE_COL
    ]
    .dropna()
    .astype(str)
)

ood_sources = set(
    necent_df.iloc[
        ood_indices
    ][
        SOURCE_COL
    ]
    .dropna()
    .astype(str)
)

shared_sources = (
    id_sources &
    ood_sources
)

print()
print(
    f"Shared sources between ID and OOD: "
    f"{len(shared_sources)}"
)

if shared_sources:

    print(
        "NOTE: source overlap is allowed because "
        "the OOD axis is held-out attack technique."
    )


# ------------------------------------------------------------
# 8. Final fail-closed check
# ------------------------------------------------------------

if errors:

    print()
    print("=" * 80)
    print("SPLIT INTEGRITY FAILED")
    print("=" * 80)

    for error in errors:
        print(
            f"ERROR: {error}"
        )

    raise AssertionError(
        "Locked split integrity validation failed."
    )


print()
print("=" * 80)
print("ALL LOCKED SPLIT INTEGRITY CHECKS PASSED")
print("=" * 80)

print(
    f"Dataset rows: {len(necent_df):,}"
)

print(
    f"Dataset revision: "
    f"{DATASET_REVISION_SHA}"
)

print(
    f"Split SHA256: "
    f"{SPLIT_FILE_SHA256}"
)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

mark_stage_done(
    "stage3_split_validation",
    split_file_sha256=
        SPLIT_FILE_SHA256,
    dataset_revision=
        DATASET_REVISION_SHA,
    sizes={
        k: int(len(v))
        for k, v in locked_splits.items()
    },
    ood_attack_techniques=
        sorted(OOD_ATTACKS),
)

save_manifest(
    RUN_DIR
)


# ### Stage 3d — Explicit `behavioral_test` Immutability Assertion (Issue #10)
# 
# The pairwise check above already covers this, but this stage makes the
# guarantee an explicit, standalone gate — run immediately, well before any
# evaluation stage, and re-checked here on its own so it's visible in the
# manifest as its own completed step.

# In[ ]:


_other_splits = [s for s in SPLIT_NAMES if s != "behavioral_test"]
_bt = locked_splits["behavioral_test"]
_immutability_errors = []
for s in _other_splits:
    inter = np.intersect1d(_bt, locked_splits[s])
    if len(inter) > 0:
        _immutability_errors.append(f"behavioral_test overlaps '{s}' on {len(inter)} rows.")
if _immutability_errors:
    raise AssertionError("Test immutability check FAILED:\n- " + "\n- ".join(_immutability_errors))
print(f"Test immutability confirmed: behavioral_test (n={len(_bt)}) has zero "
      f"overlap with {_other_splits}.")
mark_stage_done("stage3d_test_immutability")
save_manifest(RUN_DIR)


# ## Stage 4 — Frozen Exp017 Probe Loading + Validation (Issue #11)
# 
# Cheap validation before any GPU work. Missing/malformed artifacts STOP the
# run — never a silent fallback to retraining or substitution.

# In[ ]:


from pathlib import Path

print("Drive contents:")
for p in Path("/content/drive").iterdir():
    print(" ", p)

print("\nSearching entire mounted Drive for Exp017 files...")

patterns = [
    "unsafe_intent__layer19.meta.json",
    "unsafe_intent__layer19.weight.npy",
]

for pattern in patterns:
    print(f"\nSearching: {pattern}")

    found = list(
        Path("/content/drive").rglob(pattern)
    )

    if found:
        for p in found:
            print("FOUND:", p)
    else:
        print("NOT FOUND")


# In[ ]:


# ============================================================
# PROBE ARTIFACT AUDIT — CANONICAL EXP017 FORMAT
# ============================================================

from pathlib import Path
import json
import numpy as np

PROBE_DIR = Path(
    "/content/drive/MyDrive/NFW-001/artifacts/exp017"
)

EXPECTED_LAYERS = [19, 20, 21, 22]

print("=" * 80)
print("EXP017 PROBE ARTIFACT AUDIT")
print("=" * 80)

if not PROBE_DIR.exists():
    raise FileNotFoundError(
        f"Probe directory does not exist:\n{PROBE_DIR}"
    )

print(f"Probe directory:\n{PROBE_DIR}\n")

for L in EXPECTED_LAYERS:

    meta_path = (
        PROBE_DIR /
        f"unsafe_intent__layer{L}.meta.json"
    )

    weight_path = (
        PROBE_DIR /
        f"unsafe_intent__layer{L}.weight.npy"
    )

    print(f"\n--- Layer {L} ---")

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing metadata:\n{meta_path}"
        )

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing weights:\n{weight_path}"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    weights = np.load(
        weight_path,
        allow_pickle=False,
    )

    print(f"Metadata: {meta_path.name}")
    print(f"Weights:  {weight_path.name}")
    print(f"Weight shape: {weights.shape}")
    print(f"Weight dtype: {weights.dtype}")
    print(
        f"Finite weights: "
        f"{np.isfinite(weights).all()}"
    )

    print("Metadata:")
    print(
        json.dumps(
            meta,
            indent=2,
            default=str,
        )
    )

print()
print("=" * 80)
print("PROBE ARTIFACT AUDIT COMPLETE")
print("=" * 80)


# In[ ]:


PROBE_DIR = Path(
    "/content/drive/MyDrive/NFW-001/artifacts/exp017/"
)


# In[ ]:


from pathlib import Path

PROBE_DIR = Path(
    "/content/drive/MyDrive/NFW-001/artifacts/exp017"
)

print("Directory exists:", PROBE_DIR.exists())
print("Directory:", PROBE_DIR)

print("\nACTUAL FILES:")
for p in sorted(PROBE_DIR.iterdir()):
    print(" ", p.name)

print("\nExpected files:")
for L in [19, 20, 21, 22]:
    print(
        L,
        "META:",
        (
            PROBE_DIR /
            f"unsafe_intent__layer{L}.meta.json"
        ).exists(),
        "WEIGHT:",
        (
            PROBE_DIR /
            f"unsafe_intent__layer{L}.weight.npy"
        ).exists(),
    )


# In[ ]:


# ============================================================
# 4a — LOAD FROZEN EXP017 PROBES
# Canonical Exp017 format
#
# score = dot(raw_coef, activation) + intercept
# ============================================================

from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

PROBE_DIR = Path(
    "/content/drive/MyDrive/NFW-001/artifacts/exp017"
)

REQUIRED_LAYERS = [19, 20, 21, 22]

print("=" * 80)
print("4a — LOADING FROZEN EXP017 PROBES")
print("=" * 80)

if not PROBE_DIR.exists():
    raise FileNotFoundError(
        f"Probe directory does not exist:\n{PROBE_DIR}"
    )

FROZEN_PROBES = {}

for layer in REQUIRED_LAYERS:

    meta_path = (
        PROBE_DIR /
        f"unsafe_intent__layer{layer}.meta.json"
    )

    weight_path = (
        PROBE_DIR /
        f"unsafe_intent__layer{layer}.weight.npy"
    )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing Exp017 metadata:\n{meta_path}"
        )

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing Exp017 weights:\n{weight_path}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # --------------------------------------------------------
    # Load raw LogisticRegression coefficient
    # --------------------------------------------------------

    coef = np.asarray(
        np.load(
            weight_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    ).reshape(-1)

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    declared_layer = int(
        meta["layer_idx"]
    )

    if declared_layer != layer:
        raise AssertionError(
            f"Layer mismatch: "
            f"filename={layer}, "
            f"metadata={declared_layer}"
        )

    expected_shape = tuple(
        meta["weight_shape"]
    )

    if tuple(coef.shape) != expected_shape:
        raise AssertionError(
            f"Layer {layer}: weight shape "
            f"{coef.shape} != metadata "
            f"{expected_shape}"
        )

    if meta["weight_normalized"] is not False:
        raise AssertionError(
            f"Layer {layer}: "
            "Exp017 weight_normalized is not False."
        )

    if meta["scoring"] != (
        "dot(raw_coef, activation) + intercept"
    ):
        raise AssertionError(
            f"Layer {layer}: unexpected scoring rule:\n"
            f"{meta['scoring']}"
        )

    if not np.isfinite(coef).all():
        raise AssertionError(
            f"Layer {layer}: non-finite weights."
        )

    bias = float(
        meta["bias"]
    )

    if not np.isfinite(bias):
        raise AssertionError(
            f"Layer {layer}: non-finite bias."
        )

    # --------------------------------------------------------
    # Verify stored norm
    # --------------------------------------------------------

    actual_norm = float(
        np.linalg.norm(coef)
    )

    stored_norm = float(
        meta["weight_norm"]
    )

    if not np.isclose(
        actual_norm,
        stored_norm,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise AssertionError(
            f"Layer {layer}: weight norm mismatch.\n"
            f"Actual:  {actual_norm}\n"
            f"Stored:  {stored_norm}"
        )

    # --------------------------------------------------------
    # Create probe object compatible with the notebook
    # --------------------------------------------------------

    probe = SimpleNamespace()

    # Identity
    probe.layer = layer
    probe.layer_idx = layer

    # Dimensions
    probe.hidden_dim = int(
        coef.size
    )
    probe.weight_dim = int(
        coef.size
    )

    # Exact raw probe
    probe.coef = coef
    probe.weights = coef

    # Exact LogisticRegression intercept
    probe.bias = bias
    probe.intercept = bias

    # Metadata
    probe.meta = meta
    probe.meta_path = str(
        meta_path
    )
    probe.weight_path = str(
        weight_path
    )
    probe.source_path = str(
        weight_path
    )

    # Exp017 semantics
    probe.weight_normalized = False
    probe.weight_norm = actual_norm

    probe.scoring_rule = (
        "dot(raw_coef, activation) + intercept"
    )

    # Original Exp017 threshold.
    # This is provenance only; NFW-001 recalibrates
    # thresholds on Necent.
    probe.exp017_threshold = float(
        meta["threshold"]
    )
    probe.threshold = probe.exp017_threshold
    FROZEN_PROBES[layer] = probe

    print(
        f"Layer {layer}: "
        f"hidden_dim={probe.hidden_dim}, "
        f"norm={probe.weight_norm:.9f}, "
        f"bias={probe.bias:.10f}, "
        f"Exp017 threshold="
        f"{probe.exp017_threshold}"
    )


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

if sorted(
    FROZEN_PROBES.keys()
) != REQUIRED_LAYERS:

    raise AssertionError(
        f"Loaded probe layers "
        f"{sorted(FROZEN_PROBES.keys())} "
        f"!= required layers "
        f"{REQUIRED_LAYERS}"
    )

print()
print(
    "FROZEN_PROBES layers:",
    sorted(FROZEN_PROBES.keys())
)

print(
    "Scoring:",
    "dot(raw_coef, activation) + intercept"
)

print(
    "Normalization:",
    "none"
)

print(
    "Retraining/fallback:",
    "disabled"
)

print()
print("=" * 80)
print("4a — FROZEN EXP017 PROBES LOADED")
print("=" * 80)

RUN_MANIFEST[
    "probe_sources"
] = {
    str(L): {
        "weight":
            FROZEN_PROBES[L].weight_path,
        "metadata":
            FROZEN_PROBES[L].meta_path,
        "scoring":
            FROZEN_PROBES[L].scoring_rule,
        "bias":
            FROZEN_PROBES[L].bias,
        "exp017_threshold":
            FROZEN_PROBES[L].exp017_threshold,
    }
    for L in REQUIRED_LAYERS
}

RUN_MANIFEST[
    "fallback_probes_used"
] = []

mark_stage_done(
    "stage4a_probes_loaded",
    probe_layers=REQUIRED_LAYERS,
)

save_manifest(
    RUN_DIR
)


# In[ ]:


# ============================================================
# 4b — VALIDATE FROZEN EXP017 PROBES AGAINST QWEN
# ============================================================

import numpy as np

print("=" * 80)
print("4b — FROZEN EXP017 PROBE VALIDATION")
print("=" * 80)

REQUIRED_LAYERS = [19, 20, 21, 22]

if "QWEN_MODEL" not in globals():
    raise RuntimeError(
        "QWEN_MODEL is not loaded. "
        "Run the Qwen model-loading cell first."
    )

hidden_size = int(
    QWEN_MODEL.config.hidden_size
)

num_layers = int(
    QWEN_MODEL.config.num_hidden_layers
)

print(
    f"Qwen/Qwen2.5-3B-Instruct: "
    f"hidden_size={hidden_size}, "
    f"num_hidden_layers={num_layers}"
)

if sorted(
    FROZEN_PROBES.keys()
) != REQUIRED_LAYERS:

    raise AssertionError(
        f"Loaded probe layers "
        f"{sorted(FROZEN_PROBES.keys())} "
        f"!= required layers "
        f"{REQUIRED_LAYERS}"
    )

probe_errors = []

for layer in REQUIRED_LAYERS:

    probe = FROZEN_PROBES[
        layer
    ]

    # --------------------------------------------------------
    # Layer identity
    # --------------------------------------------------------

    if probe.layer != layer:

        probe_errors.append(
            f"Layer {layer}: metadata declares "
            f"layer={probe.layer}."
        )

    # --------------------------------------------------------
    # Dimension
    # --------------------------------------------------------

    if probe.coef.shape != (
        hidden_size,
    ):

        probe_errors.append(
            f"Layer {layer}: coefficient shape "
            f"{probe.coef.shape} != "
            f"({hidden_size},)."
        )

    # --------------------------------------------------------
    # Finite values
    # --------------------------------------------------------

    if not np.isfinite(
        probe.coef
    ).all():

        probe_errors.append(
            f"Layer {layer}: coefficient contains "
            "NaN/Inf."
        )

    if not np.isfinite(
        probe.bias
    ):

        probe_errors.append(
            f"Layer {layer}: bias is non-finite."
        )

    # --------------------------------------------------------
    # Exact scoring metadata
    # --------------------------------------------------------

    if probe.scoring_rule != (
        "dot(raw_coef, activation) + intercept"
    ):

        probe_errors.append(
            f"Layer {layer}: unexpected scoring rule "
            f"{probe.scoring_rule!r}."
        )

    if probe.weight_normalized is not False:

        probe_errors.append(
            f"Layer {layer}: probe marked as normalized."
        )

    print(
        f"Layer {layer}: "
        f"dim={probe.coef.size}, "
        f"norm={np.linalg.norm(probe.coef):.9f}, "
        f"bias={probe.bias:.10f}, "
        f"finite=True"
    )


# ------------------------------------------------------------
# Fail closed
# ------------------------------------------------------------

if probe_errors:

    raise AssertionError(
        "Frozen Exp017 probe validation FAILED:\n- "
        +
        "\n- ".join(probe_errors)
    )


print()
print("=" * 80)
print("ALL FROZEN EXP017 PROBE VALIDATION CHECKS PASSED")
print("=" * 80)

print(
    "Required layers:",
    REQUIRED_LAYERS
)

print(
    "Hidden size:",
    hidden_size
)

print(
    "Scoring rule:",
    "dot(raw_coef, activation) + intercept"
)

print(
    "Probe normalization:",
    "none"
)

print(
    "Fallback/retraining:",
    "DISABLED"
)

mark_stage_done(
    "stage4b_probe_validation",
    probe_layers=REQUIRED_LAYERS,
    hidden_size=hidden_size,
)

save_manifest(
    RUN_DIR
)


# In[ ]:


# ============================================================
# EXP017 METADATA — EXACT SCORING AUDIT
# ============================================================

import json

print("=" * 80)
print("EXP017 PROBE METADATA AUDIT")
print("=" * 80)

for layer in [19, 20, 21, 22]:

    meta_path = (
        PROBE_DIR /
        f"unsafe_intent__layer{layer}.meta.json"
    )

    print()
    print("=" * 80)
    print(f"LAYER {layer}")
    print(f"FILE: {meta_path}")
    print("=" * 80)

    if not meta_path.exists():

        raise FileNotFoundError(
            f"Missing metadata file:\n{meta_path}"
        )

    with open(
        meta_path,
        "r",
    ) as f:

        meta = json.load(f)

    print(
        json.dumps(
            meta,
            indent=2,
            default=str,
        )
    )

print()
print("=" * 80)
print("METADATA AUDIT COMPLETE")
print("=" * 80)


# In[ ]:


# ============================================================
# 4c — CANONICAL EXP017 PROBE SCORING
# ============================================================

def probe_score(layer, activation):
    """
    Exact Exp017 scoring rule:

        score = dot(raw_coef, activation) + intercept

    No normalization.
    No scaler.
    No thresholding here.

    Thresholding is handled separately by NFW-001
    calibration.
    """

    if layer not in FROZEN_PROBES:
        raise KeyError(
            f"No frozen Exp017 probe for layer {layer}."
        )

    probe = FROZEN_PROBES[layer]

    x = np.asarray(
        activation,
        dtype=np.float32,
    )

    w = np.asarray(
        probe.coef,
        dtype=np.float32,
    )

    if x.shape[-1] != probe.hidden_dim:
        raise ValueError(
            f"Layer {layer}: activation hidden dimension "
            f"{x.shape[-1]} != probe dimension "
            f"{probe.hidden_dim}."
        )

    # Supports:
    #   [hidden_dim]
    #   [batch, hidden_dim]
    #   [tokens, hidden_dim]
    scores = np.matmul(
        x,
        w,
    ) + float(
        probe.intercept
    )

    return scores


print("=" * 80)
print("4c — CANONICAL EXP017 SCORING FUNCTION")
print("=" * 80)

print(
    "score(layer, activation) = "
    "dot(raw_coef, activation) + intercept"
)

print(
    "Normalization: none"
)

print(
    "Thresholding: not performed here"
)

# ------------------------------------------------------------
# Tiny synthetic sanity check
# ------------------------------------------------------------

for layer in REQUIRED_LAYERS:

    probe = FROZEN_PROBES[layer]

    test_activation = np.zeros(
        probe.hidden_dim,
        dtype=np.float32,
    )

    expected = float(
        probe.intercept
    )

    actual = float(
        probe_score(
            layer,
            test_activation,
        )
    )

    if not np.isclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise AssertionError(
            f"Layer {layer}: scoring sanity check failed.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

print()
print(
    "PASS: zero-vector scoring reproduces "
    "the stored Exp017 intercept for all layers."
)

mark_stage_done(
    "stage4c_probe_scoring",
)

save_manifest(
    RUN_DIR
)


# In[ ]:


# ============================================================
# 4d — NUMERICAL PROBE SCORING SANITY CHECK
# ============================================================

print("=" * 80)
print("4d — NUMERICAL SCORING SANITY CHECK")
print("=" * 80)

rng = np.random.default_rng(
    GLOBAL_SEED
)

for layer in REQUIRED_LAYERS:

    probe = FROZEN_PROBES[layer]

    x = rng.normal(
        size=probe.hidden_dim
    ).astype(
        np.float32
    )

    manual = float(
        np.dot(
            probe.coef,
            x,
        )
        +
        probe.intercept
    )

    function_result = float(
        probe_score(
            layer,
            x,
        )
    )

    if not np.isclose(
        manual,
        function_result,
        rtol=1e-6,
        atol=1e-7,
    ):

        raise AssertionError(
            f"Layer {layer}: manual scoring != "
            "probe_score().\n"
            f"Manual:   {manual}\n"
            f"Function: {function_result}"
        )

    print(
        f"Layer {layer}: "
        f"manual={manual:.8f}, "
        f"probe_score={function_result:.8f}, "
        f"PASS"
    )

print()
print("=" * 80)
print("SCORING SANITY CHECK PASSED")
print("=" * 80)


# ## Stage 5 — Dataset Composition Report

# In[ ]:


def composition_row(name, idx):
    sub = necent_df.iloc[idx]
    row = {"split": name, "n": len(sub)}
    ph_col = NECENT_COLS["prompt_harmful"]
    row["prompt_harmful_rate"] = float(sub[ph_col].astype(float).mean()) if ph_col else np.nan
    if NECENT_COLS["jailbreak"]:
        row["jailbreak_rate"] = float(sub[NECENT_COLS["jailbreak"]].astype(float).mean())
    if NECENT_COLS["injection"]:
        row["injection_rate"] = float(sub[NECENT_COLS["injection"]].astype(float).mean())
    if NECENT_COLS["response_harmful"]:
        rh = sub[NECENT_COLS["response_harmful"]]
        row["response_harmful_missing_n"] = int(rh.isna().sum())
        row["response_harmful_labeled_n"] = int(rh.notna().sum())
        row["response_harmful_rate_of_labeled"] = (
            float(rh.dropna().astype(float).mean()) if rh.notna().sum() > 0 else np.nan
        )
    else:
        row["response_harmful_missing_n"] = len(sub)
        row["response_harmful_labeled_n"] = 0
        row["response_harmful_rate_of_labeled"] = np.nan
    if NECENT_COLS["source"]:
        row["n_distinct_sources"] = int(sub[NECENT_COLS["source"]].nunique())
    if NECENT_COLS["model_name"]:
        row["n_distinct_response_models"] = int(sub[NECENT_COLS["model_name"]].nunique())
    return row

composition_rows = [composition_row(name, idx) for name, idx in locked_splits.items()]
composition_df = pd.DataFrame(composition_rows).set_index("split")
pd.set_option("display.width", 120)
print(composition_df.to_string())

comp_path = RUN_DIR / "dataset_composition.csv"
composition_df.to_csv(comp_path)
print(f"\nSaved composition report to {comp_path}")

total_missing_behavioral = int(composition_df["response_harmful_missing_n"].sum())
print(f"\nTOTAL missing response_harmful labels across all splits: {total_missing_behavioral}")
if total_missing_behavioral > 0:
    RUN_MANIFEST["notes"].append(
        f"{total_missing_behavioral} rows lack a response_harmful label across all splits; "
        f"behavioral stages restrict to the labeled subset, never impute."
    )
mark_stage_done("stage5_composition_report", total_missing_behavioral=total_missing_behavioral)
save_manifest(RUN_DIR)


# ## Stage 6 — Model Loading, Activation-Location Verification, Extraction Functions
# 
# Loads the frozen model once. **New in this revision:** before defining the
# streaming (per-token) hook-based extraction, we empirically verify how it
# relates to `hidden_states[L]` (the convention used by prompt-level
# extraction and, presumably, by Exp017's original training) — this resolves
# Issue #3 by proof rather than assumption, and the fix is applied to the hook
# installer used by every later stage.

# In[ ]:


# ============================================================
# Stage 6a — Resolve + PIN exact Qwen model revision
# ============================================================

import torch

from huggingface_hub import model_info
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

set_all_seeds(
    GLOBAL_SEED
)

# ------------------------------------------------------------
# Resolve exact model commit BEFORE loading weights
# ------------------------------------------------------------

print(
    f"Resolving model revision for "
    f"{CONFIG['model_id']}..."
)

try:

    _model_info = model_info(
        CONFIG["model_id"]
    )

    RESOLVED_MODEL_REVISION = (
        getattr(
            _model_info,
            "sha",
            None,
        )
    )

except Exception as e:

    raise RuntimeError(
        "Could not resolve the exact Qwen model "
        "Hub revision. Refusing to run an activation "
        "experiment against an unpinned model.\n"
        f"Error: {e}"
    )


if not RESOLVED_MODEL_REVISION:

    raise RuntimeError(
        "Hugging Face returned no model commit SHA."
    )


# ------------------------------------------------------------
# Pin it in CONFIG
# ------------------------------------------------------------

CONFIG[
    "model_revision"
] = RESOLVED_MODEL_REVISION

print(
    "PINNED Qwen revision:"
)

print(
    CONFIG["model_revision"]
)


# ------------------------------------------------------------
# Select dtype/device
# ------------------------------------------------------------

_DTYPE_MAP = {

    "bfloat16":
        torch.bfloat16,

    "float16":
        torch.float16,

    "float32":
        torch.float32,
}

_dtype = _DTYPE_MAP.get(
    CONFIG["dtype"],
    torch.float32,
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if (
    _dtype == torch.bfloat16
    and torch.cuda.is_available()
    and not torch.cuda.is_bf16_supported()
):

    print(
        "bfloat16 unsupported on this GPU; "
        "falling back to float16."
    )

    _dtype = torch.float16


print()
print(
    f"Loading frozen model:\n"
    f"  model:    {CONFIG['model_id']}\n"
    f"  revision: {CONFIG['model_revision']}\n"
    f"  device:   {DEVICE}\n"
    f"  dtype:    {_dtype}"
)


# ------------------------------------------------------------
# Load tokenizer from EXACT revision
# ------------------------------------------------------------

QWEN_TOKENIZER = (
    AutoTokenizer.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG[
            "model_revision"
        ],
    )
)

# BUG FIX: Qwen2.5 has no pad_token by default, so batched extraction
# (extract_last_token_activations, Stage 6c) would crash. Left-padding
# means the last real token of every sequence is always at position -1,
# which is what the batched extractor relies on.
if QWEN_TOKENIZER.pad_token is None:
    QWEN_TOKENIZER.pad_token = QWEN_TOKENIZER.eos_token
QWEN_TOKENIZER.padding_side = "left"


# ------------------------------------------------------------
# Load model from EXACT revision
# ------------------------------------------------------------

# OOM FIX: output_hidden_states=True was previously baked into the model
# config here, which makes EVERY forward call anywhere in this notebook
# (including every single-token step of the ~256-step generation loops in
# Stages 7, 10c, and 14 -- thousands of calls per run) materialize and
# return the full stack of hidden states for all ~36 layers, even though
# those generation loops only need the few layers captured by hooks. This
# was the single largest source of wasted GPU memory/fragmentation in the
# notebook. It is removed from the model default; the two places that
# actually need hidden_states (Stage 6a2 verification, and
# extract_last_token_activations below) request it explicitly per call.
#
# Also added: low_cpu_mem_usage=True (avoids holding a full extra CPU copy
# of the weights during load) and attn_implementation="sdpa" (memory-
# efficient attention; the default "eager" implementation materializes
# full attention score matrices, which is costly on the longer
# jailbreak/injection prompts in Necent).
QWEN_MODEL = (
    AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG[
            "model_revision"
        ],
        torch_dtype=_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    .to(DEVICE)
)


# ------------------------------------------------------------
# Freeze
# ------------------------------------------------------------

QWEN_MODEL.eval()

for p in QWEN_MODEL.parameters():

    p.requires_grad_(False)


# ------------------------------------------------------------
# Verify actual loaded commit
# ------------------------------------------------------------

actual_loaded_revision = (
    getattr(
        QWEN_MODEL.config,
        "_commit_hash",
        None,
    )
)

print()
print(
    "Model loaded."
)

print(
    "Requested revision:",
    CONFIG["model_revision"],
)

print(
    "Loaded commit:",
    actual_loaded_revision,
)


if (
    actual_loaded_revision
    and
    actual_loaded_revision
    != CONFIG["model_revision"]
):

    raise RuntimeError(
        "Loaded Qwen commit does not match "
        "the requested pinned revision.\n"
        f"Requested: {CONFIG['model_revision']}\n"
        f"Loaded:    {actual_loaded_revision}"
    )


# ------------------------------------------------------------
# Record exact model identity
# ------------------------------------------------------------

# BUG FIX: QWEN_HIDDEN_SIZE is referenced by the Stage 18 forensic report
# but was never defined anywhere in the notebook (a third undefined-name
# bug, alongside extract_last_token_activations and ensemble_decision /
# exceeded_with_persistence). Defining it here, once, right after the
# model is loaded and frozen.
QWEN_HIDDEN_SIZE = int(QWEN_MODEL.config.hidden_size)

RUN_MANIFEST[
    "artifact_hashes"
][
    "model_id"
] = CONFIG[
    "model_id"
]

RUN_MANIFEST[
    "artifact_hashes"
][
    "model_revision"
] = (
    CONFIG["model_revision"]
)

RUN_MANIFEST[
    "artifact_hashes"
][
    "model_loaded_commit"
] = (
    actual_loaded_revision
)

RUN_MANIFEST[
    "notes"
].append(
    "Qwen model revision was resolved and pinned "
    "before model loading."
)

mark_stage_done(
    "stage6a_model_loaded",
    model_id=CONFIG["model_id"],
    model_revision=CONFIG[
        "model_revision"
    ],
    device=DEVICE,
    dtype=str(_dtype),
)

save_manifest(
    RUN_DIR
)


# In[ ]:


# Stage 6a2 (Issue #3): activation-location verification.
#
# HF's stated convention for a Llama-style decoder (Qwen2 included) is:
#   hidden_states[0]  = embedding output (input to decoder layer 0)
#   hidden_states[i]  = OUTPUT of decoder layer (i - 1), for i = 1..num_layers
#
# Prompt-level extraction (Stage 6b below) indexes hidden_states[L] directly
# for L in {19,20,21,22}. Streaming/generation-time extraction instead reads
# activations via a forward hook placed on a specific decoder layer module —
# if that hook is placed on decoder_layers[L] (as the previous notebook
# version did), it captures the OUTPUT of decoder layer L, i.e. the same
# quantity as hidden_states[L+1], NOT hidden_states[L]. That is a real
# off-by-one between the two extraction paths and it would silently misalign
# streaming risk scores against the frozen Exp017 probes (which were trained
# against hidden_states[L]-convention prompt-level activations).
#
# We verify the correspondence empirically here, and the fix (hook
# decoder_layers[L-1] to match hidden_states[L]) is applied in
# _install_layer_hooks below. If verification fails for this model/revision,
# we STOP rather than guess an offset.

_verify_buffer = {}
def _verify_hook(idx):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        _verify_buffer[idx] = hs.detach()
    return hook

_verify_text = "The quick brown fox jumps over the lazy dog."
_venc = QWEN_TOKENIZER(_verify_text, return_tensors="pt").to(DEVICE)
_decoder_layers = QWEN_MODEL.model.layers

_vhandles = [_decoder_layers[i].register_forward_hook(_verify_hook(i))
             for i in range(len(_decoder_layers))]
with torch.no_grad():
    _vout = QWEN_MODEL(**_venc, output_hidden_states=True)
for h in _vhandles:
    h.remove()

_mismatches = []
for L in CONFIG["probe_layers"]:
    hs_L = _vout.hidden_states[L]
    hook_out_Lminus1 = _verify_buffer[L - 1]
    if not torch.allclose(hs_L.float(), hook_out_Lminus1.float(), atol=1e-4, rtol=1e-4):
        _mismatches.append(L)

if _mismatches:
    raise AssertionError(
        f"Activation-location verification FAILED for layers {_mismatches}: "
        f"hidden_states[L] does not match the output of decoder_layers[L-1] "
        f"for this model/revision. Refusing to guess an offset — inspect "
        f"QWEN_MODEL.model.layers before proceeding."
    )

print("Activation-location verification PASSED: hidden_states[L] == output of "
      f"decoder_layers[L-1] for layers {CONFIG['probe_layers']}. Streaming "
      "hooks below target decoder_layers[L-1] to match Stage 6b's convention.")

RUN_MANIFEST["notes"].append(
    "Activation-location verified: hidden_states[L] == decoder_layers[L-1] output "
    "for this model/revision; streaming hooks fixed to target L-1."
)
mark_stage_done("stage6a2_activation_location_verified", layers_checked=CONFIG["probe_layers"])
save_manifest(RUN_DIR)
del _verify_buffer, _venc, _vout, _vhandles


# In[ ]:


# ============================================================
# 6b — CANONICAL PROBE RISK SCORING
# ============================================================

def probe_risk_scores(activations_by_layer):
    """
    Exact Exp017 probe scoring.

    For each layer:

        logit = dot(raw_coef, activation) + intercept

    Then convert the logit to a sigmoid probability.

    IMPORTANT:
    - No scaler
    - No activation normalization
    - No probe normalization
    - No fallback/retraining
    - Exp017 thresholds are NOT applied here
    """

    scores = {}

    for layer in REQUIRED_LAYERS:

        if layer not in FROZEN_PROBES:
            raise KeyError(
                f"Missing frozen Exp017 probe for layer {layer}."
            )

        if layer not in activations_by_layer:
            raise KeyError(
                f"Missing activations for layer {layer}."
            )

        probe = FROZEN_PROBES[layer]

        x = np.asarray(
            activations_by_layer[layer],
            dtype=np.float32,
        )

        w = np.asarray(
            probe.coef,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Validate activation dimension
        # ----------------------------------------------------

        if x.shape[-1] != probe.hidden_dim:

            raise ValueError(
                f"Layer {layer}: activation dimension "
                f"{x.shape[-1]} != probe dimension "
                f"{probe.hidden_dim}."
            )

        # ----------------------------------------------------
        # Exact Exp017 scoring
        # ----------------------------------------------------

        logit = (
            np.matmul(
                x,
                w,
            )
            +
            float(
                probe.intercept
            )
        )

        # ----------------------------------------------------
        # Logistic probability
        # ----------------------------------------------------

        logit = np.clip(
            logit,
            -50,
            50,
        )

        risk = 1.0 / (
            1.0 +
            np.exp(-logit)
        )

        if not np.isfinite(risk).all():

            raise ValueError(
                f"Layer {layer}: non-finite probe risk scores."
            )

        scores[layer] = risk

    return scores


print("=" * 80)
print("CANONICAL EXP017 PROBE SCORING INSTALLED")
print("=" * 80)

print(
    "logit = dot(raw_coef, activation) + intercept"
)

print(
    "risk = sigmoid(logit)"
)

print(
    "Scaler: NONE"
)

print(
    "Activation normalization: NONE"
)

print(
    "Probe normalization: NONE"
)


# In[ ]:


# Stage 6b3 (bug fix): ensemble_decision / exceeded_with_persistence.
# Both are called throughout Stages 6c, 8, 10, 11, 14 but were MISSING from
# the notebook entirely -- this is the second undefined-function NameError
# (extract_last_token_activations was the first). Semantics are reverse-
# engineered from every call site plus the inline duplicate of this same
# logic inside firewall_monitored_generate, so behavior matches exactly.
def ensemble_decision(scores_by_layer, thresholds):
    """k_of_n ensemble over per-layer probe scores.

    scores_by_layer: dict {layer: np.array of per-row risk scores}
    thresholds:       dict {layer: threshold}, e.g. PROMPT_THRESHOLDS /
                       BEHAVIORAL_THRESHOLDS / each probe's own exp017
                       default threshold.

    Returns (votes, exceeded):
      votes:    int array, per-row count of layers whose score >= threshold
      exceeded: bool array, votes >= CONFIG["vote_k"]  (CONFIG["voting"]
                is "k_of_n"; this is the only voting scheme implemented,
                matching every other stage in this notebook)
    """
    layers = list(scores_by_layer.keys())
    n = len(np.asarray(scores_by_layer[layers[0]]))
    votes = np.zeros(n, dtype=np.int64)
    for l in layers:
        s = np.asarray(scores_by_layer[l])
        votes += (s >= thresholds[l]).astype(np.int64)
    exceeded = votes >= CONFIG["vote_k"]
    return votes, exceeded


def exceeded_with_persistence(step_exceeded, k):
    """True iff `step_exceeded` (bool array over decision steps) contains a
    run of >= k consecutive True values -- the same persistence rule used
    inline by firewall_monitored_generate (exceed_run counter, block once
    exceed_run >= k)."""
    run = 0
    for v in step_exceeded:
        run = run + 1 if v else 0
        if run >= k:
            return True
    return False

print("Defined: ensemble_decision, exceeded_with_persistence (both were previously undefined).")


# In[ ]:


# Stage 6b2 (OOM/bug fix): batched, memory-safe last-token activation
# extraction. This function is called by Stage 6c below but was MISSING
# from the notebook entirely -- every prompt-level evaluation call would
# have raised NameError before this fix.
#
# Uses the prompt-level convention (hidden_states[L], not the
# decoder_layers[L-1] hook used by streaming generation -- see Stage 6a2).
# Left-padding (set on the tokenizer in Stage 6a) means the last real token
# of every sequence in a batch sits at position -1, so no per-row unpadding
# logic is needed.
#
# Memory-safety measures (these are the actual OOM fixes):
#   - output_hidden_states=True is requested per-call, not baked into the
#     model config (see the Stage 6a fix note).
#   - use_cache=False: no KV cache is needed for a single forward pass.
#   - each batch's activations are moved to CPU/numpy immediately; nothing
#     stays on GPU past the batch that produced it.
#   - on a CUDA OOM, batch_size is halved and the batch is retried instead
#     of crashing the whole run.
#   - torch.cuda.empty_cache() + gc.collect() run periodically to avoid
#     allocator fragmentation building up over thousands of batches.
def extract_last_token_activations(prompts, batch_size=8, layers=None):
    layers = layers or CONFIG["probe_layers"]
    results = {l: [] for l in layers}
    n = len(prompts)
    i = 0
    batches_done = 0
    while i < n:
        bs = min(batch_size, n - i)
        batch_prompts = prompts[i:i + bs]
        try:
            enc = QWEN_TOKENIZER(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(DEVICE)
            with torch.no_grad():
                out = QWEN_MODEL(**enc, output_hidden_states=True, use_cache=False)
            for l in layers:
                # left-padding -> last real token is always at position -1
                results[l].append(
                    out.hidden_states[l][:, -1, :].detach().float().cpu().numpy()
                )
            del out, enc
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if bs == 1:
                raise RuntimeError(
                    f"CUDA OOM extracting activations even at batch_size=1 "
                    f"(prompt index {i}). Try CONFIG['max_new_tokens'] lower, "
                    f"a shorter max_length, or float16 instead of bfloat16."
                ) from e
            print(f"  [extract_last_token_activations] CUDA OOM at batch_size={bs}; "
                  f"halving batch_size and retrying from prompt {i}.")
            batch_size = max(1, bs // 2)
            torch.cuda.empty_cache()
            gc.collect()
            continue

        i += bs
        batches_done += 1
        if batches_done % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return {l: np.concatenate(v, axis=0) for l, v in results.items()}

print("Defined: extract_last_token_activations (batched, OOM-safe, was previously undefined).")


# In[ ]:


# Stage 6c: prompt-level probe evaluation over every split (resumable
# per-split checkpoint). Note: "exp017_default_flagged" below uses each
# probe's OWN bundled Exp017 threshold, purely as a diagnostic reference
# point — it is NOT PROMPT_THRESHOLDS or BEHAVIORAL_THRESHOLDS (those are
# calibrated in Stage 10a/10d and used for all frozen decisions below).
PROMPT_LEVEL_DIR = RUN_DIR / "prompt_level"
PROMPT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)

def run_prompt_level_eval(split_name, idx, batch_size=8):
    ckpt_path = PROMPT_LEVEL_DIR / f"{split_name}.parquet"
    if ckpt_path.exists() and not CONFIG["force_recompute"]:
        print(f"[{split_name}] loading existing checkpoint ({ckpt_path.name})")
        return pd.read_parquet(ckpt_path)

    sub = necent_df.iloc[idx].reset_index(drop=False).rename(columns={"index": "necent_row_index"})
    prompts = sub[NECENT_COLS["prompt"]].astype(str).tolist()
    print(f"[{split_name}] extracting activations for {len(prompts)} prompts...")
    acts = extract_last_token_activations(prompts, batch_size=batch_size)
    scores = probe_risk_scores(acts)

    result = sub[["necent_row_index", NECENT_COLS["prompt_harmful"]]].copy()
    result = result.rename(columns={NECENT_COLS["prompt_harmful"]: "prompt_harmful"})
    for layer, s in scores.items():
        result[f"probe_score_layer{layer}"] = s
    exp017_thresholds = {l: p.threshold for l, p in FROZEN_PROBES.items()}
    votes, exceeded = ensemble_decision(scores, exp017_thresholds)
    result["exp017_default_votes"] = votes
    result["exp017_default_flagged_DIAGNOSTIC_ONLY"] = exceeded

    result.to_parquet(ckpt_path)
    print(f"[{split_name}] checkpointed to {ckpt_path}")
    return result

PROMPT_LEVEL_RESULTS = {}
for name, idx in locked_splits.items():
    PROMPT_LEVEL_RESULTS[name] = run_prompt_level_eval(name, idx)

mark_stage_done("stage6c_prompt_level_eval", splits=list(PROMPT_LEVEL_RESULTS.keys()))
save_manifest(RUN_DIR)
print("\nPrompt-level evaluation complete for all splits.")


# ## Stage 7 — Generation-Time Activation Extraction (`behavioral_test`)
# 
# **Issue #7 fix:** trajectories now explicitly record `decision_step` and
# `predicted_token_id`. The activation captured at `decision_step t` is the
# residual-stream state used to *produce* `predicted_token_id[t]` — the state
# that exists *before* that token does, never a representation of a token
# already generated.
# 
# **Issue #3 fix applied here too:** `_install_layer_hooks` targets
# `decoder_layers[L-1]`, verified in Stage 6a2 to match `hidden_states[L]`.

# In[ ]:


# Stage 7a: streaming per-token activation extraction with corrected hooks.
_LAYER_HOOK_BUFFER = {}

def _make_hook(layer_idx):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        _LAYER_HOOK_BUFFER[layer_idx] = hs[:, -1, :].detach().float().cpu().numpy()
    return hook

def _install_layer_hooks(layers):
    """Hooks decoder_layers[l-1] for each semantic layer id l, so the
    captured activation equals hidden_states[l] (verified in Stage 6a2) —
    keyed in _LAYER_HOOK_BUFFER by the semantic id l, not the hook index."""
    handles = []
    decoder_layers = QWEN_MODEL.model.layers
    for l in layers:
        handles.append(decoder_layers[l - 1].register_forward_hook(_make_hook(l)))
    return handles

# OOM FIX: no cache clearing existed anywhere in the long per-example
# generation loops (Stage 7b, Stage 10c, Stage 14b run hundreds to
# thousands of sequential generate calls). Even though each call frees
# its own tensors, the CUDA allocator can fragment over that many
# iterations; this periodic cleanup is what actually keeps memory stable
# across a full run rather than growing until it OOMs partway through.
def _free_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def generate_with_token_trajectories(prompt, max_new_tokens=None, layers=None):
    """Returns dict with: text, decision_step (0..n-1), predicted_token_id
    (token chosen at each decision_step), layer_activations[l] (n_steps,
    hidden_dim) — the activation used to PRODUCE predicted_token_id[step]."""
    layers = layers or CONFIG["probe_layers"]
    max_new_tokens = max_new_tokens or CONFIG["max_new_tokens"]
    handles = _install_layer_hooks(layers)
    layer_activations = {l: [] for l in layers}
    predicted_token_ids = []
    try:
        enc = QWEN_TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        past = None
        cur_input = enc["input_ids"]
        for step in range(max_new_tokens):
            out = QWEN_MODEL(input_ids=cur_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            for l in layers:
                layer_activations[l].append(_LAYER_HOOK_BUFFER[l][0])
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            predicted_token_ids.append(next_id.item())
            if next_id.item() == QWEN_TOKENIZER.eos_token_id:
                break
            cur_input = next_id
        text = QWEN_TOKENIZER.decode(predicted_token_ids, skip_special_tokens=True)
        n_steps = len(predicted_token_ids)
        return {
            "text": text,
            "decision_step": np.arange(n_steps),
            "predicted_token_id": np.array(predicted_token_ids, dtype=np.int64),
            "layer_activations": {l: np.stack(v, axis=0) for l, v in layer_activations.items()},
        }
    finally:
        for h in handles:
            h.remove()

def run_generation_extraction(subset_df, out_dir, label=""):
    """Resumable, per-example checkpointed generation over `subset_df`
    (must have a necent_row_index column). Reused for behavioral_test (here)
    and the behavioral-calibration subset (Stage 10c)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_completed.csv"
    completed = (set(pd.read_csv(manifest_path)["necent_row_index"].tolist())
                 if manifest_path.exists() and not CONFIG["force_recompute"] else set())
    todo = subset_df[~subset_df["necent_row_index"].isin(completed)]
    print(f"[{label}] {len(completed)} already checkpointed, {len(todo)} remaining this session.")
    t0 = time.time()
    for n_done, (_, row) in enumerate(todo.iterrows(), start=1):
        row_idx = int(row["necent_row_index"])
        prompt = str(row[NECENT_COLS["prompt"]])
        result = generate_with_token_trajectories(prompt)
        npz_path = out_dir / f"row_{row_idx}.npz"
        np.savez_compressed(
            npz_path, generated_text=result["text"],
            decision_step=result["decision_step"],
            predicted_token_id=result["predicted_token_id"],
            **{f"layer_{l}": v for l, v in result["layer_activations"].items()},
        )
        header = not manifest_path.exists()
        pd.DataFrame([{"necent_row_index": row_idx}]).to_csv(
            manifest_path, mode="a", header=header, index=False
        )
        del result
        if n_done % CONFIG["generation_batch_checkpoint_every"] == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] ...{n_done}/{len(todo)} ({elapsed:.1f}s, {elapsed / n_done:.2f}s/ex)")
            _free_cuda_memory()
    n_total_done = len(pd.read_csv(manifest_path)) if manifest_path.exists() else 0
    print(f"[{label}] complete: {n_total_done}/{len(subset_df)} checkpointed under {out_dir}")
    return n_total_done

print("Defined: _install_layer_hooks (corrected offset), "
      "generate_with_token_trajectories, run_generation_extraction")


# In[ ]:


# Stage 7b: run generation over behavioral_test.
GEN_DIR = RUN_DIR / "generation_trajectories"
behavioral_idx = locked_splits["behavioral_test"]
behavioral_sub = necent_df.iloc[behavioral_idx].reset_index(drop=False).rename(
    columns={"index": "necent_row_index"}
)
run_generation_extraction(behavioral_sub, GEN_DIR, label="behavioral_test")
mark_stage_done("stage7_generation_extraction", n_total=len(behavioral_sub))
save_manifest(RUN_DIR)


# ## Stage 8 — Per-Token Probe Risk Trajectories (`behavioral_test`)

# In[ ]:


# Stage 8: apply frozen probes to every checkpointed per-token trajectory.
# "votes"/"exceeded" saved here use Exp017's own default thresholds and exist
# only as a quick diagnostic marker on disk — the FROZEN behavioral decisions
# (Stage 12) recompute detection from score_layer* using BEHAVIORAL_THRESHOLDS
# + LOCKED_PERSISTENCE_K explicitly, not this default flag.
def run_trajectory_scoring(traj_dir, scores_dir, thresholds_for_default_flag=None):
    scores_dir.mkdir(parents=True, exist_ok=True)
    thresholds_for_default_flag = thresholds_for_default_flag or {l: p.threshold for l, p in FROZEN_PROBES.items()}
    traj_files = sorted(traj_dir.glob("row_*.npz"))
    for i, p in enumerate(traj_files, start=1):
        row_idx = int(p.stem.split("_")[1])
        out_path = scores_dir / f"row_{row_idx}_scores.npz"
        if out_path.exists() and not CONFIG["force_recompute"]:
            continue
        z = np.load(p, allow_pickle=True)
        acts_by_layer = {l: z[f"layer_{l}"] for l in CONFIG["probe_layers"]}
        scores_by_layer = probe_risk_scores(acts_by_layer)
        votes, exceeded = ensemble_decision(scores_by_layer, thresholds_for_default_flag)
        np.savez_compressed(
            out_path,
            generated_text=str(z["generated_text"]),
            decision_step=z["decision_step"],
            predicted_token_id=z["predicted_token_id"],
            votes=votes, exceeded=exceeded,
            **{f"score_layer{l}": s for l, s in scores_by_layer.items()},
        )
        if i % 100 == 0:
            print(f"  ...{i}/{len(traj_files)} scored")
    print(f"Scored {len(traj_files)} trajectories from {traj_dir} -> {scores_dir}")
    return len(traj_files)

TRAJ_SCORES_DIR = RUN_DIR / "trajectory_scores"
run_trajectory_scoring(GEN_DIR, TRAJ_SCORES_DIR)
mark_stage_done("stage8_trajectory_scoring")
save_manifest(RUN_DIR)


# ## Stage 9 — Behavioral Labels (`response_harmful`)
# 
# Only `response_harmful` is used as behavioral ground truth. Missing labels
# are counted, never imputed. The provenance caveat from Stage 2d applies to
# every metric derived below.

# In[ ]:


def load_behavioral_labels():
    sub = behavioral_sub.copy()
    if NECENT_COLS["response_harmful"] is None:
        sub["response_harmful"] = np.nan
    else:
        sub["response_harmful"] = necent_df.iloc[sub["necent_row_index"]][
            NECENT_COLS["response_harmful"]
        ].values
    return sub

behavioral_labels_df = load_behavioral_labels()
n_total = len(behavioral_labels_df)
n_labeled = int(behavioral_labels_df["response_harmful"].notna().sum())
n_missing = n_total - n_labeled
print(f"behavioral_test: {n_total} rows, {n_labeled} with a response_harmful label, "
      f"{n_missing} MISSING (excluded below, not imputed).")
print(f"Provenance caveat (Stage 2d): {PROVENANCE['verdict']}")

labeled = behavioral_labels_df[behavioral_labels_df["response_harmful"].notna()].copy()
if len(labeled) > 0 and NECENT_COLS["prompt_harmful"]:
    labeled["prompt_harmful"] = necent_df.iloc[labeled["necent_row_index"]][
        NECENT_COLS["prompt_harmful"]
    ].values
    crosstab = pd.crosstab(labeled["prompt_harmful"], labeled["response_harmful"],
                            rownames=["prompt_harmful"], colnames=["response_harmful"])
    print("\nDescriptive cross-tab (context only, not a behavioral metric):")
    print(crosstab.to_string())

behavioral_labels_df.to_parquet(RUN_DIR / "behavioral_labels.parquet")
mark_stage_done("stage9_behavioral_labels", n_total=n_total, n_labeled=n_labeled, n_missing=n_missing)
save_manifest(RUN_DIR)


# ## Stage 10 — Calibration: `PROMPT_THRESHOLDS`, `BEHAVIORAL_THRESHOLDS`, `LOCKED_PERSISTENCE_K` (Issues #4, #5, #6)
# 
# Everything below reads **only** from the `calibration` split. `test_*`,
# `test_ood`, and `behavioral_test` are never touched in this section.

# In[ ]:


# Stage 10a: PROMPT_THRESHOLDS — calibrated against prompt_harmful, using
# calibration-split prompt-level scores already computed in Stage 6c.
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score  # noqa: E402

calib_df = PROMPT_LEVEL_RESULTS["calibration"]
calib_labels = calib_df["prompt_harmful"].astype(int).values

PROMPT_THRESHOLDS = {}
for layer in CONFIG["probe_layers"]:
    scores = calib_df[f"probe_score_layer{layer}"].values
    fpr_arr, tpr_arr, thr_arr = roc_curve(calib_labels, scores)
    valid = thr_arr[fpr_arr <= CONFIG["target_fpr"]]
    chosen = float(valid[-1]) if len(valid) else float(thr_arr[np.argmin(fpr_arr)])
    PROMPT_THRESHOLDS[layer] = chosen
    print(f"[prompt] layer {layer}: threshold={chosen:.4f} (target_fpr={CONFIG['target_fpr']})")

calib_votes, calib_exceeded = ensemble_decision(
    {l: calib_df[f"probe_score_layer{l}"].values for l in CONFIG["probe_layers"]},
    PROMPT_THRESHOLDS,
)
calib_prompt_fpr = float(calib_exceeded[calib_labels == 0].mean()) if (calib_labels == 0).any() else np.nan
calib_prompt_detection_rate = float(calib_exceeded[calib_labels == 1].mean()) if (calib_labels == 1).any() else np.nan
print(f"\nCalibration-split PROMPT_THRESHOLDS ensemble: "
      f"prompt_fpr={calib_prompt_fpr:.4f}  prompt_detection_rate={calib_prompt_detection_rate:.4f}")

prompt_thresholds_path = RUN_DIR / "PROMPT_THRESHOLDS.json"
with open(prompt_thresholds_path, "w") as f:
    json.dump({"thresholds_by_layer": PROMPT_THRESHOLDS, "target_fpr": CONFIG["target_fpr"],
               "calibrated_on": "calibration split, prompt_harmful ONLY"}, f, indent=2)
RUN_MANIFEST["artifact_hashes"]["prompt_thresholds_sha256"] = sha256_file(prompt_thresholds_path)
mark_stage_done("stage10a_prompt_thresholds", thresholds=PROMPT_THRESHOLDS)
save_manifest(RUN_DIR)


# In[ ]:


# Stage 10b (Issue #5): assemble the LOCKED calibration split's labeled
# behavioral subset -- rows that already carry a response_harmful label in
# the dataset. This is NOT behavioral_test, and is used ONLY for behavioral
# threshold + persistence-k selection below. Both classes must be present.
if NECENT_COLS["response_harmful"] is None:
    raise RuntimeError(
        "No response_harmful column available; behavioral calibration cannot "
        "proceed. STOPPING rather than fabricating a behavioral threshold "
        "from an unrelated label."
    )

calib_idx = locked_splits["calibration"]
calib_full = necent_df.iloc[calib_idx].reset_index(drop=False).rename(columns={"index": "necent_row_index"})
calib_full["response_harmful"] = necent_df.iloc[calib_idx][NECENT_COLS["response_harmful"]].values
behavioral_calib_df = calib_full[calib_full["response_harmful"].notna()].copy()
behavioral_calib_df["response_harmful_bool"] = behavioral_calib_df["response_harmful"].astype(bool)

n_behav_calib_pos = int(behavioral_calib_df["response_harmful_bool"].sum())
n_behav_calib_neg = int((~behavioral_calib_df["response_harmful_bool"]).sum())
print(f"Behavioral calibration subset (calibration split with response_harmful): "
      f"{len(behavioral_calib_df)} rows -- {n_behav_calib_pos} positive, {n_behav_calib_neg} negative.")

if n_behav_calib_pos == 0 or n_behav_calib_neg == 0:
    raise RuntimeError(
        f"Behavioral calibration subset lacks both classes "
        f"(positives={n_behav_calib_pos}, negatives={n_behav_calib_neg}). "
        f"STOPPING rather than calibrating on a degenerate set."
    )

behavioral_calib_df.to_parquet(RUN_DIR / "behavioral_calibration_subset.parquet")
mark_stage_done("stage10b_behavioral_calibration_subset",
                 n_pos=n_behav_calib_pos, n_neg=n_behav_calib_neg)
save_manifest(RUN_DIR)


# In[ ]:


# Stage 10c: generate + score trajectories for the behavioral-calibration
# subset ONLY (its own checkpoint directories; independent of behavioral_test).
BEHAV_CALIB_GEN_DIR = RUN_DIR / "behavioral_calibration_trajectories"
BEHAV_CALIB_SCORES_DIR = RUN_DIR / "behavioral_calibration_trajectory_scores"

run_generation_extraction(behavioral_calib_df, BEHAV_CALIB_GEN_DIR, label="behavioral_calibration")
run_trajectory_scoring(BEHAV_CALIB_GEN_DIR, BEHAV_CALIB_SCORES_DIR)

mark_stage_done("stage10c_behavioral_calibration_generation", n=len(behavioral_calib_df))
save_manifest(RUN_DIR)


# In[ ]:


# Stage 10d (Issue #4/#5): BEHAVIORAL_THRESHOLDS -- per-layer thresholds
# calibrated so that the max-per-response layer score predicts
# response_harmful at target_fpr, using ONLY the behavioral-calibration
# subset's own generated trajectories.
def summarize_trajectory(scores_dir, row_idx, thresholds, k=1):
    """Applies given per-layer thresholds + persistence k to a scored
    trajectory. Caller passes PROMPT_THRESHOLDS or BEHAVIORAL_THRESHOLDS
    explicitly -- never hardcoded here."""
    p = scores_dir / f"row_{row_idx}_scores.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    n_steps = len(z["decision_step"])
    if n_steps == 0:
        return None
    votes = np.zeros(n_steps, dtype=np.int64)
    for l in CONFIG["probe_layers"]:
        votes += (z[f"score_layer{l}"] >= thresholds[l]).astype(np.int64)
    step_exceeded = votes >= CONFIG["vote_k"]
    detected = exceeded_with_persistence(step_exceeded, k)
    first_exceed_decision_step = None
    if detected:
        run = 0
        for i, v in enumerate(step_exceeded):
            run = run + 1 if v else 0
            if run >= k:
                first_exceed_decision_step = i - k + 1
                break
    max_layer_scores = {l: float(np.max(z[f"score_layer{l}"])) for l in CONFIG["probe_layers"]}
    return {
        "necent_row_index": row_idx,
        "n_steps": n_steps,
        "detected": bool(detected),
        "first_exceed_decision_step": first_exceed_decision_step,
        "response_max_ensemble_score": float(np.max(np.mean(
            [z[f"score_layer{l}"] for l in CONFIG["probe_layers"]], axis=0))),
        **{f"max_score_layer{l}": v for l, v in max_layer_scores.items()},
    }

behav_calib_summaries = []
for _, row in behavioral_calib_df.iterrows():
    row_idx = int(row["necent_row_index"])
    p = BEHAV_CALIB_SCORES_DIR / f"row_{row_idx}_scores.npz"
    if not p.exists():
        continue
    z = np.load(p, allow_pickle=True)
    rec = {"necent_row_index": row_idx, "response_harmful_bool": bool(row["response_harmful_bool"])}
    for l in CONFIG["probe_layers"]:
        rec[f"max_score_layer{l}"] = float(np.max(z[f"score_layer{l}"])) if len(z["decision_step"]) else np.nan
    behav_calib_summaries.append(rec)
behav_calib_summary_df = pd.DataFrame(behav_calib_summaries)

y_behav = behav_calib_summary_df["response_harmful_bool"].astype(int).values
BEHAVIORAL_THRESHOLDS = {}
for l in CONFIG["probe_layers"]:
    scores = behav_calib_summary_df[f"max_score_layer{l}"].values
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_behav, scores)
    valid = thr_arr[fpr_arr <= CONFIG["target_fpr"]]
    chosen = float(valid[-1]) if len(valid) else float(thr_arr[np.argmin(fpr_arr)])
    BEHAVIORAL_THRESHOLDS[l] = chosen
    print(f"[behavioral] layer {l}: threshold={chosen:.4f}")

behav_calib_summary_df.to_parquet(RUN_DIR / "behavioral_calibration_summary.parquet")
behav_thresholds_path = RUN_DIR / "BEHAVIORAL_THRESHOLDS.json"
with open(behav_thresholds_path, "w") as f:
    json.dump({"thresholds_by_layer": BEHAVIORAL_THRESHOLDS, "target_fpr": CONFIG["target_fpr"],
               "calibrated_on": "calibration split rows with response_harmful ONLY",
               "n_pos": n_behav_calib_pos, "n_neg": n_behav_calib_neg}, f, indent=2)
RUN_MANIFEST["artifact_hashes"]["behavioral_thresholds_sha256"] = sha256_file(behav_thresholds_path)
mark_stage_done("stage10d_behavioral_thresholds", thresholds=BEHAVIORAL_THRESHOLDS)
save_manifest(RUN_DIR)


# In[ ]:


# Stage 10e (Issue #6): persistence k selection -- ONLY on behavioral
# calibration data, using BEHAVIORAL_THRESHOLDS. Locked before behavioral_test
# or test_ood are touched by any threshold/k selection.
persistence_rows_calib = []
for k in CONFIG["persistence_sweep"]:
    dets = []
    for _, row in behavioral_calib_df.iterrows():
        row_idx = int(row["necent_row_index"])
        summ = summarize_trajectory(BEHAV_CALIB_SCORES_DIR, row_idx, BEHAVIORAL_THRESHOLDS, k=k)
        if summ is None:
            continue
        dets.append((bool(row["response_harmful_bool"]), summ["detected"]))
    if not dets:
        continue
    ddf = pd.DataFrame(dets, columns=["harmful", "detected"])
    pos, neg = ddf[ddf["harmful"]], ddf[~ddf["harmful"]]
    persistence_rows_calib.append({
        "k": k, "n": len(ddf),
        "detection_rate": float(pos["detected"].mean()) if len(pos) else np.nan,
        "false_positive_rate": float(neg["detected"].mean()) if len(neg) else np.nan,
    })
persistence_calib_df = pd.DataFrame(persistence_rows_calib)
print("Persistence sweep on BEHAVIORAL CALIBRATION DATA ONLY:")
print(persistence_calib_df.to_string(index=False))

# Documented, locked selection rule: among k whose FPR <= target_fpr, pick the
# highest detection_rate; if none qualify, fall back to the lowest-FPR k.
eligible = persistence_calib_df[persistence_calib_df["false_positive_rate"] <= CONFIG["target_fpr"]]
if len(eligible) > 0:
    LOCKED_PERSISTENCE_K = int(eligible.sort_values("detection_rate", ascending=False).iloc[0]["k"])
    persistence_selection_rule = f"max detection_rate subject to false_positive_rate <= {CONFIG['target_fpr']}"
else:
    LOCKED_PERSISTENCE_K = int(persistence_calib_df.sort_values("false_positive_rate").iloc[0]["k"])
    persistence_selection_rule = f"no k met target_fpr={CONFIG['target_fpr']}; fell back to lowest-FPR k"

print(f"\nLOCKED_PERSISTENCE_K = {LOCKED_PERSISTENCE_K}  (rule: {persistence_selection_rule})")
persistence_calib_df.to_csv(RUN_DIR / "persistence_sweep_BEHAVIORAL_CALIBRATION_ONLY.csv", index=False)
mark_stage_done("stage10e_persistence_selection", k=LOCKED_PERSISTENCE_K, rule=persistence_selection_rule)
save_manifest(RUN_DIR)


# ## Stage 11 — Frozen Prompt-Level Evaluation on Test Splits (12A)
# 
# Uses `PROMPT_THRESHOLDS` (locked in Stage 10a). No further tuning below.

# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score  # noqa: E402

TEST_SPLIT_NAMES = ["test_benign", "test_harmful", "test_jailbreak", "test_injection"]

def evaluate_prompt_level_split(name):
    df = PROMPT_LEVEL_RESULTS[name]
    y = df["prompt_harmful"].astype(int).values
    scores_by_layer = {l: df[f"probe_score_layer{l}"].values for l in CONFIG["probe_layers"]}
    votes, exceeded = ensemble_decision(scores_by_layer, PROMPT_THRESHOLDS)
    pred = exceeded.astype(int)
    out = {"split": name, "n": len(df)}
    if len(np.unique(y)) == 2:
        out["precision"] = precision_score(y, pred, zero_division=0)
        out["recall"] = recall_score(y, pred, zero_division=0)
        out["f1"] = f1_score(y, pred, zero_division=0)
        out["balanced_accuracy"] = balanced_accuracy_score(y, pred)
        out["auroc"] = roc_auc_score(y, np.mean(list(scores_by_layer.values()), axis=0))
    out["prompt_fpr"] = float(pred[y == 0].mean()) if (y == 0).any() else np.nan
    out["prompt_detection_rate"] = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    return out, df.assign(ensemble_votes=votes, ensemble_flagged=exceeded)

frozen_eval_rows = []
FROZEN_EVAL_DETAIL = {}
for name in TEST_SPLIT_NAMES:
    summary, detail = evaluate_prompt_level_split(name)
    frozen_eval_rows.append(summary)
    FROZEN_EVAL_DETAIL[name] = detail
    detail.to_parquet(RUN_DIR / f"frozen_eval_{name}.parquet")

frozen_eval_df = pd.DataFrame(frozen_eval_rows)
print(frozen_eval_df.to_string(index=False))
frozen_eval_df.to_csv(RUN_DIR / "frozen_eval_test_splits.csv", index=False)
mark_stage_done("stage11_frozen_prompt_eval", splits=TEST_SPLIT_NAMES)
save_manifest(RUN_DIR)


# ## Stage 12 — Frozen Behavioral Evaluation on `behavioral_test` (12B)
# 
# Every threshold/k value below (`BEHAVIORAL_THRESHOLDS`, `LOCKED_PERSISTENCE_K`)
# was selected in Stage 10 using **only** the calibration split's behavioral
# subset. This is a one-shot evaluation on `behavioral_test`.

# In[ ]:


behav_test_summaries = []
for _, row in behavioral_labels_df.iterrows():
    row_idx = int(row["necent_row_index"])
    summ = summarize_trajectory(TRAJ_SCORES_DIR, row_idx, BEHAVIORAL_THRESHOLDS, k=LOCKED_PERSISTENCE_K)
    if summ is None:
        continue
    summ["response_harmful"] = row["response_harmful"]
    behav_test_summaries.append(summ)
behav_test_summary_df = pd.DataFrame(behav_test_summaries)
behav_test_labeled = behav_test_summary_df[behav_test_summary_df["response_harmful"].notna()].copy()
behav_test_labeled["response_harmful_bool"] = behav_test_labeled["response_harmful"].astype(bool)

n_behav_eval = len(behav_test_labeled)
print(f"Frozen behavioral evaluation: {n_behav_eval} labeled+scored rows "
      f"(of {n_labeled} labeled behavioral_test rows).")

behavioral_detection_rate, behavioral_fpr = np.nan, np.nan
behavioral_auroc, behavioral_auprc = np.nan, np.nan
if n_behav_eval > 0 and behav_test_labeled["response_harmful_bool"].nunique() == 2:
    y = behav_test_labeled["response_harmful_bool"].astype(int).values
    scores = behav_test_labeled["response_max_ensemble_score"].values
    behavioral_auroc = roc_auc_score(y, scores)
    behavioral_auprc = average_precision_score(y, scores)
    pos = behav_test_labeled[behav_test_labeled["response_harmful_bool"]]
    neg = behav_test_labeled[~behav_test_labeled["response_harmful_bool"]]
    behavioral_detection_rate = float(pos["detected"].mean()) if len(pos) else np.nan
    behavioral_fpr = float(neg["detected"].mean()) if len(neg) else np.nan

print(f"behavioral_auroc={behavioral_auroc}")
print(f"behavioral_auprc={behavioral_auprc}")
print(f"behavioral_detection_rate={behavioral_detection_rate}")
print(f"behavioral_fpr={behavioral_fpr}")
print(f"(BEHAVIORAL_THRESHOLDS + LOCKED_PERSISTENCE_K={LOCKED_PERSISTENCE_K}, "
      f"both selected on calibration data only.)")

# Descriptive-only persistence diagnostic on behavioral_test -- NOT used to
# select k (already locked in Stage 10e). Kept only for the figure.
persistence_diagnostic_rows = []
for k in CONFIG["persistence_sweep"]:
    dets = []
    for _, row in behavioral_labels_df.iterrows():
        row_idx = int(row["necent_row_index"])
        lbl = row["response_harmful"]
        if pd.isna(lbl):
            continue
        summ = summarize_trajectory(TRAJ_SCORES_DIR, row_idx, BEHAVIORAL_THRESHOLDS, k=k)
        if summ is None:
            continue
        dets.append((bool(lbl), summ["detected"]))
    if not dets:
        continue
    ddf = pd.DataFrame(dets, columns=["harmful", "detected"])
    pos, neg = ddf[ddf["harmful"]], ddf[~ddf["harmful"]]
    persistence_diagnostic_rows.append({
        "k": k, "n": len(ddf),
        "detection_rate": float(pos["detected"].mean()) if len(pos) else np.nan,
        "false_positive_rate": float(neg["detected"].mean()) if len(neg) else np.nan,
    })
persistence_diagnostic_test_df = pd.DataFrame(persistence_diagnostic_rows)
print("\nDescriptive-only persistence diagnostic on behavioral_test "
      "(NOT used for k selection -- k locked in Stage 10e on calibration data):")
print(persistence_diagnostic_test_df.to_string(index=False))

behav_test_summary_df.to_parquet(RUN_DIR / "frozen_behavioral_eval_behavioral_test.parquet")
persistence_diagnostic_test_df.to_csv(
    RUN_DIR / "persistence_diagnostic_behavioral_test_DESCRIPTIVE_ONLY.csv", index=False
)
mark_stage_done("stage12_frozen_behavioral_eval", n_eval=n_behav_eval,
                 behavioral_detection_rate=behavioral_detection_rate,
                 behavioral_fpr=behavioral_fpr,
                 behavioral_auroc=behavioral_auroc, behavioral_auprc=behavioral_auprc)
save_manifest(RUN_DIR)


# ## Stage 13 — OOD Evaluation (12D)
# 
# `test_ood` scored with the same frozen `PROMPT_THRESHOLDS` as Stage 11 —
# never recalibrated against OOD. Prompt-level only (no generation trajectories
# exist for `test_ood` by design), reported separately from in-distribution
# results, never conflated.

# In[ ]:


ood_summary, ood_detail = evaluate_prompt_level_split("test_ood")
ood_detail.to_parquet(RUN_DIR / "frozen_eval_test_ood.parquet")
ood_summary_df = pd.DataFrame([ood_summary])
print(ood_summary_df.to_string(index=False))

id_vs_ood = pd.concat([frozen_eval_df.assign(regime="in_distribution"),
                        ood_summary_df.assign(regime="ood")], ignore_index=True)
id_vs_ood.to_csv(RUN_DIR / "in_distribution_vs_ood.csv", index=False)
mark_stage_done("stage13_ood_eval")
save_manifest(RUN_DIR)
print("\nOOD evaluated with frozen PROMPT_THRESHOLDS; not used for calibration; "
      "prompt-level generalization only (12D).")


# ## Stage 14 — Firewall-Monitored Generation (12C)
# 
# Uses `BEHAVIORAL_THRESHOLDS` + `LOCKED_PERSISTENCE_K` (Stage 10, calibration-
# only). Fail-closed: any scoring error blocks generation rather than letting
# it proceed unmonitored.

# In[ ]:


@torch.no_grad()
def firewall_monitored_generate(prompt, max_new_tokens=None):
    """Generates token-by-token; at each decision_step, scores the activation
    used to produce that step's token against BEHAVIORAL_THRESHOLDS and blocks
    once LOCKED_PERSISTENCE_K consecutive decision_steps exceed the k_of_n
    ensemble vote. Fail-closed on any scoring error."""
    layers = CONFIG["probe_layers"]
    k = LOCKED_PERSISTENCE_K
    max_new_tokens = max_new_tokens or CONFIG["max_new_tokens"]
    handles = _install_layer_hooks(layers)
    exceed_run = 0
    decision_log = []
    blocked = False
    firewall_block_decision_step = None
    try:
        enc = QWEN_TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        past = None
        cur_input = enc["input_ids"]
        predicted_token_ids = []
        for step in range(max_new_tokens):
            try:
                out = QWEN_MODEL(input_ids=cur_input, past_key_values=past, use_cache=True)
                past = out.past_key_values
                acts = {l: _LAYER_HOOK_BUFFER[l] for l in layers}
                scores = probe_risk_scores(acts)
                votes = sum(int(scores[l][0] >= BEHAVIORAL_THRESHOLDS[l]) for l in layers)
                step_exceeded = votes >= CONFIG["vote_k"]
            except Exception as e:
                blocked = True
                firewall_block_decision_step = step
                decision_log.append({"decision_step": step, "error": str(e), "fail_closed": True})
                break
            exceed_run = exceed_run + 1 if step_exceeded else 0
            decision_log.append({"decision_step": step, "votes": votes,
                                  "exceeded": bool(step_exceeded), "consecutive_run": exceed_run})
            if exceed_run >= k:
                blocked = True
                firewall_block_decision_step = step
                break
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            predicted_token_ids.append(next_id.item())
            if next_id.item() == QWEN_TOKENIZER.eos_token_id:
                break
            cur_input = next_id
        text = QWEN_TOKENIZER.decode(predicted_token_ids, skip_special_tokens=True)
        return {"text": text, "blocked": blocked,
                "firewall_block_decision_step": firewall_block_decision_step,
                "decision_log": decision_log, "persistence_k": k}
    finally:
        for h in handles:
            h.remove()

print("Defined: firewall_monitored_generate (BEHAVIORAL_THRESHOLDS, "
      "LOCKED_PERSISTENCE_K, fail-closed).")


# In[ ]:


# Stage 14b: run firewall-monitored generation over behavioral_test (per-
# example checkpointed).
FIREWALL_DIR = RUN_DIR / "firewall_runs"
FIREWALL_DIR.mkdir(parents=True, exist_ok=True)
fw_manifest_path = FIREWALL_DIR / "_completed.csv"

def _fw_completed():
    if fw_manifest_path.exists():
        return set(pd.read_csv(fw_manifest_path)["necent_row_index"].tolist())
    return set()

fw_completed = _fw_completed() if not CONFIG["force_recompute"] else set()
fw_todo = behavioral_sub[~behavioral_sub["necent_row_index"].isin(fw_completed)]
print(f"Firewall-monitored generation: {len(fw_completed)} already done, {len(fw_todo)} remaining.")

for n_done, (_, row) in enumerate(fw_todo.iterrows(), start=1):
    row_idx = int(row["necent_row_index"])
    prompt = str(row[NECENT_COLS["prompt"]])
    result = firewall_monitored_generate(prompt)
    out_path = FIREWALL_DIR / f"row_{row_idx}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, default=str)
    header = not fw_manifest_path.exists()
    pd.DataFrame([{"necent_row_index": row_idx}]).to_csv(
        fw_manifest_path, mode="a", header=header, index=False
    )
    del result
    if n_done % CONFIG["generation_batch_checkpoint_every"] == 0:
        print(f"  ...{n_done}/{len(fw_todo)} firewall-monitored generations this session")
        _free_cuda_memory()

mark_stage_done("stage14_firewall_generation", n_completed=len(_fw_completed()))
save_manifest(RUN_DIR)
print(f"Done. {len(_fw_completed())}/{len(behavioral_sub)} firewall runs checkpointed.")


# ## Stage 15 — Detection / False-Block / `remaining_unmonitored_tokens` Analysis (Issues #8, #9)

# In[ ]:


def load_firewall_result(row_idx):
    p = FIREWALL_DIR / f"row_{row_idx}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def unmonitored_generation_length(row_idx):
    """Length in decision steps of the UNMONITORED Stage-7 generation for this
    row -- how many tokens Qwen would have produced with no firewall."""
    p = GEN_DIR / f"row_{row_idx}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return int(len(z["decision_step"]))

fw_rows = []
for _, row in behavioral_labels_df.iterrows():
    row_idx = int(row["necent_row_index"])
    fw = load_firewall_result(row_idx)
    unmon_len = unmonitored_generation_length(row_idx)
    if fw is None or unmon_len is None:
        continue
    remaining_unmonitored_tokens = (
        unmon_len - fw["firewall_block_decision_step"]
        if fw["blocked"] and fw["firewall_block_decision_step"] is not None
        else np.nan
    )
    fw_rows.append({
        "necent_row_index": row_idx,
        "response_harmful": row["response_harmful"],
        "fw_blocked": fw["blocked"],
        "firewall_block_decision_step": fw["firewall_block_decision_step"],
        "unmonitored_generation_length": unmon_len,
        "remaining_unmonitored_tokens": remaining_unmonitored_tokens,
    })
fw_df = pd.DataFrame(fw_rows)
fw_labeled = fw_df[fw_df["response_harmful"].notna()].copy()
fw_labeled["response_harmful_bool"] = fw_labeled["response_harmful"].astype(bool)

n_fw_eval = len(fw_labeled)
firewall_detection_rate, firewall_false_block_rate = np.nan, np.nan
if n_fw_eval > 0:
    harmful = fw_labeled[fw_labeled["response_harmful_bool"]]
    benign = fw_labeled[~fw_labeled["response_harmful_bool"]]
    firewall_detection_rate = float(harmful["fw_blocked"].mean()) if len(harmful) else np.nan
    firewall_false_block_rate = float(benign["fw_blocked"].mean()) if len(benign) else np.nan
    remaining_series = harmful.loc[harmful["fw_blocked"], "remaining_unmonitored_tokens"]
    print(f"firewall_detection_rate={firewall_detection_rate:.4f}")
    print(f"firewall_false_block_rate={firewall_false_block_rate:.4f}")
    print(f"remaining_unmonitored_tokens (median, harmful+blocked): "
          f"{remaining_series.median() if len(remaining_series) else float('nan'):.1f}")
    print("\nNOTE: remaining_unmonitored_tokens measures how many additional "
          "tokens the UNMONITORED generation would have produced past the "
          "firewall's block point -- a proxy for how much of the response the "
          "firewall preempted. It is NOT 'tokens before harmful content' -- "
          "Necent provides no token-level harm annotation. A block is "
          "reported as detection only, never as proof of harm prevention.")
else:
    print("No labeled rows with both a firewall run and an unmonitored "
          "trajectory available yet.")

fw_df.to_parquet(RUN_DIR / "firewall_behavioral_crosscheck.parquet")
mark_stage_done("stage15_firewall_analysis", n_eval=n_fw_eval,
                 firewall_detection_rate=firewall_detection_rate,
                 firewall_false_block_rate=firewall_false_block_rate)
save_manifest(RUN_DIR)


# In[ ]:


# ============================================================
# Stage 15b — Normalized Firewall Preemption
# ============================================================

if (
    "fw_df" not in globals()
    or len(fw_df) == 0
):

    print(
        "No firewall dataframe available; "
        "normalized preemption cannot be calculated."
    )

else:

    fw_df = fw_df.copy()

    fw_df[
        "fraction_preempted"
    ] = np.nan

    valid = (
        fw_df[
            "fw_blocked"
        ].astype(bool)
        &
        fw_df[
            "remaining_unmonitored_tokens"
        ].notna()
        &
        (
            fw_df[
                "unmonitored_generation_length"
            ] > 0
        )
    )

    fw_df.loc[
        valid,
        "fraction_preempted"
    ] = (
        fw_df.loc[
            valid,
            "remaining_unmonitored_tokens"
        ]
        /
        fw_df.loc[
            valid,
            "unmonitored_generation_length"
        ]
    )

    blocked_harmful = fw_df[
        valid
        &
        fw_df[
            "response_harmful"
        ].astype(bool)
    ]

    if len(blocked_harmful):

        median_fraction_preempted = float(
            blocked_harmful[
                "fraction_preempted"
            ].median()
        )

        print(
            "Median fraction of unmonitored generation "
            "preempted (harmful + blocked): "
            f"{median_fraction_preempted:.3f}"
        )

    else:

        median_fraction_preempted = np.nan

        print(
            "No harmful+blocked rows available "
            "for normalized preemption."
        )

    fw_df.to_parquet(
        RUN_DIR /
        "firewall_behavioral_crosscheck.parquet"
    )

    RUN_MANIFEST[
        "notes"
    ].append(
        "fraction_preempted reports the fraction of the "
        "unmonitored continuation occurring after the "
        "firewall decision point; it is not token-level "
        "harm-prevention evidence."
    )

    mark_stage_done(
        "stage15b_normalized_preemption",
        median_fraction_preempted=
            median_fraction_preempted,
    )

    save_manifest(
        RUN_DIR
    )


# ## Stage 16 — Latency Measurement

# In[ ]:


import statistics  # noqa: E402

N_TIMING_SAMPLES = min(20, len(locked_splits["calibration"]))
timing_prompts = necent_df.iloc[locked_splits["calibration"][:N_TIMING_SAMPLES]][
    NECENT_COLS["prompt"]
].astype(str).tolist()

raw_times, monitored_times = [], []
for prompt in timing_prompts:
    t0 = time.time()
    _ = generate_with_token_trajectories(prompt, max_new_tokens=32)
    raw_times.append(time.time() - t0)

    t0 = time.time()
    _ = firewall_monitored_generate(prompt, max_new_tokens=32)
    monitored_times.append(time.time() - t0)

latency_summary = {
    "n_samples": N_TIMING_SAMPLES,
    "raw_mean_s": statistics.mean(raw_times),
    "raw_stdev_s": statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0,
    "monitored_mean_s": statistics.mean(monitored_times),
    "monitored_stdev_s": statistics.stdev(monitored_times) if len(monitored_times) > 1 else 0.0,
}
latency_summary["overhead_pct"] = (
    100.0 * (latency_summary["monitored_mean_s"] - latency_summary["raw_mean_s"])
    / latency_summary["raw_mean_s"] if latency_summary["raw_mean_s"] > 0 else float("nan")
)
print(json.dumps(latency_summary, indent=2))

with open(RUN_DIR / "latency_summary.json", "w") as f:
    json.dump(latency_summary, f, indent=2)
mark_stage_done("stage16_latency")
save_manifest(RUN_DIR)


# ## Stage 17 — Figures

# In[ ]:


import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve as _roc_curve  # noqa: E402

FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 17a: ROC per test split (prompt-level, mean-of-layers score vs prompt_harmful)
fig, ax = plt.subplots(figsize=(6, 5))
for name in TEST_SPLIT_NAMES + ["test_ood"]:
    df = PROMPT_LEVEL_RESULTS[name]
    y = df["prompt_harmful"].astype(int).values
    if len(np.unique(y)) < 2:
        continue
    mean_score = np.mean([df[f"probe_score_layer{l}"].values for l in CONFIG["probe_layers"]], axis=0)
    fpr_arr, tpr_arr, _ = _roc_curve(y, mean_score)
    ax.plot(fpr_arr, tpr_arr, label=name)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — prompt-level, PROMPT_THRESHOLDS-consistent scoring")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "roc_by_split_prompt_level.png", dpi=150); plt.close(fig)

# 17b: example per-token risk trajectory (behavioral_test), with BOTH
# threshold sets drawn for comparison.
example_files = sorted(TRAJ_SCORES_DIR.glob("row_*_scores.npz"))[:1]
if example_files:
    z = np.load(example_files[0], allow_pickle=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for l in CONFIG["probe_layers"]:
        ax.plot(z["decision_step"], z[f"score_layer{l}"], label=f"layer {l}", alpha=0.8)
    ax.axhline(PROMPT_THRESHOLDS[CONFIG["probe_layers"][0]], color="blue",
               linestyle=":", linewidth=0.8, label="PROMPT_THRESHOLDS (layer 19)")
    ax.axhline(BEHAVIORAL_THRESHOLDS[CONFIG["probe_layers"][0]], color="red",
               linestyle=":", linewidth=0.8, label="BEHAVIORAL_THRESHOLDS (layer 19)")
    ax.set_xlabel("decision_step"); ax.set_ylabel("probe risk score")
    ax.set_title(f"Per-token risk trajectory — {example_files[0].stem}")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "example_trajectory.png", dpi=150); plt.close(fig)

# 17c: persistence sweep -- calibration selection curve AND behavioral_test
# descriptive-only diagnostic, drawn side by side so it's visually clear which
# one was used for selection.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
if len(persistence_calib_df) > 0:
    axes[0].plot(persistence_calib_df["k"], persistence_calib_df["detection_rate"], marker="o", label="detection rate")
    axes[0].plot(persistence_calib_df["k"], persistence_calib_df["false_positive_rate"], marker="o", label="false positive rate")
    axes[0].axvline(LOCKED_PERSISTENCE_K, color="green", linestyle="--", linewidth=0.8, label="locked k")
    axes[0].set_title("Selection: behavioral CALIBRATION only")
    axes[0].set_xlabel("persistence k"); axes[0].set_ylabel("rate")
    axes[0].set_xticks(CONFIG["persistence_sweep"]); axes[0].legend(fontsize=7)
if len(persistence_diagnostic_test_df) > 0:
    axes[1].plot(persistence_diagnostic_test_df["k"], persistence_diagnostic_test_df["detection_rate"], marker="o", label="detection rate")
    axes[1].plot(persistence_diagnostic_test_df["k"], persistence_diagnostic_test_df["false_positive_rate"], marker="o", label="false positive rate")
    axes[1].set_title("Descriptive-only diagnostic: behavioral_test\n(NOT used for selection)")
    axes[1].set_xlabel("persistence k")
    axes[1].set_xticks(CONFIG["persistence_sweep"]); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / "persistence_sweep.png", dpi=150); plt.close(fig)

# 17d: latency
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(["raw", "firewall-monitored"],
       [latency_summary["raw_mean_s"], latency_summary["monitored_mean_s"]],
       yerr=[latency_summary["raw_stdev_s"], latency_summary["monitored_stdev_s"]])
ax.set_ylabel("seconds per 32-token generation")
ax.set_title(f"Latency (n={latency_summary['n_samples']})")
fig.tight_layout(); fig.savefig(FIG_DIR / "latency.png", dpi=150); plt.close(fig)

print(f"Saved figures under {FIG_DIR}: " + ", ".join(p.name for p in FIG_DIR.glob("*.png")))
mark_stage_done("stage17_figures")
save_manifest(RUN_DIR)


# ## Stage 18 — Final Forensic Report (Issue #12)
# 
# Keeps **A) prompt-level detectability, B) behavioral prediction,
# C) runtime firewall interruption, D) OOD generalization** in separate
# sections. No collapsed "safety score". Interruption is never described as
# proof of harm prevention.

# In[ ]:


report_lines = []
report_lines.append(f"# NFW-001 Forensic Report — run_id `{CONFIG['run_id']}` (AUDITED VERSION)")
report_lines.append(f"\nGenerated: {now_iso()}\n")

report_lines.append("## Corrections Applied in This Revision")
for c in CORRECTIONS_APPLIED:
    report_lines.append(f"- {c}")

report_lines.append("\n## Reproducibility")
report_lines.append(f"- Dataset: `{CONFIG['hf_dataset_id']}` split=`{CONFIG['hf_split']}`, "
                     f"{NECENT_ROW_COUNT} rows at load time, Hub commit sha=`{DATASET_REVISION_SHA}`")
report_lines.append(f"- Split file SHA256: `{SPLIT_FILE_SHA256}`")
report_lines.append(f"- Model: `{CONFIG['model_id']}` "
                     f"(revision: `{RUN_MANIFEST['artifact_hashes'].get('model_revision')}`), "
                     f"hidden_size={QWEN_HIDDEN_SIZE}")
report_lines.append(f"- Probe layers: {CONFIG['probe_layers']}, pooling={CONFIG['pooling']} "
                     f"(activation-location verified against hidden_states[L] — see Stage 6a2)")
for l in CONFIG["probe_layers"]:
    report_lines.append(f"  - layer {l} artifact SHA256: "
                         f"`{RUN_MANIFEST['artifact_hashes'].get(f'probe_layer_{l}')}`")
report_lines.append(f"- Seed: {CONFIG['seed']}")
report_lines.append("- Package versions: " + json.dumps(RUN_MANIFEST["package_versions"]))

report_lines.append("\n## Behavioral Label Provenance (Issue #2)")
report_lines.append(f"- {PROVENANCE['verdict']}")
report_lines.append(f"- model_name column found: {PROVENANCE['model_name_column_found']}")
if PROVENANCE["model_name_values"] is not None:
    report_lines.append(f"- model_name values observed: {PROVENANCE['model_name_values']}")

report_lines.append("\n## Split Composition")
report_lines.append(composition_df.to_markdown())

report_lines.append("\n## Calibration (calibration split ONLY)")
report_lines.append(f"- PROMPT_THRESHOLDS: {PROMPT_THRESHOLDS}")
report_lines.append(f"  - calibration-split prompt_fpr={calib_prompt_fpr:.4f}, "
                     f"prompt_detection_rate={calib_prompt_detection_rate:.4f}")
report_lines.append(f"- Behavioral calibration subset: {n_behav_calib_pos} positive / "
                     f"{n_behav_calib_neg} negative (calibration split rows with response_harmful)")
report_lines.append(f"- BEHAVIORAL_THRESHOLDS: {BEHAVIORAL_THRESHOLDS}")
report_lines.append(f"- LOCKED_PERSISTENCE_K: {LOCKED_PERSISTENCE_K} (rule: {persistence_selection_rule})")

report_lines.append("\n## A) Prompt-Level Probe Detectability — In-Distribution Test Splits")
report_lines.append(frozen_eval_df.to_markdown(index=False))

report_lines.append("\n## D) OOD Generalization (frozen PROMPT_THRESHOLDS, prompt-level only)")
report_lines.append(ood_summary_df.to_markdown(index=False))

report_lines.append(
    "\n## B) External Behavioral-Label Prediction — "
    "Frozen Evaluation on `behavioral_test`"
)

report_lines.append(
    "- `response_harmful` is treated as an external Necent "
    "behavioral label. It is NOT automatically a ground-truth "
    "label for the Qwen generation produced by this notebook."
)

report_lines.append(
    "- Therefore these metrics measure prediction of the "
    "Necent response-harmful construct, not independently "
    "verified harmfulness of the Qwen output."
)
report_lines.append(f"- Evaluated on {n_behav_eval} labeled+scored rows (of {n_labeled} labeled total)")

report_lines.append(
    f"- external_behavioral_label_auroc: "
    f"{behavioral_auroc}"
)

report_lines.append(
    f"- external_behavioral_label_auprc: "
    f"{behavioral_auprc}"
)

report_lines.append(f"- behavioral_detection_rate: {behavioral_detection_rate}")
report_lines.append(f"- behavioral_fpr: {behavioral_fpr}")
report_lines.append("- Descriptive-only persistence diagnostic on behavioral_test "
                     "(NOT used for k selection):")
report_lines.append(persistence_diagnostic_test_df.to_markdown(index=False)
                     if len(persistence_diagnostic_test_df) else "N/A")

report_lines.append("\n## C) Runtime Firewall Interruption — `behavioral_test`")
report_lines.append(f"- Evaluated on {n_fw_eval} rows")
report_lines.append(f"- firewall_detection_rate: {firewall_detection_rate}")
report_lines.append(f"- firewall_false_block_rate: {firewall_false_block_rate}")
report_lines.append("- remaining_unmonitored_tokens is reported per-row in "
                     "firewall_behavioral_crosscheck.parquet (median summarized above in Stage 15 output).")
report_lines.append("- A block is reported as detection only. It is NOT interpreted as harm "
                     "prevention: this notebook does not re-score what the unblocked "
                     "continuation would have produced.")

report_lines.append("\n## Latency")
report_lines.append(f"```json\n{json.dumps(latency_summary, indent=2)}\n```")

report_lines.append("\n## Notes / Warnings Logged During This Run")
for note in RUN_MANIFEST["notes"]:
    report_lines.append(f"- {note}")

report_md = "\n".join(str(x) for x in report_lines)
report_path = RUN_DIR / "FORENSIC_REPORT.md"
with open(report_path, "w") as f:
    f.write(report_md)

mark_stage_done("stage18_forensic_report")
save_manifest(RUN_DIR)
print(f"Forensic report written to {report_path} ({len(report_md)} chars)")


# ## Stage 19 — Save All Outputs to Drive

# In[ ]:


RUN_MANIFEST["completed_at"] = now_iso()
save_manifest(RUN_DIR)

zip_path = RUN_DIR.parent / f"{CONFIG['run_id']}.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in RUN_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(RUN_DIR.parent))
print(f"Zipped run directory to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

print(f"\nAll outputs under: {RUN_DIR}")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)}")

print(f"\nNFW-001 (audited) run '{CONFIG['run_id']}' complete. Re-running in a "
      f"fresh runtime with the same run_id resumes from the last completed "
      f"stage rather than redoing everything.")

