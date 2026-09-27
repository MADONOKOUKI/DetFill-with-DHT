"""Provenance of inference outputs and the training step budget (no torch dependency, unit-tested).

``sample_to_eval`` reuses output images that already exist in its result directory. That is only correct when they
were produced under the same conditions, so every result directory carries a ``run_manifest.json`` (checkpoint hash,
seed, hint type and order, hint ratio, sketch type, sampling steps, config hash, data root) and a run refuses to
continue into a directory whose manifest differs.
"""
import hashlib
import json
import os
import warnings

MANIFEST_NAME = "run_manifest.json"
#: fields that must agree for existing outputs to be reused
IDENTITY_KEYS = ("checkpoint_sha256", "seed", "hint_type", "hint_order", "sample_ratio", "sketch_type", "sample_step",
                 "domain", "model_name", "config_sha256", "data_root")


def file_sha256(path):
    """SHA-256 of a file (None when the file does not exist)."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_run_manifest(sample_dir):
    p = os.path.join(sample_dir, MANIFEST_NAME)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def write_run_manifest(sample_dir, manifest):
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True, default=str)


def check_run_manifest(sample_dir, manifest, has_outputs):
    """Compare ``manifest`` with the one stored in ``sample_dir``.

    Raises ``RuntimeError`` when outputs exist and the stored manifest differs in an identity field (the outputs
    come from another checkpoint, seed, hint setting or data); warns when outputs exist without a manifest (older
    runs, provenance unknown) so the reuse is at least visible.
    """
    old = read_run_manifest(sample_dir)
    if old is None:
        if has_outputs:
            warnings.warn(f"{sample_dir} holds outputs without a run manifest (an earlier run); they are reused as they are",
                          stacklevel=2)
        return
    diffs = {k: (old.get(k), manifest.get(k)) for k in IDENTITY_KEYS if str(old.get(k)) != str(manifest.get(k))}
    if diffs and has_outputs:
        lines = "; ".join(f"{k}: existing={o!r} now={n!r}" for k, (o, n) in diffs.items())
        raise RuntimeError(f"{sample_dir} holds outputs of a different run ({lines}). Use another --result_path or "
                           "delete the directory; outputs of different checkpoints, seeds or hint settings are not mixed.")


def step_budget_reached(global_step, n_steps):
    """True when the micro-batch budget is used up (``n_steps`` None or <= 0 means no step budget)."""
    if n_steps is None:
        return False
    try:
        n = int(n_steps)
    except (TypeError, ValueError):
        return False
    return n > 0 and int(global_step) >= n
