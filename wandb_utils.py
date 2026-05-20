from __future__ import annotations

import os
from pathlib import Path
from typing import Any


WANDB_PROJECT = "rl-supernet-nas"


def sanitize_wandb_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_wandb_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_wandb_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_artifact_name(name: str) -> str:
    clean = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            clean.append(char)
        else:
            clean.append("-")
    return "".join(clean).strip("-") or "artifact"


def init_wandb_run(stage: str, args: Any, output_dir: str | Path):
    try:
        import wandb
    except Exception as exc:
        print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
        return None

    output_dir = Path(output_dir)
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("WANDB_MODE", "online")
    config = sanitize_wandb_value(vars(args) if hasattr(args, "__dict__") else args)
    try:
        return wandb.init(
            project=WANDB_PROJECT,
            name=f"{stage}-{output_dir.name}",
            config=config,
            dir=str(wandb_dir),
            mode=mode,
            reinit="finish_previous",
            settings=wandb.Settings(silent=True),
        )
    except Exception as exc:
        print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
        return None


def log_wandb(run: Any, values: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    payload = sanitize_wandb_value(values)
    try:
        run.log(payload, step=step)
    except Exception as exc:
        print(f"wandb_log_failed error={exc}", flush=True)


def log_wandb_artifact(
    run: Any,
    name: str,
    artifact_type: str,
    paths: list[str | Path],
) -> None:
    if run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(safe_artifact_name(name), type=artifact_type)
        has_content = False
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            if path.is_dir():
                artifact.add_dir(str(path), name=path.name)
            else:
                artifact.add_file(str(path), name=path.name)
            has_content = True
        if has_content:
            run.log_artifact(artifact)
    except Exception as exc:
        print(f"wandb_artifact_failed name={name} error={exc}", flush=True)


def finish_wandb_run(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"wandb_finish_failed error={exc}", flush=True)
