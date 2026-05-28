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


def init_wandb_run(stage: str, run_config: dict[str, Any], output_dir: str | Path):
    try:
        import wandb
    except Exception as exc:
        print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
        return None

    output_dir = Path(output_dir)
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("WANDB_MODE", "online")
    config = sanitize_wandb_value(run_config)
    
    env_id = config.get("ppo_config", {}).get("env_id", "unknown_env")
    clean_env_id = env_id.replace("/", "-")
    tags = [env_id] if env_id != "unknown_env" else []

    try:
        return wandb.init(
            project=WANDB_PROJECT,
            name=f"{stage}-{clean_env_id}",
            tags=tags,
            config=config,
            dir=str(wandb_dir),
            mode=mode,
        )
    except Exception as exc:
        print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
        return None


def log_wandb(run: Any, values: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    sanitized_values = sanitize_wandb_value(values)
    try:
        run.log(sanitized_values, step=step)
    except Exception as exc:
        print(f"wandb_log_failed error={exc}", flush=True)


def update_wandb_summary(run: Any, values: dict[str, Any]) -> None:
    if run is None:
        return
    sanitized_values = sanitize_wandb_value(values)
    try:
        for key, value in sanitized_values.items():
            run.summary[key] = value
    except Exception as exc:
        print(f"wandb_summary_failed error={exc}", flush=True)


def finish_wandb_run(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"wandb_finish_failed error={exc}", flush=True)
