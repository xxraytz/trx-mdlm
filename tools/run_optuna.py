import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
METRIC_LINE_RE = re.compile(
    rf"global step\s+(?P<step>\d+):\s*'(?P<key>[^']+)'\s+reached\s+({NUM_RE})\s+\(best\s+(?P<best>{NUM_RE})\)",
    re.IGNORECASE,
)

NOT_TOP_RE = re.compile(
    r"global step\s+(?P<step>\d+):\s*'(?P<key>[^']+)'.*?was not in top",
    re.IGNORECASE,
)

def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

def _trial_suggest(trial: optuna.trial.Trial, name: str, kind: str, kwargs: dict):
    if kind == "suggest_float":
        return trial.suggest_float(name, **kwargs)
    if kind == "suggest_int":
        return trial.suggest_int(name, **kwargs)
    if kind == "suggest_categorical":
        return trial.suggest_categorical(name, kwargs["choices"])
    raise ValueError(f"Unknown suggest kind: {kind}")

def _build_trial_overrides(
    trial: optuna.trial.Trial,
    suggestions: Sequence[Tuple[str, Tuple[str, Dict]]],
) -> List[str]:
    ov = []
    for key, spec in suggestions:
        kind, params = spec
        val = _trial_suggest(trial, key, kind, params)
        ov.append(f"{key}={_fmt(val)}")
    return ov

def _make_trial_dir(base: Path, study: str, trial_number: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / "outputs" / "optuna" / study / f"trial_{trial_number:04d}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _extract_metric_from_log(text: str, metric_key: str) -> float:
    """
    Ищет строки:
      Epoch ..., global step ...: '<metric_key>' reached X (best Y), saving model ...
    Берём последнее вхождение и возвращаем Y (best).
    """
    # Пример ключа: val/nll — в логе он в одинарных кавычках.

    pat = re.compile(
        rf":\s*'{re.escape(metric_key)}'\s+reached\s+({NUM_RE})\s+\(best\s+({NUM_RE})\)",
        re.IGNORECASE,
    )
    last = None
    for m in pat.finditer(text):
        last = m
    if last is None:
        raise RuntimeError(f"Не нашёл '{metric_key}' в логе. Проверьте, что метрика логируется.")
    best_val = float(last.group(2))
    return best_val


def _make_pruner(params: Dict) -> optuna.pruners.BasePruner:
    pr_cfg = dict(params.get("pruner", {}))
    kind = str(pr_cfg.get("kind", "median")).lower()

    if kind == "none":
        return optuna.pruners.NopPruner()

    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pr_cfg.get("n_startup_trials", params.get("n_startup_trials", 10))),
            n_warmup_steps=int(pr_cfg.get("n_warmup_steps", 5)),
            interval_steps=int(pr_cfg.get("interval_steps", 1)),
        )

    if kind == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=float(pr_cfg.get("percentile", 25.0)),
            n_startup_trials=int(pr_cfg.get("n_startup_trials", params.get("n_startup_trials", 10))),
            n_warmup_steps=int(pr_cfg.get("n_warmup_steps", 5)),
            interval_steps=int(pr_cfg.get("interval_steps", 1)),
        )

    if kind == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pr_cfg.get("min_resource", 1)),
            reduction_factor=int(pr_cfg.get("reduction_factor", 3)),
            min_early_stopping_rate=int(pr_cfg.get("min_early_stopping_rate", 0)),
        )

    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pr_cfg.get("min_resource", 1)),
            reduction_factor=int(pr_cfg.get("reduction_factor", 3)),
        )

    if kind == "threshold":
        # направление берём из params.direction
        direction = str(params.get("direction", "min")).lower()
        lower = pr_cfg.get("lower", None)
        upper = pr_cfg.get("upper", None)
        return optuna.pruners.ThresholdPruner(
            lower=float(lower) if lower is not None else None,
            upper=float(upper) if upper is not None else None,
            # В ThresholdPruner важна интерпретация направления,
            # но Optuna берёт его из study.direction — здесь ничего доп.передавать не нужно.
        )

    raise ValueError(f"Unknown pruner kind: {kind}")


def make_objective(cfg: DictConfig, project_root: Path, base_overrides: list, study_name: str,
                   run_config_dir: str, run_config_name: str):
    params = cfg.optuna.params
    suggestions = cfg.optuna.suggestions

    direction: str = str(params.get("direction", "min")).lower()
    assert direction in ("min", "max")
    metric_key: str = params.get("target_metric", "val/nll")
    sanity_steps: int = int(params.get("num_sanity_val_steps", 0))
    trial_overrides: list = list(params.get("trial_overrides", []))

    # Печатаем только краткие строки
    def _should_print(line: str) -> bool:
        return METRIC_LINE_RE.search(line) is not None

    def objective(trial: optuna.trial.Trial):
        trial_dir = _make_trial_dir(project_root, study_name, trial.number)

        trial_ov = _build_trial_overrides(trial, suggestions)
        hydra_ov = []
        hydra_ov += trial_ov
        hydra_ov += base_overrides
        hydra_ov += trial_overrides
        hydra_ov += [
            f"hydra.run.dir={str(trial_dir)}",
            "wandb=null",
            f"trainer.num_sanity_val_steps={sanity_steps}",
        ]

        # ВАЖНО: чтобы не словить сравнение int со str внутри Trainer,
        # зафиксируем валидный val_check_interval (число), если он вдруг None.
        if not any(x.startswith("trainer.val_check_interval=") for x in hydra_ov):
            hydra_ov.append("trainer.val_check_interval=50")

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"

        # cmd = [sys.executable, "main.py"] + hydra_ov
        cmd = [
            sys.executable, "-u", "main.py",
            "--config-dir", run_config_dir,
            "--config-name", run_config_name,
            *hydra_ov,
        ]
        print("[OPTUNA] launch:", " ".join(shlex.quote(x) for x in cmd))

        log_path = trial_dir / "train.log"
        best_step = None
        best_value = None

        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                
            )
            try:
                assert proc.stdout is not None

                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    lf.write(line + "\n")

                    if _should_print(line):
                        print(f"[T{trial.number:04d}] {line}")

                    m = METRIC_LINE_RE.search(line)
                        
                    if m and m.group("key") == metric_key:

                        best_step = int(m.group("step"))
                        best_value = float(m.group("best"))
                        trial.report(best_value, step=best_step)
                        if trial.should_prune():
                            proc.terminate()
                            raise optuna.TrialPruned(f"Pruned at step {best_step} with best={best_value}")

                retcode = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()

        (trial_dir / "overrides.json").write_text(json.dumps(hydra_ov, indent=2))

        if retcode != 0:
            txt = log_path.read_text()
            if "CUDA out of memory" in txt or "out of memory" in txt:
                raise optuna.TrialPruned("Pruned due to OOM")
            raise optuna.TrialPruned(f"Training failed (exit={retcode})")
        
        if best_value is None:
            try:
                best_value = _extract_metric_from_log(log_path.read_text(), metric_key)
            except Exception:
                raise optuna.TrialPruned(f"No metric '{metric_key}' observed; pruning trial.")
        print(f"Trial was completed with best_value = {best_value}")
        return best_value

    return objective, direction

def main():
    # ------------------------- CLI -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="config",
                        help="Имя главного Hydra-конфига (файл в configs/)")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Путь к папке configs (по умолчанию <repo>/configs)")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Доп. Hydra-оверрайды для main.py (напр. data=age model.length=256)")
    args = parser.parse_args()
    # ------------------------- Paths -------------------------
    project_root = Path(__file__).resolve().parents[1]
    src_config_dir = Path(args.config_dir) if args.config_dir else (project_root / "configs")

    study_root = project_root / "outputs" / "optuna" / args.study_name
    study_root.mkdir(parents=True, exist_ok=True)

    # ------------------------- Snapshot configs -------------------------
    snapshot_config_dir = study_root / "configs_snapshot"
    shutil.copytree(
        src_config_dir,
        snapshot_config_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".git", ".idea", ".vscode", "*.pyc"),
    )

    # ВАЖНО: далее все compose() делаем из снапшота
    with initialize_config_dir(version_base=None, config_dir=str(snapshot_config_dir)):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=args.overrides)

    # ------------------------- Optuna params -------------------------
    params = cfg.optuna.params
    n_trials = args.n_trials or int(params.get("n_trials", 100))
    n_startup = int(params.get("n_startup_trials", 10))
    seed = args.seed if args.seed is not None else int(params.get("seed", 1))
    storage = args.storage or params.get("storage", None)

    # ------------------------- Objective factory -------------------------

    objective, direction = make_objective(
        cfg,
        project_root,
        base_overrides=args.overrides,
        study_name=args.study_name,
        run_config_dir=str(snapshot_config_dir),
        run_config_name=args.config_name,
    )

    # ------------------------- Optuna study -------------------------
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    pruner = _make_pruner(params)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize" if direction == "min" else "maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=args.n_jobs)

    print(json.dumps(
        {"value": study.best_value, "params": study.best_params, "trial": study.best_trial.number},
        indent=2,
    ))


if __name__ == "__main__":
    main()
