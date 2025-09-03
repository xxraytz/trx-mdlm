import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

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

def make_objective(cfg: DictConfig, project_root: Path, base_overrides: List[str]):
    params = cfg.optuna.params
    suggestions = cfg.optuna.suggestions

    study_name: str = params.get("study_name", "optuna_study")
    direction: str = str(params.get("direction", "min")).lower()
    assert direction in ("min", "max")
    metric_key: str = params.get("target_metric", "val/nll")
    sanity_steps: int = int(params.get("num_sanity_val_steps", 0))

    def objective(trial: optuna.trial.Trial):
        # Своя рабочая директория для trial: Hydra туда всё сложит
        trial_dir = _make_trial_dir(project_root, study_name, trial.number)

        # Параметры, предложенные Optuna
        trial_ov = _build_trial_overrides(trial, suggestions)

        # Минимально необходимые оверрайды в main.py:
        hydra_ov = []
        hydra_ov += trial_ov
        hydra_ov += base_overrides  # ваши ручные (напр. data=age)
        hydra_ov += [
            f"hydra.run.dir={str(trial_dir)}",      # всё в папку trial'а
            "wandb=null",                            # отключить WandB
            f"trainer.num_sanity_val_steps={sanity_steps}",
        ]

        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"  # на всякий случай

        cmd = [sys.executable, "main.py"] + hydra_ov
        print("[OPTUNA] launch:", " ".join(shlex.quote(x) for x in cmd))

        ret = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Сохраняем лог и оверрайды
        (trial_dir / "train.log").write_text(ret.stdout)
        (trial_dir / "overrides.json").write_text(json.dumps(hydra_ov, indent=2))

        if ret.returncode != 0:
            print(ret.stdout)
            raise optuna.TrialPruned(f"Training failed with code {ret.returncode}")

        # Метрику берём из лога (последнее best для target_metric)
        score = _extract_metric_from_log(ret.stdout, metric_key)
        return score

    return objective, direction

def main():
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

    project_root = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir) if args.config_dir else (project_root / "configs")

    # Композируем ПОЛНЫЙ конфиг (ваш defaults уже содержит optuna)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=args.overrides)

    params = cfg.optuna.params
    n_trials = args.n_trials or int(params.get("n_trials", 100))
    n_startup = int(params.get("n_startup_trials", 10))
    seed = args.seed if args.seed is not None else int(params.get("seed", 1))
    storage = args.storage or params.get("storage", None)

    objective, direction = make_objective(cfg, project_root, base_overrides=args.overrides)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize" if direction == "min" else "maximize",
        sampler=sampler,
        load_if_exists=args.resume,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=args.n_jobs)

    print(json.dumps(
        {"value": study.best_value, "params": study.best_params, "trial": study.best_trial.number},
        indent=2,
    ))

if __name__ == "__main__":
    main()
