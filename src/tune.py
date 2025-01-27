import argparse
from typing import Any, Union

import optuna
from hydra.utils import get_method
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--output", type=str, default=None)

    return parser.parse_args()

def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f"suggest_{distribution}")(label, *args)

def sample_config(
    trial: optuna.trial.Trial,
    space: Union[bool, int, float, str, dict, list],
    label_parts: list,
) -> Any:
    if isinstance(space, (bool, int, float, str)):
        return space
    elif isinstance(space, dict):
        return {
            k: sample_config(trial, subspace, label_parts + [k])
            for k, subspace in space.items()
        }
    elif isinstance(space, list):
        if not space:
            return space
        elif space[0] != "_tune_":
            return [
                sample_config(trial, subspace, label_parts + [i])
                for i, subspace in enumerate(space)
            ]
        else:
            _, distribution, *args = space
            label = ".".join(map(str, label_parts))
            return _suggest(trial, distribution, label, *args)

def main(config: DictConfig, output: str):
    function = get_method(config.function)
    space = OmegaConf.to_container(config.space)

    def objective(trial: optuna.trial.Trial) -> float:
        raw_config = sample_config(trial, space, [])
        value = function(OmegaConf.create(raw_config))
        return value

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials)

if __name__ == "__main__":
    args = init()
    config = OmegaConf.load(args.config)
    output = args.output
    if output is None:
        output = args.config[:-len(".yaml")]

    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")
    main(config=config, output=output)
