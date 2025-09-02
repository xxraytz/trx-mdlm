import os
import sys
import pathlib
import importlib
import yaml

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import pandas as pd
from tqdm import tqdm
import diffusion
import utils
import torch.nn.functional as F
from omegaconf import OmegaConf


from dataloader import get_dataloaders

REPO = (
    pathlib.Path(__file__).resolve().parent / ".." / "transaction-generation"
).resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
    importlib.invalidate_caches()

from generation.metrics.evaluator import EvaluatorConfig  # noqa: E402
from generation.runners.utils import DataConfig  # noqa: E402
from generation.metrics.evaluator import SampleEvaluator


GEN_EVAL_CONFIG = "/home/dev/2025/trx-mdlm/configs/gen/eval.yaml"

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
    if "hf" in config.backbone:
        return diffusion.Diffusion(config).to("cuda")

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, config=config
    )


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def check_configs(base_config, trx_config):
    assert base_config.model.length == trx_config.max_seq_len
    assert base_config.loader.global_batch_size == trx_config.batch_size
    assert base_config.loader.eval_global_batch_size == trx_config.batch_size


def generate_samples(config, logger, tokenizer):
    logger.info("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            prefix = tokenizer.encode("We buy some food")

            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
                prefix_ids=prefix,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    print("Text samples:", text_samples)
    if not config.sampling.semi_ar:
        print("Generative perplexity:", model.gen_ppl_metric.compute())
    return text_samples


def pad_to_len(x: torch.Tensor, L: int, pad_value: int):
    """Right-pad/clip 2D тензор [B, Lx] до [B, L]."""
    assert x.ndim == 2
    if x.size(1) == L:
        return x
    if x.size(1) > L:
        return x[:, :L]
    return F.pad(x, (0, L - x.size(1)), value=pad_value)


def save_df_to(part, tokens, mask, cfg, eval_path):
    path = f"{eval_path}/{part}.parquet"
    x = torch.cat(tokens, dim=0).cpu()

    m = mask

    m = m.bool() if m.dtype != torch.bool else m

    seqs = [
        x[i][m[i]].numpy() for i in range(x.size(0))
    ]  # list[np.ndarray], разная длина
    lens = m.sum(dim=1).numpy()

    df = pd.DataFrame({cfg.target_token: seqs, "_seq_len": lens})

    df[cfg.index_name] = list(range(0, len(df)))
    df.to_parquet(path, index=False)

    return path


def _eval_trx_metrics(config, logger):

    logger.info("Starting Zero Shot Eval.")

    model = _load_from_checkpoint(config=config)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    resolve_configs(config)

    data_conf = DataConfig(**OmegaConf.to_container(config["data"], resolve=True))
    eval_conf = EvaluatorConfig(**OmegaConf.to_container(config["metrics"], resolve=True))
    # data_conf = DataConfig(**yaml.safe_load(open(GEN_DATA_CONFIG)))
    # eval_conf = EvaluatorConfig(**yaml.safe_load(open(GEN_EVAL_CONFIG)))
    assert isinstance(eval_conf.metrics, list), 'Something wrong with eval configs!'
    # check_configs(config, data_conf)

    common_seed = 0
    eval_path = os.getcwd() + "/evaluation"

    os.makedirs(eval_path, exist_ok=True)

    sample_evaluator = SampleEvaluator(
        eval_path,
        data_conf,
        eval_conf,
        device=eval_conf.devices[0],
        verbose=True,
    )

    (_, _, test_ds), _ = get_dataloaders(data_conf, common_seed)
    gt = []
    mask = []
    gen = []
    for i, batch in tqdm(enumerate(test_ds)):
        _, tokens = model.generate_from_batch(
            batch, dt=float(getattr(config.sampling, "dt", 0.01))
        )
        gt.append(pad_to_len(tokens, config.model.length, 0))
        gen.append(pad_to_len(batch["input_ids"], config.model.length, 0))
        mask.append(pad_to_len(batch["attention_mask"], config.model.length, 0))
        if i > 10:
            break

    mask = torch.cat(mask, dim=0)

    gt_path = save_df_to("gt", gt, mask, cfg=data_conf, eval_path=eval_path)
    gen_path = save_df_to("gen", gen, mask, cfg=data_conf, eval_path=eval_path)

    results = sample_evaluator.estimate_metrics(gt_path, gen_path)
    print(results)


def resolve_configs(config):
    config['data']['batch_size'] = config['loader']['global_batch_size'] if config['mode'] == 'train' else config['loader']['eval_global_batch_size']
    
def _train(config, logger, tokenizer=None):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    # Dataloader from transaction generation
    # dataloader_conf = DataConfig(**yaml.safe_load(open(GEN_DATA_CONFIG)))
    resolve_configs(config)
    logger.info(f"batch_size = {config['data']['batch_size']}")
    data_dict = OmegaConf.to_container(config["data"], resolve=True)
    dataloader_conf = DataConfig(**data_dict)
    common_seed = 0

    (train_ds, valid_ds, _), (internal_dataconf, data_conf) = get_dataloaders(
        dataloader_conf, common_seed
    )
    model = diffusion.Diffusion(configs=(config, data_conf, internal_dataconf))

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    if config.mode == "sample_eval":
        generate_samples(config, logger)
    elif config.mode == "eval":
        _eval_trx_metrics(config, logger)
    else:
        _train(config, logger)


if __name__ == "__main__":
    main()
