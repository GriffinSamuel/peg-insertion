"""Read local saved models and construct their unchanged evaluation adapter."""
from dataclasses import replace
from pathlib import Path
import pickle
from takehome import runtime  # noqa: F401
import jax
from takehome.config import TrainConfig
from takehome.variants import VariantConfig
from takehome.train import ExperimentRunner
from takehome.variant_runner import VariantExperimentRunner


def load_model(path, **context_changes):
    with Path(path).open('rb') as f:
        model = pickle.load(f)
    cls = VariantConfig if 'coordination' in model['config'] else TrainConfig
    cfg = cls(**model['config'])
    if context_changes:
        cfg = replace(cfg, **context_changes)
    state = jax.tree_util.tree_map(jax.numpy.asarray, model['state'])
    return cfg, state


def evaluation_runner(config):
    return VariantExperimentRunner(config) if isinstance(config, VariantConfig) else ExperimentRunner(config)
