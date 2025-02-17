"""
A helper function to get a default model for quick testing
"""
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from matanyone.model.matanyone import MatAnyone
from matanyone.inference.utils.args_utils import get_dataset_cfg

def get_matanyone_model(ckpt_path, device) -> MatAnyone:
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_our_config")
    cfg = compose(config_name="eval_matanyone_config")
    
    with open_dict(cfg):
        cfg['weights'] = ckpt_path

    # Load the network weights
    matanyone = MatAnyone(cfg, single_object=True).to(device).eval()
    model_weights = torch.load(cfg.weights, map_location=device)
    matanyone.load_weights(model_weights)

    return matanyone
