import wandb
import matplotlib.pyplot as plt
from point_e.util.plotting import plot_point_cloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

from utils import *


def build_model(cond_drop_prob, device):
    model = model_from_config(
        {**MODEL_CONFIGS[MODEL_NAME], COND_DROP_PROB: cond_drop_prob}, device
    )
    model.load_state_dict(load_checkpoint(MODEL_NAME, device))
    return model


def build_diffusion(timesteps):
    return diffusion_from_config(
        {**DIFFUSION_CONFIGS[MODEL_NAME], TIMESTEPS: timesteps}
    )


def build_pc_sampler(device, model, diffusion, num_points):
    return PointCloudSampler(
        device=device,
        models=[model],
        diffusions=[diffusion],
        num_points=[num_points],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0],
        model_kwargs_key_filter=[TEXTS],
    )


def pc_to_wanb_image(pc, prompt, theta):
    fig = plot_point_cloud(pc, theta=theta)
    img = wandb.Image(fig, caption=prompt)
    plt.close()
    return img


def build_weights_name(current_epocht):
    return f"model_step_{current_epocht}.pt"
