import wandb
import torch
import random
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from point_e.util.plotting import plot_point_cloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.transformer import CLIPImagePointDiffusionTransformer
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion, mean_flat
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from datasets.masked_control_shapenet import (
    MASKS,
    PROMPTS,
    SOURCE_LATENTS,
    TARGET_LATENTS,
)

LOSS = "loss"
TEXTS = "texts"
SOURCE = "source"
TARGET = "target"
OUTPUT = "output"
GUIDANCE = "guidance"
TIMESTEPS = "timesteps"
TRAIN_LOSS = "train_loss"
MODEL_NAME = "base40M-textvec"
MASKED_SOURCE = "masked_source"
MASKED_TARGET = "masked_target"
COND_DROP_PROB = "cond_drop_prob"
MODEL_FINAL_PT = "model_final.pt"
REGULARIZATION_LOSS = "regularization_loss"


def eval_loss(name, outputs):
    return torch.stack([x[name] for x in outputs]).mean()


class MaskedControlPointE(pl.LightningModule):

    lr: float
    beta: float
    timesteps: int
    batch_size: int
    dev: torch.device
    sampler: PointCloudSampler
    diffusion: GaussianDiffusion
    model: CLIPImagePointDiffusionTransformer

    def __init__(
        self,
        lr: float,
        beta: float,
        timesteps: int,
        num_points: int,
        batch_size: int,
        dev: torch.device,
        cond_drop_prob: float,
        validation_data_loader: DataLoader,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.beta = beta
        self.timesteps = timesteps
        self.batch_size = batch_size
        self._init_model(cond_drop_prob, num_points)
        self._init_validation_data(validation_data_loader)

    def _init_model(self, cond_drop_prob, num_points):
        self.diffusion = diffusion_from_config(
            {**DIFFUSION_CONFIGS[MODEL_NAME], TIMESTEPS: self.timesteps}
        )
        self.model = model_from_config(
            {**MODEL_CONFIGS[MODEL_NAME], COND_DROP_PROB: cond_drop_prob}, self.dev
        )
        self.model.load_state_dict(load_checkpoint(MODEL_NAME, self.dev))
        self.model.create_control_layers()
        self.sampler = PointCloudSampler(
            device=self.dev,
            models=[self.model],
            guidance_scale=[3.0],
            num_points=[num_points],
            diffusions=[self.diffusion],
            aux_channels=["R", "G", "B"],
            model_kwargs_key_filter=[TEXTS],
        )

    def _init_validation_data(self, validation_data_loader):
        log_data = {SOURCE: [], TARGET: [], MASKED_SOURCE: [], MASKED_TARGET: []}
        for batch_idx, batch in enumerate(validation_data_loader):
            self.validation_step(batch, batch_idx)
            for mask, prompt, source_latent, target_latent in zip(
                batch[MASKS],
                batch[PROMPTS],
                batch[SOURCE_LATENTS],
                batch[TARGET_LATENTS],
            ):
                for name, latent in zip(
                    [SOURCE, TARGET, MASKED_SOURCE, MASKED_TARGET],
                    [
                        source_latent,
                        target_latent,
                        source_latent * mask,
                        target_latent * mask,
                    ],
                ):
                    log_data[name].append(self._plot([latent], prompt))
        wandb.log(log_data, step=None)

    def _plot(self, samples, prompt):
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, theta=np.pi * 1 / 2)
        img = wandb.Image(fig, caption=prompt)
        plt.close()
        return img

    def _sample_t(self):
        return (
            torch.tensor(random.sample(range(self.timesteps), self.batch_size))
            .to(self.dev)
            .detach()
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        masks, prompts, source_latents, target_latents = (
            batch[MASKS],
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        terms = self.diffusion.training_losses(
            model=self.model,
            t=self._sample_t(),
            x_start=target_latents,
            model_kwargs={TEXTS: prompts, GUIDANCE: source_latents},
        )
        loss = terms[LOSS].mean()
        reg_loss = mean_flat((masks * (terms[OUTPUT] - source_latents)) ** 2).mean()
        train_loss = self.beta * loss + (1 - self.beta) * reg_loss
        wandb.log(
            {
                LOSS: loss,
                TRAIN_LOSS: train_loss,
                REGULARIZATION_LOSS: reg_loss,
            }
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0, "Validation data loader should have a single batch"
        log_data = {OUTPUT: []}
        with torch.no_grad():
            for prompt, source_latent in zip(batch[PROMPTS], batch[SOURCE_LATENTS]):
                samples = None
                for x in self.sampler.sample_batch_progressive(
                    batch_size=1,
                    model_kwargs={TEXTS: [prompt]},
                    guidances=[source_latent],
                ):
                    samples = x
                log_data[OUTPUT].append(self._plot(samples, prompt))
        wandb.log(log_data, step=None)
