import wandb
import torch
import random
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from datasets.shapenet import ShapeNet
from point_e.util.plotting import plot_point_cloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config


class PointE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        val_freq: int,
        timesteps: int,
        batch_size: int,
        use_wandb: bool,
        dataset: ShapeNet,
        device: torch.device,
        num_val_samples: int,
        cond_drop_prob: float,
    ):
        super().__init__()
        self.lr = lr
        self.dev = device
        self.dataset = dataset
        self.val_freq = val_freq
        self.use_wandb = use_wandb
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.num_val_samples = num_val_samples
        self.step_count = 0
        self.losses = []
        self.log_data = {}
        model_name = "base40M-textvec"
        self.model = model_from_config(
            {**MODEL_CONFIGS[model_name], "cond_drop_prob": cond_drop_prob}, device)
        self.model.load_state_dict(load_checkpoint(model_name, device))
        self.diffusion = diffusion_from_config(
            {**DIFFUSION_CONFIGS[model_name], "timesteps": timesteps})
        self.sampler = self.build_sampler()
        self.val_items = self.build_val_items()
        if use_wandb:
            self.init_wandb()

    def build_sampler(self):
        return PointCloudSampler(
            device=self.dev,
            models=[self.model],
            diffusions=[self.diffusion],
            num_points=[self.dataset.num_points],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0],
            model_kwargs_key_filter=['texts'],
        )

    def build_val_items(self):
        idx = 0
        val_items = []
        while len(val_items) < self.num_val_samples and idx < len(self.dataset):
            item = self.dataset[idx]
            if item["uid"] not in [x["uid"] for x in val_items]:
                val_items.append(item)
            idx += 1
        return val_items

    @staticmethod
    def create_log_pc_image(pc, prompt):
        fig = plot_point_cloud(pc,
                               theta=np.pi * 3 / 2)
        return wandb.Image(fig, caption=prompt)

    def init_wandb(self):
        images = []
        for item in self.val_items:
            pc = self.dataset.create_pc(item["uid"])
            image = self.create_log_pc_image(pc, item["texts"])
            images.append(image)
        wandb.log({"target": images})

    def training_step(self, batch, batch_idx):
        t = self.generate_t()
        terms = self.diffusion.training_losses(
            self.model, batch["x_start"], t, model_kwargs=batch)
        loss = terms['loss'].mean()
        self.losses.append(loss.item())
        return loss

    def generate_t(self):
        return torch.tensor(random.sample(range(self.timesteps), self.batch_size)).to(self.dev).detach()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def init_log_data(self):
        train_loss = sum(self.losses) / len(self.losses)
        self.losses = []
        self.log("train_loss", train_loss)
        self.log_data = {"train_loss": train_loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.init_log_data()
        if self.use_wandb:
            if self.step_count % self.val_freq == 0:
                with torch.no_grad():
                    images = []
                    for item in self.val_items:
                        samples = self.sample(item)
                        pcs = self.sampler.output_to_point_clouds(samples)
                        image = self.create_log_pc_image(pcs[0], item["texts"])
                        images.append(image)
                    self.log_data["output"] = images
            self.step_count += 1
            wandb.log(self.log_data)
        self.log_data = {}

    def sample(self, item):
        samples = None
        kwargs = self.build_sample_kwargs(item)
        for x in self.sampler.sample_batch_progressive(batch_size=1, **kwargs):
            samples = x
        return samples

    def build_sample_kwargs(self, item):
        kwargs = {"model_kwargs": {"texts": [item["texts"]]}}
        return kwargs
