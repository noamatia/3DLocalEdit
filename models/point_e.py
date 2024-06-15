import wandb
import torch
import random
import torch.optim as optim
import pytorch_lightning as pl
from datasets.shapenet import ShapeNet
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config


class PointE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        device: torch.device,
        dataset: ShapeNet,
        val_freq: int,
        use_wandb: bool,
        timesteps: int,
        num_points: int,
        batch_size: int,
        cond_drop_prob: float,
    ):
        super().__init__()
        self.lr = lr
        self.dev = device
        self.dataset = dataset
        self.val_freq = val_freq
        self.use_wandb = use_wandb
        self.timesteps = timesteps
        self.num_points = num_points
        self.batch_size = batch_size
        self.step_count = 0
        self.losses = []
        model_name = "base40M-textvec"
        self.model = model_from_config(
            {**MODEL_CONFIGS[model_name], "cond_drop_prob": cond_drop_prob}, device)
        self.model.load_state_dict(load_checkpoint(model_name, device))
        self.diffusion = diffusion_from_config(
            {**DIFFUSION_CONFIGS[model_name], "timesteps": timesteps})
        self.sampler = PointCloudSampler(
            device=device,
            models=[self.model],
            diffusions=[self.diffusion],
            num_points=[num_points],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0],
            model_kwargs_key_filter=['texts'],
        )

    def training_step(self, batch, batch_idx):
        t = self.generate_t()
        model_kwargs = dict(texts=batch["prompts"])
        terms = self.diffusion.training_losses(
            self.model, batch["pc_encodings"], t, model_kwargs)
        loss = terms['loss'].mean()
        self.losses.append(loss.item())
        return loss

    def generate_t(self):
        return torch.tensor(random.sample(range(self.timesteps), self.batch_size)).to(self.dev).detach()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        train_loss = sum(self.losses) / len(self.losses)
        self.losses = []
        self.log("train_loss", train_loss)
        if self.use_wandb:
            log_data = {"train_loss": train_loss}
            if self.step_count % self.val_freq == 0:
                with torch.no_grad():
                    output_images = []
                    for prompt, uid in zip(self.dataset.val_prompts, self.dataset.val_uids):
                        samples = None
                        for x in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
                            samples = x
                        pc = self.sampler.output_to_point_clouds(samples)[0]
                        image = self.dataset.create_log_pc_image(
                            pc, prompt, uid)
                        output_images.append(image)
                    log_data["output"] = output_images
            self.step_count += 1
            wandb.log(log_data)
