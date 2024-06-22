import os
import torch
import random
import torch.optim as optim
import pytorch_lightning as pl
from datasets.shapenet import ShapeNet

from models.models_utils import *


class PointE(pl.LightningModule):
    """
    PointE class is a PyTorch Lightning module that represents the PointE model.

    Args:
        lr (float): Learning rate for the optimizer.
        val_freq (int): Frequency of validation during training.
        timesteps (int): Number of diffusion timesteps.
        output_dir (str): Directory to save the model weights.
        batch_size (int): Batch size for training.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        device (torch.device): Device to run the model on.
        num_val_samples (int): Number of validation samples.
        cond_drop_prob (float): Conditional dropout probability.
        dataset_val (ShapeNet): Validation dataset.

    Attributes:
        lr (float): Learning rate for the optimizer.
        dev (torch.device): Device to run the model on.
        val_freq (int): Frequency of validation during training.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        timesteps (int): Number of diffusion timesteps.
        output_dir (str): Directory to save the model weights.
        batch_size (int): Batch size for training.
        dataset_val (ShapeNet): Validation dataset.
        num_val_samples (int): Number of validation samples.
        losses (list): List to store training losses.
        log_data (dict): Dictionary to store logged data.
        model (CLIPImagePointDiffusionTransformer): Pytorch model representing PointE.
        diffusion (GaussianDiffusion): Utilities for training and sampling diffusion models.
        sampler (PointCSampler): Point cloud sampler.
        val_items (list): List of validation items.
    """

    def __init__(
        self,
        lr: float,
        val_freq: int,
        timesteps: int,
        output_dir: str,
        batch_size: int,
        use_wandb: bool,
        device: torch.device,
        num_val_samples: int,
        cond_drop_prob: float,
        dataset_val: ShapeNet,
    ):
        super().__init__()
        self.lr = lr
        self.dev = device
        self.val_freq = val_freq
        self.use_wandb = use_wandb
        self.timesteps = timesteps
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.dataset_val = dataset_val
        self.num_val_samples = num_val_samples
        self.losses = []
        self.log_data = {}
        self.model = build_model(cond_drop_prob, self.dev)
        self.diffusion = build_diffusion(self.timesteps)
        self.sampler = build_pc_sampler(
            device, self.model, self.diffusion, dataset_val.num_points
        )
        self.val_items = self.build_val_items()
        if use_wandb:
            self.init_wandb()
            self.log_data_and_clear()

    def build_val_items(self):
        """
        Builds a list of validation items.

        Returns:
            val_items (list): A list of validation items with length of min(num_val_samples, len(dataset_val)).
        """
        idx = 0
        val_items = []
        while len(val_items) < self.num_val_samples and idx < len(self.dataset_val):
            item = self.dataset_val[idx]
            if item[UID] not in [x[UID] for x in val_items]:
                val_items.append(item)
            idx += 1
        return val_items

    def log_pc_images(self, create_pc_fn, log_key):
        """
        Logs images generated from point clouds.

        Args:
            create_pc_fn (function): A function that creates a point cloud from a ShapeNet dataset item.
            log_key (str): The key to store the logged images in the `log_data` dictionary.
        """
        images = []
        for item in self.val_items:
            pc = create_pc_fn(item)
            image = pc_to_wanb_image(pc, item[TEXTS], self.dataset_val.theta)
            images.append(image)
        self.log_data[log_key] = images

    def init_wandb(self):
        """
        Initializes the Weights & Biases (wandb) logs images.

        This method logs images of the grountruth of the validation dataset.
        """
        self.log_pc_images(lambda item: self.dataset_val.create_pc(item[UID]), TARGET)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the loss and output terms.
        """
        t = (
            torch.tensor(random.sample(range(self.timesteps), self.batch_size))
            .to(self.dev)
            .detach()
        )
        terms = self.diffusion.training_losses(
            self.model, batch[X_START], t, model_kwargs=batch
        )
        loss = terms[LOSS].mean()
        self.losses.append(loss.item())
        return {LOSS: loss, OUTPUT: terms[OUTPUT]}

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def log_losses(self, losses, name):
        """
        Logs the average loss.

        Args:
            losses (list): List of losses.
            name (str): Name of the loss.
        """
        if len(losses) == 0:
            return
        loss = sum(losses) / len(losses)
        self.log(name, loss)
        self.log_data[name] = loss

    def log_losses_and_clear(self):
        """
        Logs the average training loss and clears the losses list.
        """
        self.log_losses(self.losses, TRAIN_LOSS)
        self.losses = []

    def log_data_and_clear(self):
        """
        Logs the data and clears the log_data dictionary.
        """
        self.log_losses_and_clear()
        if self.use_wandb:
            wandb.log(self.log_data)
        self.log_data = {}

    def on_train_epoch_start(self) -> None:
        """
        Performs actions at the start of each training epoch.

        If `use_wandb` is True and a validation is due, it evals the model on the validation dataset,
        logs the output images and saves the model weights.
        """
        if self.use_wandb and self.current_epoch % self.val_freq == 0:
            with torch.no_grad():
                self.log_pc_images(
                    lambda item: self.sampler.output_to_point_clouds(self.sample(item))[
                        0
                    ],
                    OUTPUT,
                )
            self.save_weights(build_weights_name(self.current_epoch))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Performs actions at the end of each training batch.

        Logs data using `log_data_and_clear` method.
        """
        self.log_data_and_clear()

    def sample(self, item):
        """
        Evals the model on a single item.

        Args:
            item: A ShapeNet dataset item.

        Returns:
            samples: The sampled point clouds.
        """
        samples = None
        kwargs = self.build_sample_kwargs(item)
        for x in self.sampler.sample_batch_progressive(batch_size=1, **kwargs):
            samples = x
        return samples

    def build_sample_kwargs(self, item):
        """
        Builds keyword arguments for sampling includes the prompt text.

        Args:
            item: A ShapeNet dataset item.

        Returns:
            kwargs: The keyword arguments for sampling.
        """
        kwargs = {MODEL_KWARGS: {TEXTS: [item[TEXTS]]}}
        return kwargs

    def save_weights(self, name):
        """
        Saves the model weights.

        Args:
            name (str): The name of the weights file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))
