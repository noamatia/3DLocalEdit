from models.lora import LoRANetwork

from models.control_point_e import *


class LoraControlPointE(ControlPointE):
    """
    A class representing a LoRa ControlPointe.

    Inherits from the ControlPointE class.

    Attributes:
        negative_scale (float): The negative scale value.
        positive_scale (float): The positive scale value.
        negative_losses (list): A list to store the negative losses.
        positive_losses (list): A list to store the positive losses.
        network (LoRANetwork): The LoRa network.
    """

    def __init__(self, rank, alpha, negative_scale, positive_scale, **kwargs):
        self.negative_scale = negative_scale
        self.positive_scale = positive_scale
        self.negative_losses = []
        self.positive_losses = []
        super().__init__(**kwargs)
        self.model.freeze_all_parameters()
        self.network = LoRANetwork(self.model, rank, alpha).to(self.device)

    def build_val_items(self):
        """
        Builds the validation items, NEGATIVE and POSITIVE alternating.

        Returns:
            val_items (list): The validation items.
        """
        idx = 0
        val_items = []
        while len(val_items) < 2 * self.num_val_samples and idx < len(self.dataset_val):
            negative_item = self.dataset_val[idx][NEGATIVE]
            positive_item = self.dataset_val[idx][POSITIVE]
            negative_item[SCALE] = self.negative_scale
            positive_item[SCALE] = self.positive_scale
            if negative_item[UID] not in [x[UID] for x in val_items]:
                val_items.append(negative_item)
            if positive_item[UID] not in [x[UID] for x in val_items]:
                val_items.append(positive_item)
            idx += 1
        return val_items

    def configure_optimizers(self):
        """
        Configures the optimizers.

        Returns:
            optim.Adam: The Adam optimizer.
        """
        return optim.Adam(self.network.prepare_optimizer_params(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step.

        Args:
            batch (Tensor): The input batch.
            batch_idx (int): The batch index.

        Returns:
            dict: The training step output.
        """
        self.network.set_lora_slider(self.negative_scale)
        with self.network:
            negative_dict = super().training_step(batch[NEGATIVE], batch_idx)
        negative_loss = negative_dict[LOSS]
        self.negative_losses.append(negative_loss.item())
        self.network.set_lora_slider(self.positive_scale)
        with self.network:
            positive_dict = super().training_step(batch[POSITIVE], batch_idx)
        positive_loss = positive_dict[LOSS]
        self.positive_losses.append(positive_loss.item())
        return {
            LOSS: negative_loss + positive_loss,
            NEGATIVE_OUTPUT: negative_dict[OUTPUT],
            POSITIVE_OUTPUT: positive_dict[OUTPUT],
        }

    def log_losses_and_clear(self):
        """
        Logs the losses and clears the lists.
        """
        super().log_losses_and_clear()
        self.log_losses(self.negative_losses, TRAIN_LOSS_NEGATIVE)
        self.negative_losses = []
        self.log_losses(self.positive_losses, TRAIN_LOSS_POSITIVE)
        self.positive_losses = []

    def sample(self, item):
        """
        Samples an item.

        Args:
            item (dict): The item to sample.

        Returns:
            Tensor: The sampled item.
        """
        self.network.set_lora_slider(item[SCALE])
        with self.network:
            return super().sample(item)

    def save_weights(self, name):
        """
        Saves the weights of the LoRa network.

        Args:
            name (str): The name of the weights file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.network.save_weights(os.path.join(self.output_dir, name))
