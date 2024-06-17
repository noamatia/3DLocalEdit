import torch.optim as optim
from models.lora import LoRANetwork
from models.control_point_e import ControlPointE


class LoraControlPointE(ControlPointE):
    def __init__(self, rank, alpha, negative_scale, positive_scale, **kwargs):
        self.negative_scale = negative_scale
        self.positive_scale = positive_scale
        self.negative_losses = []
        self.positive_losses = []
        super().__init__(**kwargs)
        self.model.freeze_all_parameters()
        self.network = LoRANetwork(self.model, rank, alpha).to(self.device)

    def build_val_items(self):
        idx = 0
        val_items = []
        while len(val_items) < 2 * self.num_val_samples and idx < len(self.dataset):
            negative_item = self.dataset[idx]['negative']
            positive_item = self.dataset[idx]['positive']
            negative_item["scale"] = self.negative_scale
            positive_item["scale"] = self.positive_scale
            if negative_item["uid"] not in [x["uid"] for x in val_items]:
                val_items.append(negative_item)
            if positive_item["uid"] not in [x["uid"] for x in val_items]:
                val_items.append(positive_item)
            idx += 1
        return val_items

    def configure_optimizers(self):
        return optim.Adam(self.network.prepare_optimizer_params(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        self.network.set_lora_slider(self.negative_scale)
        with self.network:
            loss_negative = super().training_step(batch["negative"], batch_idx)
        self.negative_losses.append(loss_negative.item())
        self.network.set_lora_slider(self.positive_scale)
        with self.network:
            loss_positive = super().training_step(batch["positive"], batch_idx)
        self.positive_losses.append(loss_positive.item())
        return (loss_negative + loss_positive) / 2

    def init_log_data(self):
        super().init_log_data()
        train_loss_negative = sum(self.negative_losses) / \
            len(self.negative_losses)
        train_loss_positive = sum(self.positive_losses) / \
            len(self.positive_losses)
        self.negative_losses = []
        self.positive_losses = []
        self.log("train_loss_negative", train_loss_negative)
        self.log("train_loss_positive", train_loss_positive)
        self.log_data["train_loss_negative"] = train_loss_negative
        self.log_data["train_loss_positive"] = train_loss_positive

    def sample(self, item):
        samples = None
        kwargs = self.build_sample_kwargs(item)
        self.network.set_lora_slider(item["scale"])
        with self.network:
            for x in self.sampler.sample_batch_progressive(batch_size=1, **kwargs):
                samples = x
        return samples
