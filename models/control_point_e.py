import wandb
from models.point_e import PointE


class ControlPointE(PointE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.create_control_layers()

    def init_wandb(self):
        super().init_wandb()
        images = []
        for item in self.val_items:
            pc = self.dataset.create_pc(item["guidance_uid"])
            image = self.create_log_pc_image(pc, item["texts"])
            images.append(image)
        wandb.log({"source": images})

    def build_sample_kwargs(self, item):
        kwargs = super().build_sample_kwargs(item)
        kwargs["guidances"] = [item["guidance"]]
        return kwargs
