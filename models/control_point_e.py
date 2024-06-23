from models.point_e import *


class ControlPointE(PointE):
    """
    ControlPointE class represents a PointE model with control using cross entity attention.

    It inherits from the PointE class and adds additional functionality specific to controled generation.

    Attributes:
        model_weights (str): The model weights.
    """

    def __init__(self, model_weights, **kwargs):
        super().__init__(**kwargs)
        self.model.create_control_layers()
        if model_weights is not None:
            self.model.load_state_dict(
                torch.load(os.path.join(MODELS_WEIGHTS_DIR, model_weights))
            )

    def init_wandb(self):
        """
        Initializes the Weights & Biases (wandb) logs images.

        This method logs images of the grountruth and the guidance of the validation dataset.
        """
        super().init_wandb()
        self.log_pc_images(
            lambda item: self.dataset_val.create_pc(item[GUIDANCE_UID]), SOURCE
        )

    def build_sample_kwargs(self, item):
        """
        Builds keyword arguments for sampling includes the prompt text and the guidance pc.

        Args:
            item: A ShapeNet dataset item.

        Returns:
            kwargs: The keyword arguments for sampling.
        """
        kwargs = super().build_sample_kwargs(item)
        kwargs[GUIDANCES] = [item[GUIDANCE]]
        return kwargs
