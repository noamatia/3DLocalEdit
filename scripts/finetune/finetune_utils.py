import torch
import argparse
import pandas as pd
from datetime import datetime
from models.point_e import PointE
from datasets.shapenet import ShapeNet
from models.control_point_e import ControlPointE
from datasets.control_shapenet import ControlShapeNet
from models.lora_control_point_e import LoraControlPointE
from datasets.lora_control_shapenet import LoraControlShapeNet
from models.masked_lora_control_point_e import MaskedLoraControlPointE
from datasets.masked_lora_control_shapenet import MaskedLoraControlShapeNet

from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
torch.set_float32_matmul_precision("high")


MODEL_TYPE_DICT = {
    POINT_E: (PointE, ShapeNet),
    CONTROL_POINT_E: (ControlPointE, ControlShapeNet),
    LORA_CONTROL_POINT_E: (LoraControlPointE, LoraControlShapeNet),
    MASKED_LORA_CONTROL_POINT_E: (MaskedLoraControlPointE, MaskedLoraControlShapeNet),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
        help="The model type.",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="The trainning data csv.",
    )
    parser.add_argument(
        "--data_csv_val",
        type=str,
        help="The validation data csv. If not provided, the training data will be used.",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        help="The model weights. Currently only for ControlPointE model.",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=10,
        help="The number of validation samples",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=50,
        help="The validation frequency, testing the model on the validation dataset and saving the model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="The number of epochs. Each epoch is a full pass over the trainning dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="The batch size for training, using gradient accumulation.",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=11,
        help="The grad accumolation steps, the number of batches to accumulate before taking a step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="The learning rate for the model training.",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.5,
        help="The conditional dropout probability, the probability to ignore the prompt.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1024,
        help="The timesteps for the diffusion model.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="The number of points in the point clouds.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        help="The subset size of the dataset, if not set, the full dataset will be used.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="The alpha value for LoRA network model.",
    )
    parser.add_argument(
        "--rank", type=int, default=4, help="The rank value for the LoRA network model."
    )
    parser.add_argument(
        "--positive_scale",
        type=int,
        default="1",
        help="The positive scale value for the LoRA network model.",
    )
    parser.add_argument(
        "--negative_scale",
        type=int,
        default="-1",
        help="The negative scale value for the LoRA network model.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="The beta value for the regularization term.",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="chair_arm",
        help="The part name to use for PartNet.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="3DLocalEdit",
        help="Wandb project name to use.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, the trainning process will be logged to wandb and the model will be saved at output directory.",
    )
    args = parser.parse_args()
    return args


def build_name(args, model_type):
    date_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    dataset_str = args.data_csv.replace("/", "_").replace(".csv", "")
    name = f"{date_str}_{dataset_str}_{model_type}"
    return name


def buid_dataset_kwargs(device, args, data_csv):
    df = pd.read_csv(os.path.join(DATA_DIR, data_csv))
    dataset_kwargs = dict(
        batch_size=args.batch_size,
        df=df,
        device=device,
        num_points=args.num_points,
        subset_size=args.subset_size,
    )
    if args.model_type == MASKED_LORA_CONTROL_POINT_E:
        dataset_kwargs["part"] = args.part
    return dataset_kwargs


def buid_model_kwargs(device, args, dataset_val, output_dir):
    model_kwargs = dict(
        lr=args.lr,
        device=device,
        output_dir=output_dir,
        val_freq=args.val_freq,
        dataset_val=dataset_val,
        use_wandb=args.use_wandb,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        num_val_samples=args.num_val_samples,
        cond_drop_prob=args.cond_drop_prob,
    )
    if args.model_type != POINT_E:
        model_kwargs["model_weights"] = args.model_weights
    if (
        args.model_type == LORA_CONTROL_POINT_E
        or args.model_type == MASKED_LORA_CONTROL_POINT_E
    ):
        model_kwargs["rank"] = args.rank
        model_kwargs["alpha"] = args.alpha
        model_kwargs["negative_scale"] = args.negative_scale
        model_kwargs["positive_scale"] = args.positive_scale
        if args.model_type == "masked_lora_control_point_e":
            model_kwargs["beta"] = args.beta
    return model_kwargs
