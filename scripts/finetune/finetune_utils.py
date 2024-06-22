import os
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
torch.set_float32_matmul_precision("high")

POINT_E = "point_e"
MODEL_FINAL_PT = "model_final.pt"
CONTROL_POINT_E = "control_point_e"
LORA_CONTROL_POINT_E = "lora_control_point_e"
MASKED_LORA_CONTROL_POINT_E = "masked_lora_control_point_e"
OUTPUTS_DIR = "/scratch/noam/3d_local_edit/outputs"
DATA_DIR = "/home/noamatia/repos/point-e/datasets/data/"
MODEL_TYPES = [
    POINT_E,
    CONTROL_POINT_E,
    LORA_CONTROL_POINT_E,
    MASKED_LORA_CONTROL_POINT_E,
]
MODEL_TYPE_DICT = {
    POINT_E: (PointE, ShapeNet),
    CONTROL_POINT_E: (ControlPointE, ControlShapeNet),
    LORA_CONTROL_POINT_E: (LoraControlPointE, LoraControlShapeNet),
    MASKED_LORA_CONTROL_POINT_E: (MaskedLoraControlPointE, MaskedLoraControlShapeNet),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=MODEL_TYPES)
    parser.add_argument("--data_csv", type=str, default="shapetalk/chair/train.csv")
    parser.add_argument("--data_csv_val", type=str, default="shapetalk/chair/val.csv")
    parser.add_argument("--num_val_samples", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_acc_steps", type=int, default=11)
    parser.add_argument("--lr", type=float, default=7e-5 * 0.4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--positive_scale", type=int, default="1")
    parser.add_argument("--negative_scale", type=int, default="-1")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--part", type=str, default="chair_arm")
    parser.add_argument("--wandb_project", type=str, default="3DLocalEdit")
    parser.add_argument("--use_wandb", action="store_true")
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
