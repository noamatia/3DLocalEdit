import os
import torch
import argparse
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str,
                        default="control_point_e", help="model type")
    parser.add_argument("--dataset", type=str,
                        default="chair/armrests/v1", help="dataset name")
    parser.add_argument("--num_val_samples", type=int,
                        default=1, help="number of validation samples")
    parser.add_argument("--val_freq", type=int,
                        default=100, help="validation step frequency")
    parser.add_argument("--epochs", type=int,
                        default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=6, help="batch size")
    parser.add_argument("--grad_acc_steps", type=int,
                        default=11, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float,
                        default=7e-5*0.4, help="learning rate")
    parser.add_argument("--cond_drop_prob", type=float,
                        default=0.1, help="prompt dropout probability")
    parser.add_argument("--timesteps", type=int,
                        default=1024, help="number of diffusion timesteps")
    parser.add_argument("--num_points", type=int,
                        default=1024, help="number of points in the point cloud")
    parser.add_argument("--subset_size", type=int,
                        default=None, help="subset size of the dataset")
    parser.add_argument("--alpha", type=float,
                        default=1.0, help="LoRA alpha")
    parser.add_argument("--rank", type=int,
                        default=4, help="LoRA rank")
    parser.add_argument("--positive_scale", type=int,
                        default="1", help="positive scale for the LoRA slider")
    parser.add_argument("--negative_scale", type=int,
                        default="-1", help="negative scale for the LoRA slider")
    parser.add_argument("--wandb_project", type=str,
                        default="3DLocalEdit", help="wandb project name")
    parser.add_argument("--use_wandb", action="store_true",
                        help="run in wndb and save outputs")
    args = parser.parse_args()
    return args


def build_name(args, model_type):
    date_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    dataset_str = args.dataset.replace("/", "_")
    name = f"{date_str}_{dataset_str}_{model_type}"
    return name
