import os
import wandb
import torch
import argparse
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models.masked_control_point_e import MaskedControlPointE
from datasets.masked_control_shapenet import (
    SOURCE_UID,
    TARGET_UID,
    masked_labels_path,
    MaskedControlShapeNet,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.set_float32_matmul_precision("high")
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"

OUTPUTS_DIR = "/scratch/noam/3d_local_edit/outputs"
DATASETS_DIR = "/home/noamatia/repos/point-e/datasets/data/shapetalk"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv",
        type=str,
        default="a_chair_with_armrests/train.csv",
        help="The trainning data csv.",
    )
    parser.add_argument(
        "--data_csv_val",
        type=str,
        # default="a_chair_with_armrests/test.csv",
        help="The validation data csv. If not provided, the training data will be used.",
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=1,
        help="The number of validation samples",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=10,
        help="The validation frequency for testing the model on the validation dataset.",
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
        default=1,
        help="The subset size of the dataset, if not set, the full dataset will be used.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.75,
        help="The beta value for the regularization term.",
    )
    parser.add_argument(
        "--target_mask",
        type=bool,
        default=True,
        help="Whether to use the target ot the source mask in the dataset.",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="chair_arm",
        help="The part name to use for PartNet.",
    )
    parser.add_argument(
        "--utterance_key",
        type=str,
        default="utterance",
        choices=["utterance", "llama3_uttarance"],
        help="The utterance key in the dataset.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MaskedControlPointE",
        help="Wandb project name to use.",
    )
    args = parser.parse_args()
    return args


def load_df(data_csv):
    df = pd.read_csv(os.path.join(DATASETS_DIR, data_csv))
    for uid_key in [SOURCE_UID, TARGET_UID]:
        df = df[
            df.apply(
                lambda row: os.path.exists(masked_labels_path(row[uid_key])),
                axis=1,
            )
        ]
    return df


def main(args, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_df(args.data_csv)
    train_dataset = MaskedControlShapeNet(
        df=df,
        part=args.part,
        device=device,
        num_points=args.num_points,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        target_mask=args.target_mask,
        utterance_key=args.utterance_key,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_df = load_df(args.data_csv_val) if args.data_csv_val else df
    validation_dataset = MaskedControlShapeNet(
        device=device,
        part=args.part,
        df=validation_df,
        num_points=args.num_points,
        target_mask=args.target_mask,
        utterance_key=args.utterance_key,
        batch_size=args.num_validation_samples,
        subset_size=args.num_validation_samples,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset, batch_size=args.num_validation_samples
    )
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    model = MaskedControlPointE(
        lr=args.lr,
        dev=device,
        beta=args.beta,
        timesteps=args.timesteps,
        num_points=args.num_points,
        batch_size=args.batch_size,
        cond_drop_prob=args.cond_drop_prob,
        validation_data_loader=validation_data_loader,
    )
    wandb.watch(model)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUTS_DIR, name),
        save_top_k=-1,
        every_n_epochs=args.epochs / 10,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.grad_acc_steps,
        check_val_every_n_epoch=args.validation_freq,
    )
    trainer.fit(model, train_data_loader, validation_data_loader)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    date_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    subset_size = "full" if args.subset_size is None else args.subset_size
    name = f"{date_str}_{subset_size}_{args.part}_{args.beta}_{args.target_mask}_{args.utterance_key}"
    main(args, name)
