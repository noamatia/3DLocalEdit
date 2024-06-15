import wandb
import pytorch_lightning as pl
from models.point_e import PointE
from datasets.shapenet import ShapeNet
from torch.utils.data import DataLoader

from scripts.finetune.utils import *


def main(args, name):
    output_dir = f"/home/noamatia/repos/point-e/outputs/{name}"
    if args.use_wandb:
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(project=args.wandb_project,
                   name=name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ShapeNet(data_dir=args.dataset,
                       batch_size=args.batch_size,
                       device=device,
                       num_points=args.num_points,
                       subset_size=args.subset_size,
                       num_val_samples=args.num_val_samples if args.use_wandb else 0)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True)
    model = PointE(lr=args.lr,
                   device=device,
                   dataset=dataset,
                   val_freq=args.val_freq,
                   use_wandb=args.use_wandb,
                   timesteps=args.timesteps,
                   num_points=args.num_points,
                   batch_size=args.batch_size,
                   cond_drop_prob=args.cond_drop_prob)
    trainer = pl.Trainer(accumulate_grad_batches=args.grad_acc_steps,
                         max_epochs=args.epochs)
    trainer.fit(model, data_loader)
    if args.use_wandb:
        torch.save(model.model.state_dict(), f"{output_dir}/model_final.pt")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    name = build_name(args, "finetune_point_e")
    main(args, name)
