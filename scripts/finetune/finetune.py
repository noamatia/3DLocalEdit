import wandb
import pytorch_lightning as pl
from models.point_e import PointE
from datasets.shapenet import ShapeNet
from torch.utils.data import DataLoader
from models.control_point_e import ControlPointE
from datasets.control_shapenet import ControlShapeNet

from scripts.finetune.utils import *


def main(args, name, model_type, dataset_type):
    output_dir = f"/home/noamatia/repos/point-e/outputs/{name}"
    if args.use_wandb:
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(project=args.wandb_project,
                   name=name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_type(data_dir=args.dataset,
                           batch_size=args.batch_size,
                           device=device,
                           num_points=args.num_points,
                           subset_size=args.subset_size)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True)
    model = model_type(lr=args.lr,
                       device=device,
                       dataset=dataset,
                       val_freq=args.val_freq,
                       use_wandb=args.use_wandb,
                       timesteps=args.timesteps,
                       batch_size=args.batch_size,
                       num_val_samples=args.num_val_samples,
                       cond_drop_prob=args.cond_drop_prob)
    trainer = pl.Trainer(accumulate_grad_batches=args.grad_acc_steps,
                         max_epochs=args.epochs)
    trainer.fit(model, data_loader)
    if args.use_wandb:
        torch.save(model.model.state_dict(), f"{output_dir}/model_final.pt")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.model_type == "control_point_e":
        model_type = ControlPointE
        dataset_type = ControlShapeNet
    elif args.model_type == "point_e":
        model_type = PointE
        dataset_type = ShapeNet
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    name = build_name(args, args.model_type)
    main(args, name, model_type, dataset_type)
