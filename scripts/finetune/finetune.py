import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scripts.finetune.finetune_utils import *


def main(args, name, model_type, dataset_type):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=name, config=vars(args))
    output_dir = os.path.join(OUTPUTS_DIR, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_kwargs = buid_dataset_kwargs(device, args, args.data_csv)
    dataset = dataset_type(**dataset_kwargs)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    if args.data_csv_val is not None:
        dataset_val_kwargs = buid_dataset_kwargs(device, args, args.data_csv_val)
        dataset_val = dataset_type(**dataset_val_kwargs)
    else:
        dataset_val = dataset.copy()
    model_kwargs = buid_model_kwargs(device, args, dataset_val, output_dir)
    model = model_type(**model_kwargs)
    trainer = pl.Trainer(
        accumulate_grad_batches=args.grad_acc_steps, max_epochs=args.epochs
    )
    trainer.fit(model, data_loader)
    if args.use_wandb:
        model.save_weights(MODEL_FINAL_PT)
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    model_type, dataset_type = MODEL_TYPE_DICT[args.model_type]
    name = build_name(args, args.model_type)
    main(args, name, model_type, dataset_type)
