import wandb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud
from point_e.util.plotting import plot_point_cloud


class ShapeNet(Dataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        device: torch.device,
        num_points: int = 1024,
        subset_size: int = None,
        num_val_samples: int = 0,
        pcs_path: str = '/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering'
    ):
        super().__init__()
        self.device = device
        self.pcs_path = pcs_path
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_val_samples = num_val_samples
        self.length = 0
        self.uids = []
        self.prompts = []
        self.val_uids = []
        self.val_prompts = []
        self.pc_encodings = []
        self.target_images = []
        df = pd.read_csv(f"datasets/data/train/{data_dir}.csv")
        self.fill_data(df, subset_size)
        self.fill_last_batch()
        if len(self.target_images) > 0:
            wandb.log({"target": self.target_images})

    def append_sample(self, prompt, uid):
        self.prompts.append(prompt)
        self.uids.append(uid)
        pc = self.create_pc(uid)
        if uid not in self.val_uids and len(self.target_images) < self.num_val_samples:
            self.val_uids.append(uid)
            self.val_prompts.append(prompt)
            image = self.create_log_pc_image(pc, prompt, uid)
            self.target_images.append(image)
        pc_encoding = pc.encode().to(self.device)
        self.pc_encodings.append(pc_encoding)
        self.length += 1

    def create_log_pc_image(self, pc, prompt, uid):
        fig = plot_point_cloud(pc,
                               fixed_bounds=((-0.5, -0.5, -0.5),
                                             (0.5, 0.5, 0.5)),
                               theta=np.pi * 3 / 2)
        caption = f"{uid.replace('/', '_')}_{prompt.replace(' ', '_')}"
        return wandb.Image(fig, caption=caption)

    def create_pc(self, uid):
        pc = PointCloud.load(f'{self.pcs_path}/{uid}.npz',
                             coords_key="pointcloud",
                             add_black_color=True,
                             axes=[2, 0, 1])
        return pc.random_sample(self.num_points)

    def fill_data(self, df, subset_size):
        for _, row in df.iterrows():
            for prompt, uid in [(row["negative_prompt"], row["negative_uid"]), (row["positive_prompt"], row["positive_uid"])]:
                if subset_size is not None and self.length == subset_size:
                    return
                self.append_sample(prompt, uid)

    def fill_last_batch(self):
        r = self.length % self.batch_size
        if r == 0:
            return
        q = self.batch_size - r
        prompts, uids = self.prompts.copy(), self.uids.copy()
        while True:
            for prompt, uid in zip(prompts, uids):
                if q == 0:
                    return
                self.append_sample(prompt, uid)
                q -= 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "prompts": self.prompts[index],
            "pc_encodings": self.pc_encodings[index],
        }
