import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud


NEGATIVE = "negative"
POSITIVE = "positive"
SAMPLE_TYPES = [NEGATIVE, POSITIVE]
DATASETS_DIR = 'datasets/data/train'
PCS_DIR = '/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering'


class ShapeNet(Dataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        device: torch.device,
        num_points: int = 1024,
        subset_size: int = None,
    ):
        super().__init__()
        self.device = device
        self.num_points = num_points
        self.batch_size = batch_size
        df = pd.read_csv(f"{DATASETS_DIR}/{data_dir}.csv")
        self.create_data(df)
        self.set_length(subset_size)

    def append_sample(self, prompt, uid):
        self.prompts.append(prompt)
        self.uids.append(uid)
        pc = self.create_pc(uid)
        pc_encoding = pc.encode().to(self.device)
        self.pc_encodings.append(pc_encoding)

    def create_pc(self, uid):
        pc = PointCloud.load(f'{PCS_DIR}/{uid}.npz',
                             coords_key="pointcloud",
                             add_black_color=True,
                             axes=[2, 0, 1])
        return pc.random_sample(self.num_points)

    def create_data(self, df):
        self.uids = []
        self.prompts = []
        self.pc_encodings = []
        for _, row in df.iterrows():
            for sample_type in SAMPLE_TYPES:
                prompt, uid = row[f"{sample_type}_prompt"], row[f"{sample_type}_uid"]
                self.append_sample(prompt, uid)

    def set_length(self, subset_size):
        if subset_size is None:
            self.length = len(self.prompts)
        else:
            self.length = subset_size
        r = self.length % self.batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = self.batch_size - r
            self.logical_length = self.length + q

    def __len__(self):
        return self.logical_length

    def __getitem__(self, index):
        index = index % self.length
        return {
            "uid": self.uids[index],
            "texts": self.prompts[index],
            "x_start": self.pc_encodings[index],
        }
