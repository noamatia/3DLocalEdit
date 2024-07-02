import os
import tqdm
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

MASKS = "masks"
PROMPTS = "prompt"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
PARTNET_UID = "partnet_uid"
MASKED_LABELS = "masked_labels"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"
PARTNET_DIR = "/scratch/noam/data_v0"
MASKED_LABELS_JSON = "masked_labels.json"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"
PARTNET_MASKED_LABELS_DIR = "/home/noamatia/repos/point-e/datasets/partnet"


def masked_labels_path(uid):
    return os.path.join(PARTNET_MASKED_LABELS_DIR, uid, MASKED_LABELS_JSON)


def load_partnet_metadata(uid, part, load_masked_label):
    path = masked_labels_path(uid)
    with open(path, "r") as f:
        data = json.load(f)
    masked_labels = data[MASKED_LABELS][part] if load_masked_label else None
    return masked_labels, data[PARTNET_UID]


def load_masked_pc(partnet_uid, shapnet_uid, masked_labels=None):
    src_dir = os.path.join(PARTNET_DIR, partnet_uid)
    pc = PointCloud.load_partnet(
        os.path.join(src_dir, "point_sample", "pts-10000.txt"),
        labels_path=os.path.join(src_dir, "point_sample", "label-10000.txt"),
        masked_labels=masked_labels,
        axes=[2, 0, 1],
    )
    shapenet_pc = PointCloud.load(
        os.path.join(PCS_DIR, f"{shapnet_uid}.npz"),
        coords_key="pointcloud",
        add_black_color=True,
        axes=[2, 0, 1],
    )
    pc = pc.random_sample(10000)
    pc.rescale(shapenet_pc)
    return pc


class MaskedControlShapeNet(Dataset):
    def __init__(
        self,
        part: str,
        num_points: int,
        batch_size: int,
        df: pd.DataFrame,
        subset_size: int,
        target_mask: bool,
        utterance_key: str,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        self.source_uids = []
        self.target_uids = []
        self.masks = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            self._append_sample(
                row, utterance_key, num_points, device, part, target_mask
            )
        self._set_length(subset_size, batch_size)

    def _append_sample(self, row, utterance_key, num_points, device, part, target_mask):
        prompt, source_uid, target_uid = (
            row[utterance_key],
            row[SOURCE_UID],
            row[TARGET_UID],
        )
        source_masked_labels, source_partnet_uid = load_partnet_metadata(
            source_uid, part, not target_mask
        )
        target_masked_labels, target_partnet_uid = load_partnet_metadata(
            target_uid, part, target_mask
        )
        self.prompts.append(prompt)
        self.source_uids.append(source_uid)
        self.target_uids.append(target_uid)
        source_pc = load_masked_pc(
            source_partnet_uid,
            source_uid,
            source_masked_labels,
        )
        target_pc = load_masked_pc(
            target_partnet_uid,
            target_uid,
            target_masked_labels,
        )
        source_pc = source_pc.random_sample(num_points)
        target_pc = target_pc.random_sample(num_points)
        self.source_latents.append(source_pc.encode().to(device))
        self.target_latents.append(target_pc.encode().to(device))
        if target_masked_labels is None:
            self.masks.append(source_pc.encode_mask().to(device))
        else:
            self.masks.append(target_pc.encode_mask().to(device))

    def _set_length(self, subset_size, batch_size):
        if subset_size is None:
            self.length = len(self.prompts)
        else:
            self.length = subset_size
        r = self.length % batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = batch_size - r
            self.logical_length = self.length + q

    def __len__(self):
        return self.logical_length

    def __getitem__(self, logical_index):
        index = logical_index % self.length
        return {
            MASKS: self.masks[index],
            PROMPTS: self.prompts[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
