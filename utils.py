import os
from point_e.util.point_cloud import PointCloud


CPU = "cpu"
UID = "uid"
MLP = "MLP"
CUDA = "cuda"
LOSS = "loss"
LORA = "lora"
ALPHA = "alpha"
TEXTS = "texts"
SCALE = "scale"
PARAMS = "params"
LINEAR = "Linear"
TARGET = "target"
SOURCE = "source"
OUTPUT = "output"
X_START = "x_start"
POINT_E = "point_e"
NEGATIVE = "negative"
POSITIVE = "positive"
GUIDANCE = "guidance"
GUIDANCES = "guidances"
TIMESTEPS = "timesteps"
UTTERANCE = "utterance"
TRAIN_LOSS = "train_loss"
TARGET_UID = "target_uid"
SOURCE_UID = "source_uid"
SAMPLE_TYPE = "sample_type"
MASKED_LOSS = "masked_loss"
TRANSFORMER = "Transformer"
SAFETENSORS = ".safetensors"
GUIDANCE_UID = "guidance_uid"
MODEL_KWARGS = "model_kwargs"
MODEL_NAME = "base40M-textvec"
MASK_ENCODING = "mask_encoding"
MASKED_SOURCE = "masked_source"
MASKED_TARGET = "masked_target"
COND_DROP_PROB = "cond_drop_prob"
MODEL_FINAL_PT = "model_final.pt"
CONTROL_POINT_E = "control_point_e"
NEGATIVE_OUTPUT = "negative_output"
POSITIVE_OUTPUT = "positive_output"
PARTNET_DIR = "/scratch/noam/data_v0"
MASKED_LABELS_JSON = "masked_labels.json"
MULTIHEAD_ATTENTION = "MultiheadAttention"
TRAIN_LOSS_NEGATIVE = "train_loss_negative"
TRAIN_LOSS_POSITIVE = "train_loss_positive"
LORA_CONTROL_POINT_E = "lora_control_point_e"
RESIDUAL_ATTENTION_BLOCK = "ResidualAttentionBlock"
DATA_DIR = "/home/noamatia/repos/point-e/datasets/data"
MODELS_WEIGHTS_DIR = "/scratch/noam/3d_local_edit/outputs"
MASKED_LORA_CONTROL_POINT_E = "masked_lora_control_point_e"
MULTIHEAD_CROSS_ENTITY_ATTENTION = "MultiheadCrossEntityAttention"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"
PARTNET_MASKED_LABELS_DIR = "/home/noamatia/repos/point-e/datasets/partnet"
RESIDUAL_CROSS_ENTITY_ATTENTION_BLOCK = "ResidualCrossEntityAttentionBlock"

SAMPLE_TYPES = [NEGATIVE, POSITIVE]
MODEL_TYPES = [
    POINT_E,
    CONTROL_POINT_E,
    LORA_CONTROL_POINT_E,
    MASKED_LORA_CONTROL_POINT_E,
]
LORA_TARGET_MODULES_REPLACE = [
    LINEAR,
    RESIDUAL_ATTENTION_BLOCK,
    RESIDUAL_CROSS_ENTITY_ATTENTION_BLOCK,
    MULTIHEAD_ATTENTION,
    MULTIHEAD_CROSS_ENTITY_ATTENTION,
    TRANSFORMER,
    MLP,
]


def load_masked_pc(src_dir, masked_labels):
    pc = PointCloud.load_partnet(
        os.path.join(src_dir, "point_sample", "pts-10000.txt"),
        labels_path=os.path.join(src_dir, "point_sample", "label-10000.txt"),
        masked_labels=masked_labels,
        axes=[2, 0, 1],
    )
    return pc
