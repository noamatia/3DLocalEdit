import json
import argparse
import numpy as np
from point_e.util.plotting import plot_point_cloud

from utils import *

MODEL_ID = "model_id"
META_JSON = "meta.json"
RESULT_JSON = "result.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="paired/chair/train/armrests_20.csv",
        help="dataset name",
    )
    parser.add_argument("--part", type=str, default="chair_arm", help="part name")
    args = parser.parse_args()
    return args


def find_leaf_ids(node, target_name):
    leaf_ids = []
    if "name" in node and node["name"] == target_name:
        for child in node.get("children", []):
            if "children" in child:
                leaf_ids.extend(find_leaf_ids(child, target_name))
            else:
                leaf_ids.append(child["id"])
    else:
        for child in node.get("children", []):
            leaf_ids.extend(find_leaf_ids(child, target_name))
    return leaf_ids


def build_metadata_uid_to_uid(df):
    metadata_uid_to_uid = {}
    for _, row in df.iterrows():
        for sample_type in SAMPLE_TYPES:
            uid = row[f"{sample_type}_uid"]
            metadata_uid_to_uid[uid.split("/")[-1]] = uid
    return metadata_uid_to_uid


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_leaf_ids(metadata, part):
    leaf_ids = []
    for item in metadata:
        leaf_ids.extend(find_leaf_ids(item, part))
    return leaf_ids


def build_and_save_json_data(json_path, partnet_uid, leaf_ids, part):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        assert json_data["partnet_uid"] == partnet_uid
    else:
        json_data = {"masked_labels": {}, "partnet_uid": partnet_uid}
    json_data["masked_labels"][part] = leaf_ids
    with open(json_path, "w") as f:
        json.dump(json_data, f)


def build_and_save_pc(src_dir, tgt_dir, masked_labels, part):
    pc = load_masked_pc(src_dir, masked_labels)
    fig = plot_point_cloud(pc, theta=np.pi * 1 / 2)
    fig.savefig(os.path.join(tgt_dir, f"{part}.png"))
