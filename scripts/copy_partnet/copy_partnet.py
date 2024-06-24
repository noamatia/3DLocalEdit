import tqdm
import pandas as pd

from scripts.copy_partnet.copy_partnet_utils import *


def main(args):
    df = pd.read_csv(os.path.join(DATA_DIR, args.dataset))
    metadata_uid_to_uid = build_metadata_uid_to_uid(df)
    partnet_uids = os.listdir(PARTNET_DIR)
    for partnet_uid in tqdm.tqdm(
        partnet_uids, total=len(partnet_uids), desc="Copying PartNet"
    ):
        src_dir = os.path.join(PARTNET_DIR, partnet_uid)
        metadata = load_json(os.path.join(src_dir, META_JSON))
        model_id = metadata[MODEL_ID]
        if model_id in metadata_uid_to_uid:
            tgt_dir = os.path.join(
                PARTNET_MASKED_LABELS_DIR, metadata_uid_to_uid[model_id]
            )
            os.makedirs(tgt_dir, exist_ok=True)
            metadata = load_json(os.path.join(src_dir, RESULT_JSON))
            leaf_ids = build_leaf_ids(metadata, args.part)
            json_path = os.path.join(tgt_dir, MASKED_LABELS_JSON)
            build_and_save_json_data(json_path, partnet_uid, leaf_ids, args.part)
            build_and_save_pc(src_dir, tgt_dir, leaf_ids, args.part)


if __name__ == "__main__":
    args = parse_args()
    main(args)
