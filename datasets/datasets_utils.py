from utils import *


def load_pc(uid):
    pc = PointCloud.load(
        os.path.join(PCS_DIR, f"{uid}.npz"),
        coords_key="pointcloud",
        add_black_color=True,
        axes=[2, 0, 1],
    )
    return pc


def sample_type_prompt(i):
    return f"{SAMPLE_TYPES[i]}_prompt"


def sample_type_uid(i):
    return f"{SAMPLE_TYPES[i]}_uid"
