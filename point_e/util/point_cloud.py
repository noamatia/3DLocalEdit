import random
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from .ply_util import write_ply

COLORS = frozenset(["R", "G", "B", "A"])


def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data


@dataclass
class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]
    mask: Optional[np.ndarray] = None

    @classmethod
    def load(cls, f: Union[str, BinaryIO], coords_key="coords", add_black_color=False, axes=None) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader, coords_key=coords_key, add_black_color=add_black_color, axes=axes)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            coords = obj[coords_key].astype(np.float32)
            if axes is not None:
                coords[:, [0, 1, 2]] = coords[:, axes]
            if add_black_color:
                channels = {k: np.zeros_like(coords[:, 0], dtype=np.float32) for k in [
                    "R", "G", "B"]}
            else:
                channels = {k: obj[k].astype(np.float32)
                            for k in keys if k != coords_key}
            return PointCloud(
                coords=coords,
                channels=channels,
            )

    @classmethod
    def load_partnet(cls, path: str, labels_path: str, masked_labels: list, axes=None, paint_labels=False) -> "PointCloud":
        """
        Load the partnet point cloud from a .txt file. 
        """
        coords = np.loadtxt(path, dtype=np.float32)
        labels = np.loadtxt(labels_path, dtype=int)
        if axes is not None:
            coords[:, [0, 1, 2]] = coords[:, axes]
        channels = {k: np.zeros_like(coords[:, 0], dtype=np.float32) for k in [
            "R", "G", "B"]}
        mask = np.isin(labels, masked_labels)
        if paint_labels:
            channels["R"][mask] = 1
        return PointCloud(
            coords=coords,
            channels=channels,
            mask=1 - mask.astype(int)
        )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(
            len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(
            len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
                mask=self.mask[indices] if self.mask is not None else None,
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(
            coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name)
                        for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            dists = norms + np.sum(batch**2, axis=-
                                   1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )

    def encode(self) -> torch.Tensor:
        """
        Encode the point cloud to a Kx6 tensor where K is the number of points.
        """
        coords = self.coords.T
        aux_list = []
        for name in self.channels:
            aux_data = self.channels[name]
            if name in {"R", "G", "B", "A"}:
                aux_data = (aux_data * 255).astype(np.uint8)
            aux_list.append(torch.tensor(aux_data, dtype=torch.float32))
        pos_tensor = torch.tensor(coords, dtype=torch.float32)
        if len(aux_list) == 0:
            return pos_tensor
        aux_tensor = torch.stack(aux_list, dim=0)
        return torch.cat([pos_tensor, aux_tensor], dim=0)

    def encode_mask(self) -> torch.Tensor:
        """
        Encode the mask to a tensor.
        """
        num_aux = 0
        for name in self.channels:
            if name in {"R", "G", "B", "A"}:
                num_aux += 1
        return torch.tensor(np.tile(self.mask, (num_aux + 3, 1)), dtype=torch.float32)

    def set_color_by_dist(self, other: "PointCloud"):
        """
        Set the color of each point based on the distance to the nearest point
        in another point cloud.
        """
        distances = np.sqrt(np.sum((self.coords - other.coords)**2, axis=1))
        norm = Normalize(vmin=distances.min(), vmax=distances.max())
        norm_distances = norm(distances)
        cmap = plt.get_cmap("turbo")
        rgb_values = cmap(norm_distances)[:, :3]
        self.channels["R"] = rgb_values[:, 0]
        self.channels["G"] = rgb_values[:, 1]
        self.channels["B"] = rgb_values[:, 2]