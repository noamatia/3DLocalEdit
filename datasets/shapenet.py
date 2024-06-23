import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from datasets.datasets_utils import *


class ShapeNet(Dataset):
    """
    A PyTorch dataset class for PointE.

    Args:
        num_points (int): The number of points to sample from each point cloud.
        batch_size (int): The batch size for the dataset.
        subset_size (int): The size of the subset to use from the dataset.
        df (pd.DataFrame): The DataFrame containing the dataset.
        device (torch.device): The device to use for computations.

    Attributes:
        device (torch.device): The device to use for computations.
        num_points (int): The number of points to sample from each point cloud.
        batch_size (int): The batch size for the dataset.
        uids (list): A list to store the unique identifiers of the samples.
        prompts (list): A list to store the prompts of the samples.
        pc_encodings (list): A list to store the point cloud encodings of the samples.
        theta (float): The angle in radians used for creating point clouds.
        paired (bool): A flag indicating whether the dataset contains paired samples or ShapeTalk samples.
        length (int): The length of the dataset.
        logical_length (int): The logical length of the dataset.
    """

    def __init__(
        self,
        num_points: int,
        batch_size: int,
        subset_size: int,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.num_points = num_points
        self.batch_size = batch_size
        self.uids = []
        self.prompts = []
        self.pc_encodings = []
        self.theta = np.pi * 3 / 2
        self.paired = sample_type_prompt(0) in df.columns
        self.create_data(df)
        self.set_length(subset_size)

    def append_sample(self, prompt, uid, pc_encoding):
        """
        Append a sample to the dataset.

        Args:
            prompt (str): The prompt for the sample.
            uid (int): The unique identifier of the groundtruth pc of the sample.
            pc_encoding (list): The point cloud encoding of the groundtruth pc.
        """
        self.uids.append(uid)
        self.prompts.append(prompt)
        self.pc_encodings.append(pc_encoding)

    def create_pc(self, uid):
        """
        Create a point cloud with a specified number of points.

        Args:
            uid (str): The unique identifier of the point cloud.

        Returns:
            PointCloud: The randomly sampled point cloud with the specified number of points.
        """
        pc = load_pc(uid)
        return pc.random_sample(self.num_points)

    def create_pc_encoding(self, uid):
        """
        Creates a point cloud encoding for the given unique identifier (uid).

        Args:
            uid (str): The unique identifier of the point cloud.

        Returns:
            torch.Tensor: The encoded representation of the point cloud.
        """
        pc = self.create_pc(uid)
        return pc.encode().to(self.device)

    def append_sample_paired(self, row):
        """
        Appends a paired sample to the dataset.

        Args:
            row (pd.Series): A row from the dataset (DataFrame).
        """
        for i in range(len(SAMPLE_TYPES)):
            prompt = row[sample_type_prompt(i)]
            uid = row[sample_type_uid(i)]
            pc_encoding = self.create_pc_encoding(uid)
            self.append_sample(prompt, uid, pc_encoding)

    def append_sample_shapetalk(self, row):
        """
        Appends a sample to the dataset using the Shapetalk format.

        Args:
            row (pd.Series): A row from the dataset (DataFrame).
        """
        prompt = row[UTTERANCE]
        uid = row[TARGET_UID]
        pc_encoding = self.create_pc_encoding(uid)
        self.append_sample(prompt, uid, pc_encoding)

    def create_data(self, df):
        """
        Creates data based on the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
        """
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            # df is rows of pairs of negative and positive samples
            if self.paired:
                self.append_sample_paired(row)
            # df is ShapeTalk subset
            else:
                self.append_sample_shapetalk(row)

    def set_length(self, subset_size):
        """
        Sets the length of the dataset validating it is divisible by the batch size.

        Args:
            subset_size (int or None): The desired size of the subset. If None, the length is set to the total number of prompts.
        """
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

    def get_item(self, index):
        """
        Get an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the item information with the following keys:
                - UID: The unique identifier of the groundtruth pc of the item.
                - TEXTS: The prompts associated with the item.
                - X_START: The groundtruth pc encodings of the item.
        """
        item = {
            UID: self.uids[index],
            TEXTS: self.prompts[index],
            X_START: self.pc_encodings[index],
        }
        return item

    def eval_index(self, logical_index):
        """
        Get the index of the item in the dataset for evaluation.

        Args:
            logical_index (int): The logical index of the item to retrieve.

        Returns:
            int: The index of the item in the dataset for evaluation.
        """
        return logical_index % self.length

    def __len__(self):
        return self.logical_length

    def __getitem__(self, logical_index):
        index = self.eval_index(logical_index)
        self.get_item(index)
