from datasets.shapenet import *


class ControlShapeNet(ShapeNet):
    """
    A class representing a dataset for ControlPointE.

    Inherits from the ShapeNet class.

    Attributes:
        sample_types (list): A list of sample types (negative or positive).
        guidance_uids (list): A list of guidance pc unique identifiers.
        guidance_pc_encodings (list): A list of guidance pc encodings.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ControlShapeNet object.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        self.sample_types = []
        self.guidance_uids = []
        self.guidance_pc_encodings = []
        super().__init__(**kwargs)

    def append_sample_paired(self, row):
        """
        Appends a paired sample to the dataset.

        Args:
            row (pd.Series): A row from the dataset (DataFrame).
        """
        super().append_sample_paired(row)
        for sample_type in SAMPLE_TYPES:
            self.sample_types.append(sample_type)

    def append_sample_shapetalk(self, row):
        """
        Appends a sample to the dataset using the Shapetalk format.

        Args:
            row (pd.Series): A row from the dataset (DataFrame).
        """
        super().append_sample_shapetalk(row)
        guidance_uid = row[SOURCE_UID]
        self.guidance_uids.append(guidance_uid)
        guidance_pc_encoding = self.create_pc_encoding(guidance_uid)
        self.guidance_pc_encodings.append(guidance_pc_encoding)

    def get_item_paired(self, index):
        """
        Returns the paired item at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: The paired item at the given index.

        Raises:
            ValueError: If the sample type is invalid.
        """
        item = self.get_item(index)
        if self.sample_types[index] == NEGATIVE:
            item[GUIDANCE] = self.pc_encodings[index + 1]
            item[GUIDANCE_UID] = self.uids[index + 1]
        elif self.sample_types[index] == POSITIVE:
            item[GUIDANCE] = self.pc_encodings[index - 1]
            item[GUIDANCE_UID] = self.uids[index - 1]
        else:
            raise ValueError("Invalid sample type")
        return item

    def get_item_shapetalk(self, index):
        """
        Returns the ShapeTalk item at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: The ShapeTalk item at the given index.
        """
        item = self.get_item(index)
        item[GUIDANCE] = self.guidance_pc_encodings[index]
        item[GUIDANCE_UID] = self.guidance_uids[index]
        return item

    def __getitem__(self, index):    
        logical_index = index % self.length
        if self.paired:
            return self.get_item_paired(logical_index)
        else:
            return self.get_item_shapetalk(logical_index)
