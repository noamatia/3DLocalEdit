from datasets.control_shapenet import *


class LoraControlShapeNet(ControlShapeNet):
    """
    A dataset class for generating pairs of negative and positive samples from ShapeNet dataset for LoraControlPointE.
    """

    def set_length(self, subset_size):
        """
        Sets the length of the dataset using parent class method.
        Since each item includes a pair of negative and positive samples, the length is divided by 2.

        Args:
            subset_size (int, optional): The size of the subset to use from the prompts. If None, the length is set to half of the total number of prompts.
        """
        if subset_size is None:
            subset_size = len(self.prompts) // 2
        super().set_length(subset_size)

    def __getitem__(self, logical_index):
        index = self.eval_index(logical_index)
        negative_index = index * 2
        positive_index = negative_index + 1
        if self.paired:
            return {
                NEGATIVE: self.get_item_paired(negative_index),
                POSITIVE: self.get_item_paired(positive_index),
            }
        else:
            return {
                NEGATIVE: self.get_item_shapetalk(negative_index),
                POSITIVE: self.get_item_shapetalk(positive_index),
            }
