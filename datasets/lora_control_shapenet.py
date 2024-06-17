from datasets.control_shapenet import ControlShapeNet


class LoraControlShapeNet(ControlShapeNet):
    def set_length(self, subset_size):
        if subset_size is None:
            subset_size = len(self.prompts) // 2
        super().set_length(subset_size)

    def __getitem__(self, index):
        index = index % self.length
        negative_index = index * 2
        positive_index = negative_index + 1
        return {
            "negative": super().get_item(negative_index),
            "positive": super().get_item(positive_index)
        }
