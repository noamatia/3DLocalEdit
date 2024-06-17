from datasets.shapenet import ShapeNet, NEGATIVE


class ControlShapeNet(ShapeNet):
    def __getitem__(self, index):
        return self.get_item(index % self.length)

    def get_item(self, index):
        item = super().get_item(index)
        if self.sample_types[index] == NEGATIVE:
            item["guidance"] = self.pc_encodings[index + 1]
            item["guidance_uid"] = self.uids[index + 1]
        else:
            item["guidance"] = self.pc_encodings[index - 1]
            item["guidance_uid"] = self.uids[index - 1]
        return item
