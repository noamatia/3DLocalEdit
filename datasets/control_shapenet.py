from datasets.shapenet import ShapeNet, NEGATIVE


class ControlShapeNet(ShapeNet):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        index = index % self.length
        if self.sample_types[index] == NEGATIVE:
            item["guidance"] = self.pc_encodings[index + 1]
            item["guidance_uid"] = self.uids[index + 1]
        else:
            item["guidance"] = self.pc_encodings[index - 1]
            item["guidance_uid"] = self.uids[index - 1]
        return item
