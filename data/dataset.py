import os
import json
from .example import Example
from .dataset import PairedDataset


class IUXray(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_file, split_ratio=(0.7, 0.15, 0.15), seed=42):
        # Load full annotation JSON
        with open(ann_file, 'r') as f:
            annotations = json.load(f)['annotations']

        # Extract (image, caption) pairs
        examples = [Example.fromdict({
            'image': os.path.join(img_root, ann['image_id']),
            'text': ann['caption']
        }) for ann in annotations]

        # Shuffle and split
        import random
        random.seed(seed)
        random.shuffle(examples)
        total = len(examples)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)

        self.train_examples = examples[:train_end]
        self.val_examples = examples[train_end:val_end]
        self.test_examples = examples[val_end:]

        all_examples = self.train_examples + self.val_examples + self.test_examples
        super(IUXray, self).__init__(all_examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train = PairedDataset(self.train_examples, self.fields)
        val = PairedDataset(self.val_examples, self.fields)
        test = PairedDataset(self.test_examples, self.fields)
        return train, val, test
