import logging
import os
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.affordance_dataset import AffordanceDataset

@registry.register_builder("affordance_dataset")
class AffordanceDatasetBuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/affordance/defaults.yaml"}
    def __init__(self, cfg=None):
        #super().__init__()
        self.config = cfg

    def build_datasets(self):
        datasets = self.build()
        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        #self.build_processors()

        build_info = self.config.build_info
        data_type = build_info.type
        ann_info = build_info.annotations

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # annotation path
            point_path = ann_info.get(split).point_path
            img_path = ann_info.get(split).img_path
            description_path = ann_info.get(split).description_path

            # create datasets
            datasets[split] = AffordanceDataset(split, data_type, point_path, img_path, description_path)

        return datasets

