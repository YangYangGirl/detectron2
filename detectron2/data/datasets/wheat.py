# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import pandas as pd
import itertools

__all__ = ["register_wheat", "load_wheat_instances"]


# fmt: off
CLASS_NAMES = [ "wheat",]
# fmt: on


def load_wheat_instances(df, image_dir):
    dicts = []
    for img_id, img_name in enumerate(df.image_id.unique()):

        record = {}
        image_df = df[df['image_id'] == img_name]
        img_path = image_dir + img_name + '.jpg'

        record['file_name'] = img_path
        record['image_id'] = img_id
        record['height'] = int(image_df['height'].values[0])
        record['width'] = int(image_df['width'].values[0])

        objs = []
        for _, row in image_df.iterrows():
            x_min = int(row.xmin)
            y_min = int(row.ymin)
            x_max = int(row.xmax)
            y_max = int(row.ymax)

            poly = [(x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max)]

            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }

            objs.append(obj)

        record['annotations'] = objs
        dicts.append(record)

    return dicts


def register_wheat(name, df, image_dir):
    DatasetCatalog.register(name, lambda: load_wheat_instances(df, image_dir))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES
    )
