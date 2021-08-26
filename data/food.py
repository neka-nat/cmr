from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from skimage.io import imread
import yaml

from absl import flags

import torch
from torch.utils.data import Dataset

from . import base as base_data

# -------------- flags ------------- #
# ---------------------------------- #
kData = './misc/01_simulation'
curr_path = osp.dirname(osp.abspath(__file__))
flags.DEFINE_string('food_dir', osp.join(curr_path, '..', kData), 'Food Data Directory')

opts = flags.FLAGS


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class AnnoData:
    def __init__(self, basepath, rel_path):
        self.rel_path = rel_path
        self.basepath = basepath
        self.id = rel_path.split(".")[0]
        self._set_yaml()
        self._set_mask()

    def _set_yaml(self):
        yaml_path = osp.join(self.basepath, "Annotations", self.id + ".yaml")
        yaml_data = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
        self.bbox = BBox(*yaml_data["bbox"])
        self.parts = np.array(yaml_data["key_points"]).T
        self.scale = np.array([[yaml_data["camera_param"]["scale"]]])
        self.trans = np.array(yaml_data["camera_param"]["trans"])
        self.rot = np.array(yaml_data["camera_param"]["rot"])

    def _set_mask(self):
        mask_path = osp.join(self.basepath, "ForegroundMasks", self.id + ".png")
        self.mask = imread(mask_path)


class FoodDataset(base_data.BaseDataset):
    '''
    Food Data loader
    '''
    def __init__(self, opts, filter_key=None):
        super(FoodDataset, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.food_dir
        self.data_cache_dir = opts.cub_cache_dir

        self.img_dir = osp.join(self.data_dir, 'ColorImages')
        self.anno = []
        for f in os.listdir(self.img_dir):
            self.anno.append(AnnoData(self.data_dir, f))
        self.anno_sfm = self.anno

        self.filter_key = filter_key
        data_len = len(self.anno)
        self.num_imgs = data_len
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 7, 8]) - 1



#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(FoodDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(FoodDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(FoodDataset, batch_size, opts, filter_key='mask')
