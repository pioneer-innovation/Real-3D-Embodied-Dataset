import argparse
from os import path as osp
import numpy as np
import torch

# create testset of home1 and home2
from R3AD_create.active_R3AD_converter import create_indoor_info_file
data_path = '../data/'
create_indoor_info_file(data_path)
