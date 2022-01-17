import argparse
from os import path as osp
import numpy as np
import torch

from R3AD_create.R3AD_converter import create_indoor_info_file
data_path = '../data/'
create_indoor_info_file(data_path)
