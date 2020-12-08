#!/usr/bin/env python3

import os,sys
import torch

if __name__ == "__main__":
  while True:
    train_data_list = torch.load(sys.argv[1])
