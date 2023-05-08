import os
import argparse

import yaml
import numpy as np
import torch

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("dataset_root", type=str, help="Root of Dataset")
    parser.add_argument("config_file", type=str, help="Configuration YAML file")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # TODO: Settings, e.g., your model, optimizer, device, ...



    # TODO: Codes for training (and saving models)
    # Just an example for illustration.
    for batch_idx, batch_data in enumerate(train_loader):
        # Please check the data structure here.
        print(batch_data['output'])

    # TODO: Codes for test (if)




