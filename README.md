![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7983467.svg)](https://doi.org/10.5281/zenodo.7983467)

# Toolbox for MM-Fi Dataset

## Introduction

MM-Fi is the first multi-modal non-intrusive 4D human pose estimation dataset with 27 daily or rehabilitation action categories for high-level wireless human sensing tasks. MM-Fi consists of over 320k synchronized frames of five modalities from 40 human subjects in four domains. The annotations include 2D/3D human pose keypoints, 3D position, 3D dense pose, and the category of action. 

**MM-Fi has been published in NeurIPS 2023 Datasets and Benchmarks Track** _**[[Paper]](https://arxiv.org/abs/2305.10345)**_. 

For more details and demos about MMFi dataset, please refer to _**[[Project Page]](https://ntu-aiot-lab.github.io/mm-fi)**_.

<span id=download>Please download the dataset</span> through _**[[Google Drive]](https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz?usp=sharing)**_ or _**[[Baidu Netdisk]](https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316)**_. 

*The anonymized RGB images can be downloaded from _**[[Ali Cloud]](https://mmfi-dataset.oss-ap-southeast-1.aliyuncs.com/anonymized_rgb_images/all_images.zip)**_.

## Quick Start for MMFi Toolbox

To get started, follow the instructions in this section. We will introduce the simple steps and how you can customize the configuration.&#x20;

### Step 1

#### Dependencies

Please make sure you have installed the following dependencies before using MMFi dataset.&#x20;

* Python 3+ distribution
* Pytorch >= 1.1.0

Quick installation of depedencies (in one local or vitual environment)

```
pip install git+https://github.com/ybhbingo/MMFi_dataset
```

### Step 2

Once the environment is built successfully, download the [dataset](#download);

After unziping all four parts, the dataset directory structure should be as follows.&#x20;

#### Dataset Directory Structure

```
${DATASET_ROOT}
|-- E01
|   |-- S01
|   |   |-- A01
|   |   |   |-- rgb
|   |   |   |-- mmwave
|   |   |   |-- wifi-csi
|   |   |   |-- ...
|   |   |-- A02
|   |   |-- ...
|   |   |-- A27
|   |-- S02
|   |-- ...
|   |-- S10
|-- E02
|......
|-- E03
|......
|-- E04
|......
```

### Step 3

Edit your code and configuration file (_**.yaml**_ file) carefully before running. For details of the configuration, please check the [keys description](#keys_description).&#x20;

Here we just take the code snippets in the _**example.py**_ for instance.&#x20;

```
import yaml
import numpy as np
import torch

# Please add the downloaded mmfi directory into your python project. 
from mmfi import make_dataset, make_dataloader

dataset_root = '/data3/MMFi_Dataset'  # path will not be same in your server.
with open('config.yaml', 'r') as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)

train_dataset, val_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

# Coding

```

### Step 4

Now you can start the implementation. For example, using the commands below:&#x20;

```
cd your_project_dir
// make coding
python your_script_name.py path_to_dataset_dir your_config.yaml
```

## <span id=keys_description>Description of Keys in Configuration</span>

<mark style="color:green;">**`modality`**</mark>

*   **Single modality**

    Please use one of the followings:

    **rgb**, **infra1**, **infra2**, **depth**, **lidar**, **mmwave**, **wifi-csi**

    > `Note` that every modality should be in `lowercase`.
    >
    > **Currently, the raw images (rgb, infra1 and infra2) of subjects are not publicly available for the privacy concerns. Thus, we provide the 17 body keypoints extracted from images using the common ResNet-48 model.**&#x20;
*   **Multiple modalities**:

    Please use `|` to connect different modalities.

    > `Note` that `space` is not allowed in the connection. For example, `wifi-csi|mmwave` is OK, but `wifi-csi | mmwave` will not be accepted.

<mark style="color:green;">**`data_unit`**</mark>

*   **sequence**

    The data generator will return data with sequence as the unit, _e.g._, each sample contains **297** frames.
*   **frame**

    The data generator will return data with frame as the unit, _e.g._, each sample only has **1** frame.

<mark style="color:green;">**`protocol`**</mark>

This key defines how many [activities](#activity_list) could be enabled in your training/testing.&#x20;

* **protocol 1:** Only the daily activities are enabled.&#x20;
* **protocol 2:** Only the rehabilitation activities are enabled.
* **protocol 3:** All activities are enabled.

<mark style="color:green;">**`split`**</mark>

The train/test split of your code. There are already 3 splits which are used in our paper.&#x20;

* **manual\_split:** Please refer to the example in .yaml file and _**customize your own dataset split**_ setting here (which subjects and actions are regarded as the testing data).&#x20;
* **split\_to\_use:** Specify the split you want. &#x20;

<mark style="color:green;">**`train_loader`**</mark>    <mark style="color:green;">**`validation_loader`**</mark>

These two options define the parameters which are used to construct your dataloaders. We keep these two options open so that you could customize freely.&#x20;

## More Details about MMFi Dataset

### <span id=activity_list>Activities Included</span>

MMFi dataset constains two types of actions: _**daily activities**_ and _**rehabilitation activities**_.&#x20;

| Activity | Description                  | Category                  |
| -------- | ---------------------------- | ------------------------- |
| A01      | Stretching and relaxing      | Rehabilitation activities |
| A02      | Chest expansion(horizontal)  | Daily activities          |
| A03      | Chest expansion (vertical)   | Daily activities          |
| A04      | Twist (left)                 | Daily activities          |
| A05      | Twist (right)                | Daily activities          |
| A06      | Mark time                    | Rehabilitation activities |
| A07      | Limb extension (left)        | Rehabilitation activities |
| A08      | Limb extension (right)       | Rehabilitation activities |
| A09      | Lunge (toward left-front)    | Rehabilitation activities |
| A10      | Lunge (toward right-front)   | Rehabilitation activities |
| A11      | Limb extension (both)        | Rehabilitation activities |
| A12      | Squat                        | Rehabilitation activities |
| A13      | Raising hand (left)          | Daily activities          |
| A14      | Raising hand (right)         | Daily activities          |
| A15      | Lunge (toward left side)     | Rehabilitation activities |
| A16      | Lunge (toward right side)    | Rehabilitation activities |
| A17      | Waving hand (left)           | Daily activities          |
| A18      | Waving hand (right)          | Daily activities          |
| A19      | Picking up things            | Daily activities          |
| A20      | Throwing (toward left side)  | Daily activities          |
| A21      | Throwing (toward right side) | Daily activities          |
| A22      | Kicking (toward left side)   | Daily activities          |
| A23      | Kicking (toward right side)  | Daily activities          |
| A24      | Body extension (left)        | Rehabilitation activities |
| A25      | Body extension (right)       | Rehabilitation activities |
| A26      | Jumping up                   | Rehabilitation activities |
| A27      | Bowing                       | Daily activities          |

### Subjects and Environments

_**40 volunteers**_ (11 females and 29 males) aging from 23 to 40 participated in the data collection of MMFi. We appreciate their kind assitance in the completion of this work!&#x20;

In addition, the 40 volunteers were divided into 4 groups corresponding to 4 different environmental settings so that _**cross-domain**_ research could be conducted for the WiFi sensing.&#x20;

### Action Segments

We have also extracted the action segments from the raw sequences, with relevant information stored into a .csv form, which can be referred to in the [dataset directory](#download). 

## Reference

Please cite the following paper if you find MMFi dataset and toolbox benefit your research. Thank you for your support!&#x20;
```
@inproceedings{
    yang2023mm,
    title={MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
    author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=1uAsASS1th}
}
```
