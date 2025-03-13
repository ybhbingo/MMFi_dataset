import os
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config


class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


class MMFi_Dataset(Dataset):
    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                if self.data_unit == 'sequence':
                    data_dict = {'modality': self.modality,
                                 'scene': self.get_scene(subject),
                                 'subject': subject,
                                 'action': action,
                                 'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, 'ground_truth.npy')
                                 }
                    for mod in self.modality:
                        data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, mod)
                    data_info.append(data_dict)
                elif self.data_unit == 'frame':
                    frame_num = 297
                    for idx in range(frame_num):
                        data_dict = {'modality': self.modality,
                                     'scene': self.get_scene(subject),
                                     'subject': subject,
                                     'action': action,
                                     'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                             action, 'ground_truth.npy'),
                                     'idx': idx
                                     }
                        data_valid = True
                        for mod in self.modality:
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, "frame{:03d}".format(idx+1) + self.get_data_type(mod))

                            if os.path.getsize(data_dict[mod+'_path']) == 0:
                                data_valid = False
                        if data_valid:
                            data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
        return data_info

    def read_dir(self, dir):
        _, mod = os.path.split(dir)
        data = []
        if mod in ['infra1', 'infra2', 'rgb']:  # rgb, infra1 and infra2 are 2D keypoints
            for arr_file in sorted(glob.glob(os.path.join(dir, "frame*.npy"))):
                arr = np.load(arr_file)
                data.append(arr)
            data = np.array(data)
        elif mod == 'depth':
            for img in sorted(glob.glob(os.path.join(dir, "frame*.png"))):
                _cv_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)  # Default depth value is 16-bit
                _cv_img *= 0.001  # Convert unit to meter
                data.append(_cv_img)
            data = np.array(data)
        elif mod == 'lidar':
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 3)
                data.append(data_tmp)
        elif mod == 'mmwave':
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.copy().reshape(-1, 5)
                    # data_tmp = data_tmp[:, :3]
                data.append(data_tmp)
        elif mod == 'wifi-csi':
            for csi_mat in sorted(glob.glob(os.path.join(dir, "frame*.mat"))):
                data_mat = scio.loadmat(csi_mat)['CSIamp']
                data_mat[np.isinf(data_mat)] = np.nan
                for i in range(10):  # 32
                    temp_col = data_mat[:, :, i]
                    nan_num = np.count_nonzero(temp_col != temp_col)
                    if nan_num != 0:
                        temp_not_nan_col = temp_col[temp_col == temp_col]
                        temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
                data_mat = (data_mat - np.min(data_mat)) / (np.max(data_mat) - np.min(data_mat))
                data_frame = np.array(data_mat)
                data.append(data_frame)
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            data = cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.copy().reshape(-1, 5)
                # data = data[:, :3]
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(data)] = np.nan
            for i in range(10):  # 32
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)

        if self.data_unit == 'sequence':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'output': gt_torch
                      }
            for mod in item['modality']:
                data_path = item[mod+'_path']
                if os.path.isdir(data_path):
                    data_mod = self.read_dir(data_path)
                else:
                    data_mod = np.load(data_path + '.npy')
                sample['input_'+mod] = data_mod
        elif self.data_unit == 'frame':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'idx': item['idx'],
                      'output': gt_torch[item['idx']]
                      }
            for mod in item['modality']:
                data_path = item[mod + '_path']
                if os.path.isfile(data_path):
                    data_mod = self.read_frame(data_path)
                    sample['input_'+mod] = data_mod
                else:
                    raise ValueError('{} is not a file!'.format(data_path))
        else:
            raise ValueError('Unsupport data unit!')
        return sample


def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
    train_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['val_dataset'])
    return train_dataset, val_dataset


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    batch_data = {'modality': batch[0]['modality'],
                  'scene': [sample['scene'] for sample in batch],
                  'subject': [sample['subject'] for sample in batch],
                  'action': [sample['action'] for sample in batch],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                  }
    _output = [np.array(sample['output']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output

    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 0, 2)
            batch_data['input_' + mod] = _input
        else:
            _input = [np.array(sample['input_' + mod]) for sample in batch]
            _input = torch.FloatTensor(np.array(_input))
            batch_data['input_' + mod] = _input

    return batch_data

def make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader


