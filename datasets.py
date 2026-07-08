from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd
import json
import pickle
import random
import glob
from PIL import Image
from torch_videovision.torchvideotransforms import video_transforms, volume_transforms


def _get_mtl_transforms(train=True):
    if train:
        return video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((228, 128)),
            video_transforms.RandomCrop(112),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
        ])
    else:
        return video_transforms.Compose([
            video_transforms.Resize((228, 128)),
            video_transforms.CenterCrop(112),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
        ])


def stratified_label_split(labels, labeled_ratio=0.4, n_bins=5, seed=42):
    """
    对训练集做分层抽样，返回每个样本的 is_labeled 标志列表。
    labels: List of [name, score]
    返回: np.ndarray of bool, shape=(len(labels),)
    """
    scores = np.array([l[1] for l in labels])
    # 按分数分 n_bins 个 bin
    bin_edges = np.linspace(scores.min(), scores.max() + 1e-6, n_bins + 1)
    bin_ids = np.digitize(scores, bin_edges) - 1  # 0-indexed
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rng = np.random.RandomState(seed)
    is_labeled = np.zeros(len(labels), dtype=bool)

    for b in range(n_bins):
        indices = np.where(bin_ids == b)[0]
        if len(indices) == 0:
            continue
        n_labeled = max(1, int(round(len(indices) * labeled_ratio)))
        chosen = rng.choice(indices, size=n_labeled, replace=False)
        is_labeled[chosen] = True

    return is_labeled


class RGDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = os.path.join(video_feat_path, action_type + '_rgb_VST.npy')
        self.erase_path = video_feat_path + '_erTrue'
        self.score_range = 25.
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

        # 训练集做分层抽样划分有标签/无标签
        if self.train:
            self.is_labeled = stratified_label_split(
                self.labels,
                labeled_ratio=args.labeled_ratio,
                n_bins=5,
                seed=args.seed_data
            )
            n_labeled = self.is_labeled.sum()
            print(f"[RGDataset] Total train: {len(self.labels)}, "
                  f"Labeled: {n_labeled} ({n_labeled / len(self.labels) * 100:.1f}%), "
                  f"Unlabeled: {len(self.labels) - n_labeled}")
            # 将无标签数据的分数替换为 NaN
            for i, labeled in enumerate(self.is_labeled):
                if not labeled:
                    self.labels[i][1] = float('nan')
        else:
            # 测试集全部视为有标签
            self.is_labeled = np.ones(len(self.labels), dtype=bool)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx['Total_Score']])])
        return labels

    def _pad_or_crop(self, feat, target_len, random_crop):
        if len(feat) > target_len:
            if random_crop:
                st = np.random.randint(0, len(feat) - target_len + 1)
            else:
                st = (len(feat) - target_len) // 2
            feat = feat[st:st + target_len]
        elif len(feat) < target_len:
            new_feat = np.zeros((target_len, feat.shape[1]))
            new_feat[:feat.shape[0]] = feat
            feat = new_feat
        return feat

    def __getitem__(self, idx):
        video_feat = np.load(self.video_path, allow_pickle=True).item()[self.labels[idx][0]]

        random_crop = self.train
        video_feat = self._pad_or_crop(video_feat, self.clip_num, random_crop)
        video_feat = torch.from_numpy(video_feat).float()  # (T, D)

        label = self.normalize_score(self.labels[idx][1])
        is_labeled = bool(self.is_labeled[idx])

        # 关键：返回稳定 sample_index（=数据集内 idx）
        sample_index = int(idx)

        return video_feat, label, False, is_labeled, sample_index

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range


class FineFSDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26, action_type='TES', train=True, args=None):
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        score_type = args.score_type
        if score_type == "SP":
            score_idx = {'TES': 70, 'PCS': 50}
        else:
            score_idx = {'TES': 130, 'PCS': 100}
        self.score_range = score_idx[action_type]
        args.score_range = self.score_range
        self.clip_num = clip_num
        print(score_type)
        if score_type == "SP":
            ranges = np.arange(0, 729)
            np.random.seed(42)
            np.random.shuffle(ranges)
            if self.train:
                self.ran = ranges[:583]
            else:
                self.ran = ranges[583:]
        else:
            ranges = np.arange(729, 1167)
            np.random.seed(42)
            np.random.shuffle(ranges)
            if self.train:
                self.ran = ranges[:350]
            else:
                self.ran = ranges[350:]
        self.labels = self.read_label(label_path, action_type, score_type)

        # 训练集分层抽样
        if self.train:
            self.is_labeled = stratified_label_split(
                self.labels, labeled_ratio=args.labeled_ratio, n_bins=5, seed=args.seed_data
            )
            n_labeled = self.is_labeled.sum()
            print(f"[FineFSDataset] Total train: {len(self.labels)}, "
                  f"Labeled: {n_labeled} ({n_labeled / len(self.labels) * 100:.1f}%), "
                  f"Unlabeled: {len(self.labels) - n_labeled}")
            # 将无标签数据的分数替换为 NaN
            for i, labeled in enumerate(self.is_labeled):
                if not labeled:
                    self.labels[i][1] = float('nan')
        else:
            self.is_labeled = np.ones(len(self.labels), dtype=bool)

    def read_label(self, label_path, action_type, score_type):

        idx = {'TES': "total_element_score", 'PCS': "total_program_component_score(factored)"}
        labels = []
        score = []
        for i in self.ran:
            with open(os.path.join(label_path, str(i) + '.json')) as f:
                label = json.load(f)
            labels.append([str(i), float(label[idx[action_type]])])
            score.append(float(label[idx[action_type]]))
        print("max:", max(score))
        print("min:", min(score))
        return labels

    def __getitem__(self, idx):
        name = self.labels[idx][0]
        video_feat = torch.load(os.path.join(self.video_path, self.labels[idx][0] + '.pkl'), weights_only=True)

        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat

        is_labeled = bool(self.is_labeled[idx])
        # 关键：返回稳定 sample_index（=数据集内 idx）
        sample_index = int(idx)

        return video_feat, self.normalize_score(self.labels[idx][1]), False, is_labeled, sample_index

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range


# ── MTL-AQA 数据集 ───────────────────────────────────────────────
class MTLAQADataset(Dataset):
    """
    单样本读取流程（不读取对比样本）。
    从原始视频帧目录读取帧图像，__getitem__ 返回:
        frames_tensor : (C, T_frames, H, W)  待 I3D 提取特征
        label         : float, 归一化分数
        is_labeled    : bool
    I3D 特征提取在模型的 extract_feat 阶段完成（见 model.py）。
    """

    def __init__(self, args, train=True):
        self.train = train
        self.data_root = args.video_path  # 视频帧根目录
        self.label_path = args.label_path
        self.frame_length = args.frame_length
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.usingDD = args.usingDD

        # score_range：MTL-AQA final_score 最大值约 104.5，使用 100 归一化
        if self.usingDD:
            self.score_range = 30
        else:
            self.score_range = 104.5
        args.score_range = self.score_range

        self.label_dict = self._read_pickle(args.label_path)
        if train:
            self.dataset = self._read_pickle(args.train_split)
        else:
            self.dataset = self._read_pickle(args.test_split)

        self.transforms = _get_mtl_transforms(train)

        # 分层抽样划分有标签/无标签（仅训练集）
        if train:
            label_list = [[key, self.label_dict[key]['final_score']] for key in self.dataset]
            self.is_labeled = stratified_label_split(
                label_list,
                labeled_ratio=args.labeled_ratio,
                n_bins=5,
                seed=args.seed_data
            )
            n_labeled = self.is_labeled.sum()
            print(f"[MTLAQADataset] Total train: {len(self.dataset)}, "
                  f"Labeled: {n_labeled} ({n_labeled / len(self.dataset) * 100:.1f}%), "
                  f"Unlabeled: {len(self.dataset) - n_labeled}")
            # 将无标签数据的分数替换为 NaN
            for i, labeled in enumerate(self.is_labeled):
                if not labeled:
                    self.label_dict[self.dataset[i]]['final_score'] = float('nan')
        else:
            self.is_labeled = np.ones(len(self.dataset), dtype=bool)

    def _read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_video(self, video_file_name, phase):
        """
        读取帧图像，返回 PIL Image 列表（长度 = self.frame_length）。
        video_file_name: label_dict 中的 key，格式同 MTLPair.py
        """
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, str('{:02d}'.format(video_file_name[0])), '*.jpg'))))
        end_frame = self.label_dict.get(video_file_name).get('end_frame')
        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            end_frame = end_frame + temporal_aug_shift
        start_frame = end_frame - self.frame_length

        video = [Image.open(image_list[start_frame + i]) for i in range(self.frame_length)]
        return self.transforms(video)

    def __getitem__(self, idx):
        key = self.dataset[idx]
        phase = 'train' if self.train else 'test'
        video_tensor = self._load_video(key, phase)  # (C, T, H, W)

        if self.usingDD:
            diff = self.label_dict[key]['difficulty']
            final_score = self.label_dict[key]['completeness'] if 'completeness' in self.label_dict[key] \
                else self.label_dict[key]['final_score'] / diff
        else:
            final_score = self.label_dict[key]['final_score']
            diff = []

        label = self.normalize_score(final_score)
        is_labeled = bool(self.is_labeled[idx])
        # 关键：返回稳定 sample_index（=数据集内 idx）
        sample_index = int(idx)
        return video_tensor, label, diff, is_labeled, sample_index

    def __len__(self):
        return len(self.dataset)

    def normalize_score(self, score):
        return score / self.score_range


# ── FineDiving 数据集 ───────────────────────────────────────────────
class FineDivingDataset(Dataset):
    """
    单样本读取流程（不读取对比样本）。
    从原始视频帧目录读取帧图像，__getitem__ 返回:
        frames_tensor : (C, T_frames, H, W)  待 I3D 提取特征
        label         : float, 归一化分数
        is_labeled    : bool
    I3D 特征提取在模型的 extract_feat 阶段完成（见 model.py）。
    """

    def __init__(self, args, train=True):
        self.train = train
        self.data_root = args.video_path  # 视频帧根目录
        self.label_path = args.label_path
        self.frame_length = args.frame_length
        self.usingDD = args.usingDD

        # score_range：FineDiving final_score 最大值约 115
        if self.usingDD:
            self.score_range = 30
        else:
            self.score_range = 115
        args.score_range = self.score_range

        self.data_anno = self._read_pickle(args.label_path)
        if train:
            self.dataset = self._read_pickle(args.train_split)
        else:
            self.dataset = self._read_pickle(args.test_split)

        self.transforms = _get_mtl_transforms(train)

        # 统计数据集的 final_score 最大值和最小值
        scores = [self.data_anno.get(key)[1] / self.data_anno.get(key)[2] for key in self.dataset]
        print(f"[FineDivingDataset] {'Train' if train else 'Test'} set - "
              f"Final score max: {max(scores):.2f}, min: {min(scores):.2f}")

        # 分层抽样划分有标签/无标签（仅训练集）
        if train:
            label_list = [[key, self.data_anno.get(key)[1]] for key in self.dataset]
            self.is_labeled = stratified_label_split(
                label_list,
                labeled_ratio=args.labeled_ratio,
                n_bins=5,
                seed=args.seed_data
            )
            n_labeled = self.is_labeled.sum()
            print(f"[FineDivingDataset] Total train: {len(self.dataset)}, "
                  f"Labeled: {n_labeled} ({n_labeled / len(self.dataset) * 100:.1f}%), "
                  f"Unlabeled: {len(self.dataset) - n_labeled}")
            # 将无标签数据的分数替换为 NaN
            for i, labeled in enumerate(self.is_labeled):
                if not labeled:
                    self.data_anno[self.dataset[i]][1] = float('nan')
        else:
            self.is_labeled = np.ones(len(self.dataset), dtype=bool)

    def _read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_video(self, video_file_name):
        """
        读取帧图像，返回 PIL Image 列表（长度 = self.frame_length）。
        video_file_name: label_dict 中的 key，格式同 MTLPair.py
        """
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))
        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])
        frame_list = np.linspace(start_frame, end_frame, self.frame_length).astype(np.int_)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.frame_length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.frame_length)]
        return self.transforms(video)

    def __getitem__(self, idx):
        key = self.dataset[idx]
        video_tensor = self._load_video(key)  # (C, T, H, W)

        if self.usingDD:
            diff = self.data_anno.get(key)[2]
            final_score = self.data_anno.get(key)[1] / self.data_anno.get(key)[2]
        else:
            final_score = self.data_anno.get(key)[1]
            diff = []

        label = self.normalize_score(final_score)
        is_labeled = bool(self.is_labeled[idx])
        # 关键：返回稳定 sample_index（=数据集内 idx）
        sample_index = int(idx)
        return video_tensor, label, diff, is_labeled, sample_index

    def __len__(self):
        return len(self.dataset)

    def normalize_score(self, score):
        return score / self.score_range

# -- JIG 数据集 -------------------------------------------------------
class JIGDataset(Dataset):
    """
    单样本读取流程（不读取参考样本）。
    从 JIG 帧目录读取目标样本 *_capture1，__getitem__ 返回:
        frames_tensor : (C, 160, H, W)  待 I3D 提取特征
        label         : float, sum(raw_score) / 30
        diff          : []，JIG 不使用 DD/DN
        is_labeled    : bool
        sample_index  : int
    """

    def __init__(self, args, train=True):
        self.train = train
        self.frames_dir = args.video_path
        self.info_dir = args.label_path
        self.cls = args.jig_cls if getattr(args, 'jig_cls', None) else args.action_type
        self.fold = args.jig_fold
        self.frame_length = 160
        self.score_range = 30.0
        args.score_range = self.score_range
        args.frame_length = self.frame_length
        args.usingDD = False

        self.label_dict = self._read_pickle(os.path.join(self.info_dir, 'label.pkl'))
        self.samples = self._load_fold(train)
        self.transforms = _get_mtl_transforms(train)

        scores = [sample[1] for sample in self.samples]
        print(f"[JIGDataset] {'Train' if train else 'Test'} set - "
              f"Final score max: {max(scores):.2f}, min: {min(scores):.2f}")

        if train:
            self.is_labeled = stratified_label_split(
                self.samples,
                labeled_ratio=args.labeled_ratio,
                n_bins=5,
                seed=args.seed_data
            )
            n_labeled = self.is_labeled.sum()
            print(f"[JIGDataset] Total train: {len(self.samples)}, "
                  f"Labeled: {n_labeled} ({n_labeled / len(self.samples) * 100:.1f}%), "
                  f"Unlabeled: {len(self.samples) - n_labeled}")
            for i, labeled in enumerate(self.is_labeled):
                if not labeled:
                    self.samples[i][1] = float('nan')
        else:
            self.is_labeled = np.ones(len(self.samples), dtype=bool)

    def _read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _score(self, name):
        score = self.label_dict[name]
        if torch.is_tensor(score):
            score = score.detach().cpu().numpy()
        return float(np.asarray(score, dtype=np.float32).sum())

    def _load_fold(self, train):
        split_path = os.path.join(self.info_dir, 'splits.pkl')
        cv_file = self._read_pickle(split_path)
        if self.cls not in cv_file:
            if len(cv_file) == 1:
                self.cls = next(iter(cv_file.keys()))
                print(f"[JIGDataset] --jig-cls not found; using only split key: {self.cls}")
            else:
                raise KeyError(f"JIG class '{self.cls}' not found in {split_path}; available: {list(cv_file.keys())}")
        all_list = cv_file[self.cls]
        folds = list(range(len(all_list)))
        if self.fold < 0 or self.fold >= len(folds):
            raise ValueError(f"Invalid JIG fold {self.fold}; expected 0 <= fold < {len(folds)}")

        if train:
            folds.pop(self.fold)
        else:
            folds = [self.fold]

        samples = []
        for fold in folds:
            for vid in all_list[fold]:
                sample_name = vid + '_capture1'
                samples.append([sample_name, self._score(vid)])
        return samples

    def _load_video(self, video_file_name):
        image_list = sorted(glob.glob(os.path.join(self.frames_dir, video_file_name, '*.jpg')))
        if len(image_list) == 0:
            raise FileNotFoundError(f"No jpg frames found for JIG sample: {video_file_name}")

        frame_idx = np.linspace(0, len(image_list) - 1, num=self.frame_length, dtype=np.int_)
        video = [Image.open(image_list[i]).convert('RGB') for i in frame_idx]
        return self.transforms(video)

    def __getitem__(self, idx):
        sample_name, score = self.samples[idx]
        video_tensor = self._load_video(sample_name)
        label = self.normalize_score(score)
        is_labeled = bool(self.is_labeled[idx])
        sample_index = int(idx)
        return video_tensor, label, [], is_labeled, sample_index

    def __len__(self):
        return len(self.samples)

    def normalize_score(self, score):
        return score / self.score_range

