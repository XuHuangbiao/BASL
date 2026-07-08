import numpy as np


class ActiveSelector:
    """
    主动选择器：基于百分比 + 方差筛选边界样本

    改进：不再做额外的增强推理，而是使用训练过程中记录的预测分数
    """

    def __init__(self, args):
        self.args = args
        self.boundary_percent = args.boundary_percent
        self.boundary_std_thresh = args.boundary_std_thresh

    def select(self, unlabeled_predictions, current_boundary_percent=None):
        """
        进行主动选择（从记录的预测中）
        
        Args:
            unlabeled_predictions: dict {sample_index: (score_original, score_weak, score_strong)}
                                   这些来自训练epoch的记录
            current_boundary_percent: 当前百分比（用于阶段C动态调整）

        Returns:
            boundary_samples: dict {sample_index: pseudo_label}
        """
        if current_boundary_percent is None:
            current_boundary_percent = self.boundary_percent

        if len(unlabeled_predictions) == 0:
            return {}

        boundary_samples = {}

        # 计算每个样本的均值和方差
        all_sample_indices = []
        all_mean_scores = []
        all_std_scores = []
        all_pseudo_labels = []

        for sid, (score_orig, score_weak, score_strong) in unlabeled_predictions.items():
            # 使用三个预测的均值与方差
            scores = np.array([score_orig, score_weak, score_strong])
            mean_score = scores.mean()
            std_score = scores.std()

            all_sample_indices.append(sid)
            all_mean_scores.append(mean_score)
            all_std_scores.append(std_score)
            all_pseudo_labels.append(score_orig)  # 伪标签使用原始预测

        all_mean_scores = np.array(all_mean_scores)
        all_std_scores = np.array(all_std_scores)
        all_pseudo_labels = np.array(all_pseudo_labels)
        all_sample_indices = np.array(all_sample_indices)

        # 百分比选择：top 和 bottom
        sorted_indices = np.argsort(all_mean_scores)
        n_boundary = max(1, int(len(all_mean_scores) * current_boundary_percent))

        top_indices = sorted_indices[-n_boundary:]  # 高分 top
        bottom_indices = sorted_indices[:n_boundary]  # 低分 bottom

        candidate_indices = np.concatenate([top_indices, bottom_indices])

        # 在候选中筛选方差低（一致性高）的样本
        for idx in candidate_indices:
            std = all_std_scores[idx]
            if std <= self.boundary_std_thresh:
                sid = int(all_sample_indices[idx])
                pseudo_label = float(all_pseudo_labels[idx])
                boundary_samples[sid] = pseudo_label

        print(f"[Active Selector] 候选样本数: {len(candidate_indices)}, "
              f"过滤后边界样本数: {len(boundary_samples)}")

        return boundary_samples