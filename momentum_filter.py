from collections import defaultdict, deque


class MomentumFilter:
    """
    动态动量过滤器：记录样本连续被选为边界的次数

    目标（严格版）：
    - 对每个样本维护长度为 window_size 的布尔队列
    - 每个 epoch 都要对其 append(True/False)
    - 只有最近 window_size 个 epoch 全为 True 才确认

    注意：
    - 为了实现“连续”，update() 需要知道“本 epoch 参与统计的样本全集”。
      你可以从训练集构造一个 all_sample_indices=list(range(len(train_data)))
      然后每个 epoch 调用 update(boundary_samples, all_sample_indices)。
    """

    def __init__(self, args):
        self.window_size = int(args.momentum_window)

        # key: sample_index, value: deque[bool] (True=本epoch被选, False=未被选)
        self.history = defaultdict(lambda: deque(maxlen=self.window_size))

        # 存储当前确认的伪标签（最近一次确认值）
        self.confirmed_pseudo_labels = {}

    def update(self, boundary_samples, all_sample_indices=None):
        """
        每个 epoch 调用一次，更新“连续窗口”历史。

        Args:
            boundary_samples: dict {sample_index: pseudo_label}
            all_sample_indices: iterable[int]，本epoch要被计入连续统计的样本全集
                               （推荐传入训练集所有无标签样本索引，或训练集全体索引）

        Returns:
            boundary_set: set[int]
        """
        boundary_set = set(int(idx) for idx in boundary_samples.keys())

        if all_sample_indices is None:
            # 退化策略：
            # - 对本epoch被选中的样本 append(True)
            # - 对历史中存在但本epoch未被选中的样本 append(False)
            # 这样至少能保证“连续”对出现过的样本成立，但对“从未出现过历史”的样本无法补齐False。
            existing_indices = list(self.history.keys())
            for idx in existing_indices:
                self.history[int(idx)].append(int(idx) in boundary_set)

            for idx in boundary_set:
                # 确保新出现的 idx 也被记录（本epoch为True）
                self.history[int(idx)].append(True)

            return boundary_set

        # 严格策略：对全集每个样本都 append(True/False)
        for idx in all_sample_indices:
            idx = int(idx)
            self.history[idx].append(idx in boundary_set)

        return boundary_set

    def get_confirmed_boundaries(self, boundary_samples):
        """
        返回最近 window_size 个 epoch 都被选中的样本（正式确认）。

        Args:
            boundary_samples: dict {sample_index: pseudo_label}
                             （通常为本epoch active selector 产物，用于提供伪标签值）

        Returns:
            confirmed_boundaries: dict {sample_index: pseudo_label}
        """
        confirmed_boundaries = {}

        for idx, label in boundary_samples.items():
            idx = int(idx)

            h = self.history.get(idx, None)
            if h is None:
                continue

            if len(h) >= self.window_size and all(h):
                confirmed_boundaries[idx] = float(label)
                self.confirmed_pseudo_labels[idx] = float(label)

        return confirmed_boundaries

    def reset(self):
        """重置过滤器（阶段切换时使用）"""
        self.history.clear()
        self.confirmed_pseudo_labels.clear()