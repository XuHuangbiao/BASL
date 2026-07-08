import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='./RG/RG-feature')
parser.add_argument('--clip-num', type=int, default=68)

parser.add_argument('--dataset', type=str, choices=['MTL-AQA', 'FineDiving', 'FineFS', 'RG', 'JIG'], default='RG')
parser.add_argument('--train-label-path', type=str, default='./GDLT_data/train.txt')
parser.add_argument('--test-label-path', type=str, default='./GDLT_data/test.txt')

parser.add_argument('--action-type', type=str, default='TES')
parser.add_argument('--score-type', type=str, default='Total_Score')

parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--seed_data', type=int, default=0)

parser.add_argument('--optim', type=str, default='adam')

parser.add_argument("--lr-decay", type=str, default=None, help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_encoder', type=int, default=3)

parser.add_argument('--eps', type=float, default=0.05)
parser.add_argument('--mask_ratio', type=float, default=0.15)

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--score_range', type=int, default=None)
parser.add_argument('--labeled_ratio', type=float, default=0.4)

# ── MTL-AQA/FineDiving 专用参数 ──────────────────────────────────────────────
parser.add_argument('--pretrained-i3d-weight', type=str,
                    default='./models/model_rgb.pth',
                    help='预训练 I3D 权重路径（MTL-AQA 模式使用）')
parser.add_argument('--label-path', type=str,
                    default='./MTL-AQA/info/final_annotations_dict_with_dive_number.pkl',
                    help='MTL-AQA label pickle 文件路径')
parser.add_argument('--train-split', type=str,
                    default='./MTL-AQA/info/train_split_0.pkl',
                    help='MTL-AQA 训练集 split pickle 路径')
parser.add_argument('--test-split', type=str,
                    default='./MTL-AQA/info/test_split_0.pkl',
                    help='MTL-AQA 测试集 split pickle 路径')
parser.add_argument('--frame-length', type=int, default=103,
                    help='MTL-AQA 每个视频采样帧数')
parser.add_argument('--temporal-shift-min', type=int, default=-5,
                    help='MTL-AQA 时序增强最小偏移')
parser.add_argument('--temporal-shift-max', type=int, default=5,
                    help='MTL-AQA 时序增强最大偏移')
parser.add_argument('--usingDD', action='store_true', default=False,
                    help='MTL-AQA: 是否使用 Difficulty Degree')
parser.add_argument('--jig-cls', type=str, default=None,
                    help='JIG 类别名；默认使用 --action-type')
parser.add_argument('--jig-fold', type=int, default=3,
                    help='JIG cross-validation fold index')

# ═══════════════════════════════════════════
# 新增参数：三阶段控制
# ═══════════════════════════════════════════
parser.add_argument('--stageA_epochs', type=int, default=10, help='阶段A epoch数（仅标签回归+全量一致性）')
parser.add_argument('--stageB_epochs', type=int, default=50, help='阶段B epoch数（主动选择+伪标签）')
parser.add_argument('--stageC_epochs', type=int, default=100, help='阶段C epoch数（逐步扩大百分比）')

# 新增参数：主动选择与动量过滤
parser.add_argument('--boundary_percent', type=float, default=0.1, help='初始 top/bottom 百分比 (e.g., 0.1 for 10%)')
parser.add_argument('--boundary_std_thresh', type=float, default=0.05, help='方差上界阈值（选择一致性要求）')
parser.add_argument('--momentum_window', type=int, default=3, help='动量过滤窗口：连续N个epoch都被选才确认')

# 新增参数：阶段C扩大策略
parser.add_argument('--expand_interval', type=int, default=30, help='阶段C多少个epoch扩大一次百分比')
parser.add_argument('--expand_step', type=float, default=0.05, help='阶段C百分比扩大步长 (e.g., 0.05 from 10% to 15%)')

# 新增参数：一致性损失权重
parser.add_argument('--lambda_cons', type=float, default=1.0, help='一致性损失权重')
parser.add_argument('--lambda_reg', type=float, default=1.0, help='伪标签回归损失权重')