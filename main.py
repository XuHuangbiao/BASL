import os
import torch
import torch.optim as optim_module
import numpy as np
from datasets import RGDataset, FineFSDataset, MTLAQADataset, FineDivingDataset, JIGDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model import BASL
from models.loss import LossFun
from test import test_epoch
from active_selector import ActiveSelector
from momentum_filter import MomentumFilter
from train import train_stageA_epoch, train_stageB_epoch, train_stageC_epoch
import options


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_optim(model, args):
    trainable_params = model.get_trainable_parameters()
    lr = args.lr

    if args.optim == 'sgd':
        optim = optim_module.SGD(trainable_params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = optim_module.Adam(trainable_params, lr=lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = optim_module.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = optim_module.RMSprop(trainable_params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")

    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = optim_module.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = optim_module.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


def is_better_metric(coef, rl2, best_coef, best_rl2, eps=1e-12):
    return coef > best_coef + eps or (abs(coef - best_coef) <= eps and rl2 < best_rl2)


def save_stage_checkpoint(model, args, stage, epoch, coef, rl2):
    ckpt_path = f'./ckpt//best_{stage}_{args.model_name}.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f"✓ Stage best model saved at epoch {epoch} ({stage}) | Spearman: {coef:.4f}, RL2: {rl2:.4f}")


def save_global_checkpoint(model, args, stage, epoch, coef, rl2):
    ckpt_path = f'./ckpt//best_global_{args.model_name}.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f"✓ Global best model saved at epoch {epoch} ({stage}) | Spearman: {coef:.4f}, RL2: {rl2:.4f}")


if __name__ == '__main__':
    args = options.parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'RG':
        train_data = RGDataset(args.video_path, args.train_label_path,
                               clip_num=args.clip_num, action_type=args.action_type, args=args)
        test_data = RGDataset(args.video_path, args.test_label_path,
                              clip_num=args.clip_num, action_type=args.action_type, train=False, args=args)
    elif args.dataset == 'FineFS':
        train_data = FineFSDataset(args.video_path, args.train_label_path,
                                   clip_num=args.clip_num, action_type=args.action_type, args=args)
        test_data = FineFSDataset(args.video_path, args.test_label_path,
                                  clip_num=args.clip_num, action_type=args.action_type, train=False, args=args)
    elif args.dataset == 'MTL-AQA':
        train_data = MTLAQADataset(args, train=True)
        test_data = MTLAQADataset(args, train=False)
    elif args.dataset == 'FineDiving':
        train_data = FineDivingDataset(args, train=True)
        test_data = FineDivingDataset(args, train=False)
    elif args.dataset == 'JIG':
        train_data = JIGDataset(args, train=True)
        test_data = JIGDataset(args, train=False)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=False)

    print(f'Train size: {len(train_data)}, Test size: {len(test_data)}')
    print('=============Load dataset successfully=============')

    net = BASL(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        n_encoder=args.n_encoder,
        dropout=args.dropout,
        config=args
    ).to(device)

    loss_fn = LossFun(args).to(device)

    os.makedirs("./ckpt/", exist_ok=True)
    os.makedirs("./logs/" + args.model_name, exist_ok=True)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))

    print(f'Model: {net.__class__.__name__}')
    print(f'Optimizer: {args.optim}, Learning rate: {args.lr}')
    print('=============Model created successfully=============')

    if args.test:
        if args.ckpt is not None:
            checkpoint = torch.load(args.ckpt)
            net.load_state_dict(checkpoint)
        test_loss, coef, RL2 = test_epoch(0, net, test_loader, None, device, args, stage_prefix="")
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}\tTest RL2: {:.3f}'.format(test_loss, coef, RL2))
        raise SystemExit

    print("\n" + "=" * 60)
    print("阶段A：仅标签回归 + 全量一致性（含无标签）")
    print("=" * 60)

    net.train()
    optimizer = get_optim(net, args)
    scheduler = get_scheduler(optimizer, args)

    global_best_coef = -float('inf')
    global_best_rl2 = float('inf')
    global_best_stage = None

    best_coef_a = -float('inf')
    best_rl2_a = float('inf')

    if args.warmup:
        warmup_stage1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: t / args.warmup
        )
    else:
        warmup_stage1 = None

    for epoch in range(args.stageA_epochs):
        if warmup_stage1 is not None and epoch < args.warmup:
            warmup_stage1.step()

        _ = train_stageA_epoch(epoch, net, loss_fn, train_loader, optimizer, logger, device, args)
        loss_a, coef_a, RL2_a = test_epoch(epoch, net, test_loader, logger, device, args, stage_prefix='stageA_')

        if scheduler is not None:
            scheduler.step()

        if is_better_metric(coef_a, RL2_a, best_coef_a, best_rl2_a):
            best_coef_a = coef_a
            best_rl2_a = RL2_a
            save_stage_checkpoint(net, args, 'stageA', epoch, coef_a, RL2_a)

        if is_better_metric(coef_a, RL2_a, global_best_coef, global_best_rl2):
            global_best_coef = coef_a
            global_best_rl2 = RL2_a
            global_best_stage = 'stageA'
            save_global_checkpoint(net, args, global_best_stage, epoch, coef_a, RL2_a)
        # if epoch in [18]:
        #     ckpt_path = f'./ckpt//{args.model_name}_other_{str(epoch)}.pth'
        #     torch.save(net.state_dict(), ckpt_path)
        #     print(f"Model Saved!!!")

    print(f"\n✓ 阶段A完成！Best Spearman: {best_coef_a:.4f}, RL2 at Best Spearman: {best_rl2_a:.4f}")

    print("\n" + "=" * 60)
    print("阶段B：主动选择 + 动量过滤 + 伪标签回归")
    print("=" * 60)
    
    # 加载当前全局最佳模型权重进入阶段B
    best_global_ckpt = f'./ckpt//best_global_{args.model_name}.pth'
    if os.path.exists(best_global_ckpt):
        net.load_state_dict(torch.load(best_global_ckpt, weights_only=True))
        print(f"✓ 阶段B加载当前全局最佳模型权重: {best_global_ckpt}")
    else:
        print(f"✗ 全局最佳模型权重未找到，继续使用当前模型参数")

    active_selector = ActiveSelector(args)
    momentum_filter = MomentumFilter(args)

    # 新增：构造“全局无标签 idx”作为动量过滤的连续统计全集
    if hasattr(train_data, "is_labeled"):
        all_unlabeled_indices = [int(i) for i, flag in enumerate(train_data.is_labeled) if not bool(flag)]
    else:
        # 兜底：如果某个 dataset 未暴露 is_labeled，就退化为全体 idx
        all_unlabeled_indices = list(range(len(train_data)))

    best_coef_b = -float('inf')
    best_rl2_b = float('inf')
    current_boundary_percent = args.boundary_percent
    confirmed_boundaries = {}

    for epoch in range(args.stageB_epochs):
        unlabeled_predictions = train_stageB_epoch(
            epoch, net, loss_fn, train_loader, optimizer, logger, device, args,
            boundary_samples=None if epoch == 0 else confirmed_boundaries
        )

        print(f"[ActiveSelector] Epoch {epoch}, boundary_percent={current_boundary_percent:.2%}")
        boundary_samples = active_selector.select(unlabeled_predictions, current_boundary_percent)

        momentum_filter.update(boundary_samples, all_sample_indices=all_unlabeled_indices)

        confirmed_boundaries = momentum_filter.get_confirmed_boundaries(boundary_samples)
        print(f"[MomentumFilter] 确认的边界样本数: {len(confirmed_boundaries)}")

        loss_b, coef_b, RL2_b = test_epoch(epoch, net, test_loader, logger, device, args, stage_prefix='stageB_')

        if scheduler is not None:
            scheduler.step()

        if is_better_metric(coef_b, RL2_b, best_coef_b, best_rl2_b):
            best_coef_b = coef_b
            best_rl2_b = RL2_b
            save_stage_checkpoint(net, args, 'stageB', epoch, coef_b, RL2_b)

        if is_better_metric(coef_b, RL2_b, global_best_coef, global_best_rl2):
            global_best_coef = coef_b
            global_best_rl2 = RL2_b
            global_best_stage = 'stageB'
            save_global_checkpoint(net, args, global_best_stage, epoch, coef_b, RL2_b)

    print(f"\n✓ 阶段B完成！Best Spearman: {best_coef_b:.4f}, RL2 at Best Spearman: {best_rl2_b:.4f}")

    print("\n" + "=" * 60)
    print("阶段C：逐步扩大伪标签范围")
    print("=" * 60)

    # 加载当前全局最佳模型权重进入阶段C
    best_global_ckpt = f'./ckpt//best_global_{args.model_name}.pth'
    if os.path.exists(best_global_ckpt):
        net.load_state_dict(torch.load(best_global_ckpt, weights_only=True))
        print(f"✓ 阶段C加载当前全局最佳模型权重: {best_global_ckpt}")
    else:
        print(f"✗ 全局最佳模型权重未找到，继续使用当前模型参数")
        
    momentum_filter.reset()

    best_coef_c = -float('inf')
    best_rl2_c = float('inf')
    current_boundary_percent = args.boundary_percent
    confirmed_boundaries = {}

    for epoch in range(args.stageC_epochs):
        if epoch % args.expand_interval == 0:
            current_boundary_percent = min(current_boundary_percent + args.expand_step, 0.5)
            print(f"\n[Expansion] 百分比扩大到 {current_boundary_percent:.2%}")

        unlabeled_predictions = train_stageC_epoch(
            epoch, net, loss_fn, train_loader, optimizer, logger, device, args,
            boundary_samples=confirmed_boundaries
        )

        print(f"[ActiveSelector] Epoch {epoch}, boundary_percent={current_boundary_percent:.2%}")
        boundary_samples = active_selector.select(unlabeled_predictions, current_boundary_percent)

        momentum_filter.update(boundary_samples, all_sample_indices=all_unlabeled_indices)

        confirmed_boundaries = momentum_filter.get_confirmed_boundaries(boundary_samples)
        print(f"[MomentumFilter] 确认的边界样本数: {len(confirmed_boundaries)}")

        loss_c, coef_c, RL2_c = test_epoch(epoch, net, test_loader, logger, device, args, stage_prefix='stageC_')

        if scheduler is not None:
            scheduler.step()

        if is_better_metric(coef_c, RL2_c, best_coef_c, best_rl2_c):
            best_coef_c = coef_c
            best_rl2_c = RL2_c
            save_stage_checkpoint(net, args, 'stageC', epoch, coef_c, RL2_c)

        if is_better_metric(coef_c, RL2_c, global_best_coef, global_best_rl2):
            global_best_coef = coef_c
            global_best_rl2 = RL2_c
            global_best_stage = 'stageC'
            save_global_checkpoint(net, args, global_best_stage, epoch, coef_c, RL2_c)

    print(f"\n✓ 阶段C完成！Best Spearman: {best_coef_c:.4f}, RL2 at Best Spearman: {best_rl2_c:.4f}")

    print("\n" + "=" * 60)
    print("训练完成总结")
    print("=" * 60)
    print(f"阶段A 最佳 Spearman: {best_coef_a:.4f}, 对应 RL2: {best_rl2_a:.4f}")
    print(f"阶段B 最佳 Spearman: {best_coef_b:.4f}, 对应 RL2: {best_rl2_b:.4f}")
    print(f"阶段C 最佳 Spearman: {best_coef_c:.4f}, 对应 RL2: {best_rl2_c:.4f}")
    print(f"全局最佳 Spearman: {global_best_coef:.4f}, 对应 RL2: {global_best_rl2:.4f}, 来源: {global_best_stage}")
    print("=" * 60)
    print(f"{best_coef_a:.3f}, {best_rl2_a:.3f}")
    print(f"{best_coef_b:.3f}, {best_rl2_b:.3f}")
    print(f"{best_coef_c:.3f}, {best_rl2_c:.3f}")

    logger.close()