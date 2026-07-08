import numpy as np
import torch
from scipy.stats import spearmanr
from utils import AverageMeter


def train_stageA_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    """
    阶段A训练：仅有标签回归 + 全量一致性（强弱增强）
    不引入伪标签
    返回：unlabeled_predictions
    """
    model.train()

    preds = np.array([])
    labels_all = np.array([])
    losses = AverageMeter('loss', logger)

    unlabeled_predictions = {}  # {sample_index: (score_original, score_weak, score_strong)}

    for i, batch_data in enumerate(train_loader):
        if len(batch_data) != 5:
            raise RuntimeError(f"Expected 5 elements, got {len(batch_data)}")

        video_feat, label, diff, is_labeled, sample_index = batch_data

        B = video_feat.shape[0]
        video_feat = video_feat.to(device)
        label = label.float().to(device)
        is_labeled = is_labeled.bool().to(device)
        sample_index_cpu = sample_index.detach().cpu().numpy().astype(int)

        if args.usingDD:
            diff = diff.float().to(device)
        else:
            diff = None

        if is_labeled.sum().item() == 0:
            continue

        outputs = model(video_feat, mode='train', return_aug=True)

        # 记录无标签样本预测（只对无标签）
        score_original = outputs["score_original"].detach().cpu().numpy()
        score_weak = outputs['score_weak'].detach().cpu().numpy()
        score_strong = outputs['score_strong'].detach().cpu().numpy()

        unlabeled_mask_cpu = (~is_labeled).detach().cpu().numpy()
        for j, (sid, orig, weak, strong) in enumerate(zip(sample_index_cpu, score_original, score_weak, score_strong)):
            if unlabeled_mask_cpu[j]:
                unlabeled_predictions[int(sid)] = (float(orig), float(weak), float(strong))

        # 损失：监督只用 is_labeled；一致性用全量
        loss, loss_dict = loss_fn(outputs, label, boundary_info=None, labeled_mask=is_labeled)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss_dict['total'], B)

        # 记录预测（仅有标签样本）
        pred = outputs["output"]
        if is_labeled.sum() > 0:
            pred_labeled = pred[is_labeled].detach().cpu().numpy()
            label_labeled = label[is_labeled].detach().cpu().numpy()

            # 确保训练时的分数缩放逻辑与测试时一致
            if args.usingDD:
                diff_labeled = diff[is_labeled].detach().cpu().numpy()
                if args.dataset == 'MTL-AQA':
                    max_score = 104.5
                elif args.dataset == 'FineDiving':
                    max_score = 115
                else:
                    max_score = args.score_range if args.score_range else 100

                pred_labeled = pred_labeled * diff_labeled * args.score_range / max_score
                label_labeled = label_labeled * diff_labeled * args.score_range / max_score

            preds = np.concatenate([preds, pred_labeled]) if len(preds) > 0 else pred_labeled
            labels_all = np.concatenate([labels_all, label_labeled]) if len(labels_all) > 0 else label_labeled

    coef = spearmanr(preds, labels_all)[0] if len(preds) > 1 else 0.0

    # 若一个 epoch 恰好全被跳过（极端情况），AverageMeter.count 可能为 0，避免除零
    if losses.count == 0:
        avg_loss = 0.0
    else:
        avg_loss = losses.done(epoch)

    if logger is not None:
        logger.add_scalar('stageA_train_loss', avg_loss, epoch)
        logger.add_scalar('stageA_train_spearman', coef, epoch)

    print(f"[StageA Epoch {epoch}] Train Loss: {avg_loss:.4f}, Spearman ρ: {coef:.4f}")
    return unlabeled_predictions


def train_stageB_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args, boundary_samples=None):
    """
    阶段B训练：有标签回归 + 一致性 + 边界伪标签回归
    返回：unlabeled_predictions
    """
    model.train()

    preds = np.array([])
    labels_all = np.array([])
    losses = AverageMeter('loss', logger)

    unlabeled_predictions = {}
    boundary_samples = boundary_samples or {}
    boundary_set = set(int(x) for x in boundary_samples.keys())

    epoch_boundary_used = set()

    for i, batch_data in enumerate(train_loader):
        if len(batch_data) != 5:
            raise RuntimeError(f"Expected 5 elements, got {len(batch_data)}")

        video_feat, label, diff, is_labeled, sample_index = batch_data

        B = video_feat.shape[0]
        video_feat = video_feat.to(device)
        label = label.float().to(device)
        is_labeled = is_labeled.bool().to(device)
        sample_index_cpu = sample_index.detach().cpu().numpy().astype(int)

        if args.usingDD:
            diff = diff.float().to(device)
        else:
            diff = None

        boundary_mask = torch.zeros(B, dtype=torch.bool, device=device)
        pseudo_labels = torch.zeros(B, dtype=torch.float, device=device)

        for j, sid in enumerate(sample_index_cpu):
            sid_int = int(sid)
            if (sid_int in boundary_set) and (not bool(is_labeled[j].item())):
                boundary_mask[j] = True
                pseudo_labels[j] = float(boundary_samples[sid_int])
                epoch_boundary_used.add(sid_int)

        boundary_info = {'mask': boundary_mask, 'pseudo_labels': pseudo_labels} if boundary_mask.any() else None

        outputs = model(video_feat, mode='train', return_aug=True)

        score_original = outputs["score_original"].detach().cpu().numpy()
        score_weak = outputs['score_weak'].detach().cpu().numpy()
        score_strong = outputs['score_strong'].detach().cpu().numpy()

        unlabeled_mask_cpu = (~is_labeled).detach().cpu().numpy()
        for j, (sid, orig, weak, strong) in enumerate(zip(sample_index_cpu, score_original, score_weak, score_strong)):
            if unlabeled_mask_cpu[j]:
                unlabeled_predictions[int(sid)] = (float(orig), float(weak), float(strong))

        loss, loss_dict = loss_fn(outputs, label, boundary_info=boundary_info, labeled_mask=is_labeled)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss_dict['total'], B)

        pred = outputs["output"]
        if is_labeled.sum() > 0:
            pred_labeled = pred[is_labeled].detach().cpu().numpy()
            label_labeled = label[is_labeled].detach().cpu().numpy()

            # 确保训练时的分数缩放逻辑与测试时一致
            if args.usingDD:
                diff_labeled = diff[is_labeled].detach().cpu().numpy()
                if args.dataset == 'MTL-AQA':
                    max_score = 104.5
                elif args.dataset == 'FineDiving':
                    max_score = 115
                else:
                    max_score = args.score_range if args.score_range else 100

                pred_labeled = pred_labeled * diff_labeled * args.score_range / max_score
                label_labeled = label_labeled * diff_labeled * args.score_range / max_score

            preds = np.concatenate([preds, pred_labeled]) if len(preds) > 0 else pred_labeled
            labels_all = np.concatenate([labels_all, label_labeled]) if len(labels_all) > 0 else label_labeled
    
    coef = spearmanr(preds, labels_all)[0] if len(preds) > 1 else 0.0
    avg_loss = losses.done(epoch)

    if logger is not None:
        logger.add_scalar('stageB_train_loss', avg_loss, epoch)
        logger.add_scalar('stageB_train_spearman', coef, epoch)

    print(f"\n[StageB Epoch {epoch}] Train Loss: {avg_loss:.4f}, Spearman ρ: {coef:.4f}, "
          f"Boundary samples used (unique): {len(epoch_boundary_used)}")

    return unlabeled_predictions


def train_stageC_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args, boundary_samples=None):
    """
    阶段C训练：类似阶段B，但百分比逐步扩大
    """
    return train_stageB_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args, boundary_samples)