import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def test_epoch(epoch, model, test_loader, logger, device, args, stage_prefix=""):
    """
    测试函数
    """
    mse_loss = nn.MSELoss().to(device)
    model.eval()
    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if len(batch) == 5:
                video_feat, label, diff, _, sample_index = batch
            else:
                video_feat, label, diff, _ = batch
                sample_index = None

            video_feat = video_feat.to(device)
            label = label.float().to(device)

            if args.usingDD:
                diff = diff.float().to(device)
            else:
                diff = None

            outputs = model(video_feat, mode='test', return_aug=False)
            pred = outputs['output']

            # DD 仅用于指标缩放
            if args.usingDD:
                if args.dataset == 'MTL-AQA':
                    max_score = 104.5
                elif args.dataset == 'FineDiving':
                    max_score = 115
                else:
                    max_score = args.score_range
                pred = pred * diff * args.score_range / max_score
                label = label * diff * args.score_range / max_score

            loss = mse_loss(pred, label)
            tol_loss += loss.item() * label.shape[0]
            tol_sample += label.shape[0]

            p = pred.detach().cpu().numpy()
            l = label.detach().cpu().numpy()
            preds = np.concatenate((preds, p)) if len(preds) else p
            labels = np.concatenate((labels, l)) if len(labels) else l
    
    avg_coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)
    RL2 = 100 * np.power((preds - labels) / (labels.max() - labels.min()), 2).sum() / labels.shape[0]
    if logger is not None:
        logger.add_scalar(f'{stage_prefix}test_coef', avg_coef, epoch)
        logger.add_scalar(f'{stage_prefix}test_loss', avg_loss, epoch)
        logger.add_scalar(f'{stage_prefix}test_RL2', RL2, epoch)

    print(f"[{stage_prefix}Test Epoch {epoch}] Test Loss: {avg_loss:.4f}, Spearman ρ: {avg_coef:.4f}, RL2: {RL2:.4f}")
    return avg_loss, avg_coef, RL2