import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFun(nn.Module):
    """
    三阶段统一损失函数

    阶段A：仅监督损失 + 全量一致性（KL + MSE）
    阶段B/C：监督损失 + 一致性 + 边界伪标签回归
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lambda_cons = args.lambda_cons
        self.lambda_reg = args.lambda_reg

        # 基础回归损失
        self.regression_loss = nn.MSELoss()

    def forward(self, outputs, targets, boundary_info=None, labeled_mask: torch.Tensor = None):
        """
        Args:
            outputs: dict
              - 'output': (B,) 主预测分数（建议为原始视图）
              - 'score_weak': (B,) 弱增强分数（可选）
              - 'score_strong': (B,) 强增强分数（可选）
              - 'feat_weak': (B, D) 弱增强投影特征（可选）
              - 'feat_strong': (B, D) 强增强投影特征（可选）
            targets: (B,) 真实标签（注意：训练集无标签样本也可能携带真实分数，但不能用于监督）
            boundary_info: dict（阶段B/C使用）
              - 'mask': (B,) bool tensor，True 表示边界样本
              - 'pseudo_labels': (B,) 伪标签（无效位置为 0）
            labeled_mask: (B,) bool tensor，True 表示该样本允许使用真实标签做监督损失

        Returns:
            loss: scalar
            loss_dict: dict 记录各项损失
        """
        pred_scores = outputs['output']
        device = pred_scores.device

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # ═══════════════════════════════════════════
        # [1] 有标签监督损失（必须显式用 labeled_mask，避免无标签泄露）
        # ═══════════════════════════════════════════
        if targets is not None and labeled_mask is not None:
            if labeled_mask.dtype != torch.bool:
                labeled_mask = labeled_mask.bool()
            if labeled_mask.sum() > 0:
                l_sup = self.regression_loss(pred_scores[labeled_mask], targets[labeled_mask])
                total_loss += l_sup
                loss_dict['l_sup'] = l_sup.item()
            else:
                loss_dict['l_sup'] = 0.0
        else:
            loss_dict['l_sup'] = 0.0

        # ═══════════════════════════════════════════
        # [2] 一致性损失：KL散度 + 分数MSE（全量样本）
        # ═══════════════════════════════════════════
        if 'score_weak' in outputs and 'score_strong' in outputs:
            score_cons = self.regression_loss(outputs['score_strong'], outputs['score_weak'].detach())

            feat_cons = 0.0
            if 'feat_weak' in outputs and 'feat_strong' in outputs:
                feat_w = outputs['feat_weak']
                feat_s = outputs['feat_strong']

                feat_w_norm = F.normalize(feat_w, dim=-1)
                feat_s_norm = F.normalize(feat_s, dim=-1)

                # 改进：KL散度参数 input 必须是 log_prob (强增强预测)，target 是 prob (弱增强真值) 并截断梯度
                kl_loss = F.kl_div(
                    F.log_softmax(feat_s_norm, dim=-1),
                    F.softmax(feat_w_norm.detach(), dim=-1),
                    reduction='batchmean'
                )
                feat_cons = kl_loss

            l_cons = score_cons + 0.1 * feat_cons
            total_loss += self.lambda_cons * l_cons
            loss_dict['l_cons'] = float(l_cons.detach().item())
        else:
            loss_dict['l_cons'] = 0.0

        # ═══════════════════════════════════════════
        # [3] 边界样本伪标签回归损失（阶段B/C）
        # ═══════════════════════════════════════════
        if boundary_info is not None:
            boundary_mask = boundary_info['mask']
            pseudo_labels = boundary_info['pseudo_labels']

            if boundary_mask.sum() > 0:
                l_reg = self.regression_loss(pred_scores[boundary_mask], pseudo_labels[boundary_mask])
                total_loss += self.lambda_reg * l_reg
                loss_dict['l_reg'] = l_reg.item()
                loss_dict['n_boundary'] = int(boundary_mask.sum().item())
            else:
                loss_dict['l_reg'] = 0.0
                loss_dict['n_boundary'] = 0
        else:
            loss_dict['l_reg'] = 0.0
            loss_dict['n_boundary'] = 0

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict