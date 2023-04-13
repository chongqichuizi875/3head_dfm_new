import torch.nn
import torch.nn.functional as F

from utils import stable_log1pex


def losses_for_mf_based_fits(name):
    if name == "MF":
        return mf_loss
    if name == "MF_IPS":
        return mf_ips_loss
    if name == "MF_SNIPS":
        return mf_snips_loss


def mf_loss(pred, sub_y, one_over_zl, selected_idx):
    return torch.nn.BCELoss()(sub_y, pred)


def mf_ips_loss(pred, sub_y, one_over_zl, selected_idx):
    inv_prop = one_over_zl[selected_idx]
    return F.binary_cross_entropy(sub_y, pred, weight=inv_prop)


def mf_snips_loss(pred, sub_y, one_over_zl, selected_idx):
    inv_prop = one_over_zl[selected_idx]
    return F.binary_cross_entropy(sub_y, pred, weight=inv_prop, reduction="sum")


def three_head_loss(targets, outputs):
    yobs = targets[:, 2]
    o = outputs["observe"]
    c = outputs["C"]
    d = targets[:, 1]
    lamb = F.softplus(outputs["lamb"])
    log_lamb = torch.log(lamb)
    e = targets[:, 3]
    po = torch.sigmoid(o)
    pc = torch.sigmoid(c)
    pos_loss = -(-stable_log1pex(o) - stable_log1pex(c) + log_lamb - lamb * d)
    neg_loss = -torch.log(1 - po + po * (1 - pc) + po * pc * torch.exp(-lamb * e))
    sum_loss = pos_loss * yobs * 1.5 + neg_loss * (1 - yobs)
    sum_loss[sum_loss.isnan()] = 0
    return torch.mean(sum_loss)


def exploss(targets, outputs):
    yobs = targets[:, 2]
    # o = outputs["observe"]
    c = outputs["C"]
    d = targets[:, 1]
    lamb = F.softplus(outputs["lamb"])
    log_lamb = torch.log(lamb)
    e = targets[:, 3]
    # po = torch.sigmoid(o)
    pc = torch.sigmoid(c)
    pos_loss = -(-stable_log1pex(c) + log_lamb - lamb * d)
    neg_loss = -torch.log((1 - pc) + pc * torch.exp(-lamb * e))
    sum_loss = pos_loss * yobs + neg_loss * (1 - yobs)
    sum_loss[sum_loss.isnan()] = 0
    return torch.mean(sum_loss)


def default_bce_loss(targets, outputs):
    yobs = targets[:, 2].to(torch.float32)
    c = outputs["C"]
    pc = torch.sigmoid(c)
    pos_loss = stable_log1pex(c)
    neg_loss = -(1 - pc) * torch.log(1 - pc)
    sum_loss = pos_loss * yobs + neg_loss * (1 - yobs)
    sum_loss[sum_loss.isnan()] = 0
    return torch.mean(sum_loss)
