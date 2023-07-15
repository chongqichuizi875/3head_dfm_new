import torch
from utils import stable_log1pex
import torch.nn.functional as F


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


def es_dfm_loss(targets, outputs):
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
    # yobs = targets[:, 2].to(torch.float32)
    yobs = targets[:, 2]
    c = outputs["C"]
    pc = torch.sigmoid(c)
    pos_loss = -torch.log(pc)
    neg_loss = -torch.log(1 - pc)
    sum_loss = pos_loss * yobs + neg_loss * (1 - yobs)
    sum_loss[sum_loss.isnan()] = 0
    return torch.mean(sum_loss)


def oracle_loss(targets, outputs):
    # yobs = targets[:, 0].to(torch.float32)
    # c = outputs["C"]
    # pc = torch.sigmoid(c)
    # pos_loss = stable_log1pex(c)
    # neg_loss = -torch.log(1 - pc)
    # sum_loss = pos_loss * yobs + neg_loss * (1 - yobs)
    # sum_loss[sum_loss.isnan()] = 0
    # return torch.mean(sum_loss)

    yobs = targets[:, 0]
    o = outputs["observe"]
    c = outputs["C"]
    po = torch.sigmoid(o)
    pc = torch.sigmoid(c)
    # d = targets[:, 1]
    # lamb = F.softplus(outputs["lamb"])
    # log_lamb = torch.log(lamb)
    # e = targets[:, 3]
    pos_loss = -(torch.log(po) + torch.log(pc))
    neg_loss = -torch.log(1 - po + po * (1 - pc))

    sum_loss = pos_loss * yobs + neg_loss * (1 - yobs)
    sum_loss[sum_loss.isnan()] = 0
    return torch.mean(sum_loss)
