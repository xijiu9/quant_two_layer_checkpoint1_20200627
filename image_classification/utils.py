import os
import numpy as np
import torch
import shutil
import torch.distributed as dist


def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints # and (epoch < 10 or epoch % 10 == 0)
    return _sbc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='./', backup_filename=None):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def dict_add(x, y):
    if x is None:
        return y
    return {k: x[k] + y[k] for k in x}


def dict_minus(x, y):
    return {k: x[k] - y[k] for k in x}


def dict_sqr(x):
    return {k: x[k]**2 for k in x}


def dict_sqrt(x):
    return {k: torch.sqrt(x[k]) for k in x}


def dict_mul(x, a):
    return {k: x[k]*a for k in x}


def dict_clone(x):
    return {k: x[k].clone() for k in x}

def twolayer_mm(m1, m2):
    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)
    torch.set_printoptions(profile="full")
    normalized_norm = torch.clamp(m1.shape[0] * vec_norm / (2 * vec_norm.sum()), 0, 1)
    mid = torch.sort(normalized_norm)
    mid = mid[0][int(m2.shape[0] / 2)]
    index = [x for x in range(len(normalized_norm)) if normalized_norm[x] >= mid]
    # index = [x for x in range(len(normalized_norm))]
    m1, m2 = m1[index, :], m2[index, :]
    output = m1.t().mm(m2)
    return output

def twolayer_convsample(m1, m2):
    # print(m1.mean(), m1.max(), m1.min(), m2.mean(), m2.max(), m2.min())
    m1_len, m2_len = m1.mean(dim=(2, 3)).square().sum(dim=1), m2.sum(dim=(2, 3)).square().sum(dim=1)
    vec_norm = m1_len.mul(m2_len)
    torch.set_printoptions(profile="full")
    normalized_norm = torch.clamp(m1.shape[0] * vec_norm / (2 * vec_norm.sum()), 0, 1)
    mid = torch.sort(normalized_norm)
    mid = mid[0][int(m2.shape[0] / 2)]
    index = [x for x in range(len(normalized_norm)) if normalized_norm[x] >= mid]
    # index = [x for x in range(len(normalized_norm))]
    m1, m2 = m1[index, :], m2[index, :]
    return m1, m2


