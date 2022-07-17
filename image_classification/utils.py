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

def twolayer_linearsample(m1, m2, epoch):

    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)

    # mid = torch.sort(vec_norm)
    # mid = mid[0][epoch]
    # index = [x for x in range(len(vec_norm)) if vec_norm[x] >= mid]

    index, norm_x = sample_index_from_bernouli(vec_norm)
    m1 = m1 / norm_x.unsqueeze(1)

    m1, m2 = m1[index, :], m2[index, :]

    return m1, m2

def twolayer_convsample(m1, m2, epoch):
    # print(m1.mean(), m1.max(), m1.min(), m2.mean(), m2.max(), m2.min())
    m1_len, m2_len = m1.mean(dim=(2, 3)).square().sum(dim=1), m2.sum(dim=(2, 3)).square().sum(dim=1)
    vec_norm = m1_len.mul(m2_len)
    
    # mid = torch.sort(vec_norm)
    # mid = mid[0][epoch]
    # index = torch.nonzero((vec_norm >= mid)).squeeze()

    # index = [x for x in range(len(vec_norm)) if vec_norm[x] >= mid]

    index, norm_x = sample_index_from_bernouli(vec_norm)
    m1 = m1 / norm_x.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    m1, m2 = m1[index, :], m2[index, :]

    return m1, m2

def sample_index_from_bernouli(x):
    # print(x.max(), x.min(), x.mean())
    len_x = len(x)
    norm_x = x * len_x / (2 * x.sum())
    # print(norm_x)
    typeflag ='NoNoNo'
    randflag = torch.rand(1)

    cnt = 0
    while norm_x.max() > 1 and cnt < len_x / 2:
        small_index = torch.nonzero((norm_x < 1)).squeeze()
        small_value = norm_x[small_index]
        cnt = len_x - len(small_index)
        norm_x = torch.clamp(norm_x, 0, 1)
        if small_value.max() == 0 and small_value.min() == 0:
            break
        # print(len(x), cnt)
        small_value = small_value * (len_x // 2 - cnt) / small_value.sum()
        norm_x[small_index] = small_value

        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("norm x is {}".format(norm_x))
        # print("sorted norm x is {}".format(norm_x.sort()[0]))
        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("sum up to {}".format(norm_x.sum()))
        # print("cnt is {}".format(cnt))
        # print("_______________________________________________________________________________________________________")
        # exit(0)
    if norm_x.max() > 1 or norm_x.min() < 0:
        typeflag = 'debug'
        print("We change it to debug mode because of the Bernoulli")
    if typeflag == 'debug':
        with open("debug.txt", "a") as f:
            f.write("raw {} is {}\n".format(randflag, x))
        with open("debug.txt", "a") as f:
            f.write("the after norm {} is {}\n".format(randflag, norm_x))
    # print("norm x is {}".format(norm_x))
    sample_index = torch.bernoulli(norm_x)
    # print("sample_index is {}
    if typeflag == 'debug':
        with open("debug.txt", "a") as f:
            f.write("sample index {} is {}\n".format(randflag, sample_index))
    # index = [x for x in range(len(sample_index)) if sample_index[x] == 1]
    # try:
    if sample_index.max() > 1 or sample_index.min() < 0:
        print(sample_index)
        print(x)
    index = torch.nonzero((sample_index == 1)).squeeze()
    if typeflag == 'debug':
        with open("debug.txt", "a") as f:
            f.write("index {} is {}\n".format(randflag, index))
    # print("bernoulli", x, '\n', index, '\n', norm_x, '\n', len(index))
    return index, norm_x






