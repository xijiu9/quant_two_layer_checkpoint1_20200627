import torch
import math
import time
import pytorch_minimax
from quantizers import get_transform
import numpy as np


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]
Qqs = [torch.tensor(1.0), torch.ones(1)]
Qmax = [1.0, 1.0]


def init(max_bs):
    for i in range(2, max_bs+1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])
        Qqs.append(H)
        Qmax.append(Hmax)


class Preconditioner:
    def __init__(self, x, num_bits, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1

        self.x = self.flatten(x)
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        self.x_shape2 = x.shape
        self.x_shape_double = torch.cat([x, x], dim=0).shape
        return x.view(x.shape[0], -1)

    def deflatten(self, Tx):
        try:
            x = Tx.view(*self.x_shape2)
        except:
            x = Tx.view(*self.x_shape_double)
        return x

    def forward(self):
        return self.Tx

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        qzero = -self.zero_point * self.scale
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ScalarPreconditionerAct(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditionerAct, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = x.min() - 1e-8
            mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class DiagonalPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(DiagonalPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            if self.left:
                mn = pytorch_minimax.min(x).unsqueeze(1) - 1e-8
                mx = pytorch_minimax.max(x).unsqueeze(1) + 1e-8
            else:
                mn = x.min(0, keepdims=True)[0] - 1e-8
                mx = x.max(0, keepdims=True)[0] + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


total_time = 0


class BlockwiseHouseholderPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(BlockwiseHouseholderPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        # self.T = self.get_transform(x)
        with torch.no_grad():
            mvec = pytorch_minimax.max(x) - pytorch_minimax.min(x) + 1e-8
        self.T, self.T_inv = get_transform(mvec.cpu(), Qqs, Qmax)
        self.T = self.T.cuda()
        self.T_inv = self.T_inv.cuda()
        # self.T = torch.eye(x.shape[0]).cuda()

        x = self.T @ x
        with torch.no_grad():
            mn = pytorch_minimax.min(x).unsqueeze(1) - 1e-8
            mx = pytorch_minimax.max(x).unsqueeze(1) + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        # print('fin')
        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        x = x / self.scale + self.zero_point
        return self.T_inv @ x
        # return self.T.inverse() @ x
        # return self.T @ x

    @staticmethod
    def get_transform(x):
        global total_time

        N = x.shape[0]
        x = x.view(N, -1)

        mvec = pytorch_minimax.max(x) - pytorch_minimax.min(x)
        rank = (-mvec).argsort()
        values = mvec[rank]

        # Get block configurations
        num_zeros = 0
        total_values = values.sum()
        while True:
            num_zeros += 1
            total_values -= values[N - num_zeros]
            num = num_zeros * values[N - num_zeros - 1] / total_values
            if num >= 1:
                break

        num_nonzeros = N - num_zeros
        nums = (num_zeros * values / total_values)[:num_nonzeros]
        nums = torch.round(torch.cumsum(nums, 0)).int()

        # Construct the matrix
        T = torch.zeros(N, N).cuda()
        all_s = torch.zeros(N).cuda()

        cnt = num_nonzeros
        indices = []
        index_cnt = 0

        for i in range(num_nonzeros):
            # [i] + [cnt ~ num_nonzeros + nums[i]]
            indices.append(i)
            lambda_1 = values[i]
            lambda_2 = values[cnt]
            sz = num_nonzeros + nums[i] - cnt + 1
            Q, Qmax = Qs[sz]
            w = torch.tensor([lambda_1 / math.sqrt(sz), lambda_2 * Qmax])
            s = torch.tensor([w[0] ** (-1 / 3), (w[1] / (sz - 1)) ** (-1 / 3)])
            s *= (1 / s).norm()
            all_s[index_cnt] = s[0]
            all_s[index_cnt+1 : index_cnt+sz] = s[1]
            T[index_cnt:index_cnt+sz, index_cnt:index_cnt+sz] = Q
            index_cnt += sz
            for j in range(cnt, num_nonzeros + nums[i]):
                indices.append(j)
            cnt = num_nonzeros + nums[i]

        t = time.time()
        # print(nums)
        # print(indices)
        # print(num_nonzeros)
        assert len(indices) == N

        T = T @ torch.diag(all_s)
        indices = rank[indices]
        inv_indices = torch.zeros(N, dtype=torch.int64).cuda()
        inv_indices[indices] = torch.arange(N).cuda()

        T = T[inv_indices]
        T = T[:, inv_indices]
        total_time += time.time() - t
        print(total_time)

        return T

class TwoLayerWeightPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(TwoLayerWeightPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point1 = mn
        self.scale1 = self.num_bins / (mx - mn)

        qzero = -self.zero_point1 * self.scale1
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale1 = self.num_bins / (mx - mn)

        first_transform = (x - self.zero_point1) * self.scale1
        # noise = first_transform.new(first_transform.shape).uniform_(-0.5, 0.5)
        # first_transform.add_(noise)
        first_transform.clamp_(0.0, self.num_bins).round_()
        first_quantize = first_transform / self.scale1 + self.zero_point1

        residual = x - first_quantize

        with torch.no_grad():
            mn = min(residual.min() - 1e-8, 0)
            mx = max(residual.max() + 1e-8, 0)

        self.zero_point2 = mn
        self.scale2 = self.num_bins / (mx - mn)

        qzero = -self.zero_point2 * self.scale2
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale2 = self.num_bins / (mx - mn)

        second_transform = (residual - self.zero_point2) * self.scale2
        output = torch.cat([first_transform, second_transform], dim=0)
        # print("the integer is {}".format(output))
        # print("quantize shape is {}".format(output.shape))
        return output

    def inverse_transform(self, x):
        half_shape = int(x.shape[0] / 2)
        first, second = torch.split(x, [half_shape, half_shape], dim=0)
        # print("first shape is {}, second shape is {}".format(first.shape, second.shape))
        first = first / self.scale1 + self.zero_point1
        second = second / self.scale2 + self.zero_point2
        dequantize = torch.cat([first, second], dim=0)
        # print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1, self.zero_point2))
        # print("dequantize shape:{}".format(dequantize.shape))
        # print("input is {}, dequantize is {}, difference is ".format(x, dequantize, x-dequantize))
        return dequantize



class lsq_per_tensor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale, config, bits, symm, inputtype=''):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2 if symm else 0
        num_features = input.numel()
        grad_scale = 1.0 / np.sqrt(num_features * num_bins)
        # grad_scale = 1.0 / np.sqrt(num_features)

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        # Step size gradient
        error = vbar - transformed
        mask = torch.logical_and(transformed >= 0, transformed <= num_bins)
        case1 = (transformed < 0).float() * bias
        case2 = mask.float() * error
        case3 = (transformed > num_bins).float() * (bias + num_bins)
        # TODO gradient scale might be too small, so optimizing without AdaGrad might be problematic...
        ss_gradient = (case1 + case2 + case3) * grad_scale  # * 100 * scale
        ctx.save_for_backward(mask, ss_gradient)
        ctx.others = config, inputtype
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        mask, ss_gradient = ctx.saved_tensors
        config, inputtype = ctx.others
        # if config.epoch < config.freeze_step and inputtype == "activation":
        #     return grad_output * mask.float(), (grad_output * ss_gradient).sum() * config.epoch / config.freeze_step, None, None, None, None
        return grad_output * mask.float(), (grad_output * ss_gradient).sum(), None, None, None, None

