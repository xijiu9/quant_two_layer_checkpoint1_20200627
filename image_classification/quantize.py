from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

import matplotlib.pyplot as plt
import actnn.cpp_extension.backward_func as ext_backward_func
import numpy as np
from tqdm import trange

try:
    from image_classification.preconditioner import ScalarPreconditioner, DiagonalPreconditioner, \
        BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct, TwoLayerWeightPreconditioner
    from image_classification.utils import twolayer_linearsample, twolayer_convsample
except:
    from utils import twolayer_linearsample, twolayer_convsample, sample_index_from_bernouli, twolayer_linearsample_debug, twolayer_convsample_debug
    from preconditioner import ScalarPreconditioner, DiagonalPreconditioner, \
        BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct, TwoLayerWeightPreconditioner


class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.bweight_num_bits_c = 8
        self.bweight_num_bits_l = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True
        self.twolayer_weight = False
        self.epoch = 0

    def activation_preconditioner(self):
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self):
        if self.hadamard:
            return lambda x: BlockwiseHouseholderPreconditioner(x, self.backward_num_bits)
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self, layertype=''):
        # if self.backward_persample:
        #     return lambda x: DiagonalPreconditioner(x, self.bweight_num_bits, left=False)
        if self.twolayer_weight:
            if layertype == 'linear' or layertype == 'conv':
                # if layertype == 'linear':
                return lambda x: TwoLayerWeightPreconditioner(x, self.bweight_num_bits_l)
        # else:
        return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)


config = QuantizationConfig()

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False, debug=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('---')
        #     print(input.view(-1)[:10], input.min(), input.max())
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(0.0, preconditioner.num_bins).round_()

            inverse_output = preconditioner.inverse(output)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        if not debug:
            return inverse_output
        return output, inverse_output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False, debug=False):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace, debug)


class conv2d_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.saved = input, weight, bias
        ctx.other_args = stride, padding, dilation, groups
        ctx.inplace = False
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        # torch.save(grad_output, 'image_classification/ckpt/grad_output_conv_180.pt')
        # print(grad_output.mean(), grad_output.max(), grad_output.min())
        grad_output_weight_condi = quantize(grad_output, config.weight_gradient_preconditioner(layertype='conv'),
                                            stochastic=True)
        grad_output_active_condi = quantize(grad_output, config.activation_gradient_preconditioner(), stochastic=True)
        input, weight, bias = ctx.saved
        stride, padding, dilation, groups = ctx.other_args

        # torch.save(
        #     {"input": input, "weight": weight, "bias": bias, "stride": stride, "padding": padding, "dilation": dilation
        #         , "groups": groups}, 'image_classification/ckpt/inputs_conv_180.pt')
        if config.twolayer_weight:
            input_sample, grad_output_weight_condi_sample = twolayer_convsample(torch.cat([input, input], dim=0),
                                                                                grad_output_weight_condi, config.epoch)
            _, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [False, True])
            # torch.save(grad_weight, 'image_classification/ckpt/grad_weight_conv_180.pt')
            # exit(0)
        else:
            _, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output_weight_condi, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [False, True])

        grad_input, _ = ext_backward_func.cudnn_convolution_backward(
            input, grad_output_active_condi, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [True, False])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class linear_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.saved = input, weight, bias
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # torch.set_printoptions(profile="full")
        # print(grad_output[:, :])
        # exit(0)
        torch.set_printoptions(profile="full", linewidth=160)
        # torch.save(grad_output, 'image_classification/ckpt/grad_output_linear_0.pt')
        grad_output_weight_conditioner = quantize(grad_output,
                                                  config.weight_gradient_preconditioner(layertype='linear'),
                                                  stochastic=True)
        grad_output_active_conditioner = quantize(grad_output, config.activation_gradient_preconditioner(),
                                                  stochastic=True)
        # exit(0)
        input, weight, bias = ctx.saved
        # torch.save(input, 'image_classification/ckpt/input_linear_0.pt')
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        # rank = len(grad_output.shape)

        grad_output_flatten = grad_output.view(-1, C_out)
        grad_output_flatten_weight = grad_output_weight_conditioner.view(-1, C_out)
        grad_output_flatten_active = grad_output_active_conditioner.view(-1, C_out)
        input_flatten = input.view(-1, C_in)

        # print(torch.linalg.norm(grad_output_flatten_weight, dim=1), len(torch.linalg.norm(grad_output_flatten_weight, dim=1)))
        # print(grad_output_flatten_weight[:, :5], grad_output_flatten_weight.shape)
        grad_input = grad_output_flatten_active.mm(weight)
        try:
            grad_weight = grad_output_flatten_weight.t().mm(input_flatten)
        except:
            m1, m2 = twolayer_linearsample(grad_output_flatten_weight, input_flatten, config.epoch)
            grad_weight = m1.t().mm(m2)
        # print(grad_weight.shape, weight.shape)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        #
        # torch.save(grad_weight, "image_classification/ckpt/grad_weight_linear_0.pt")
        #
        # exit(0)
        return grad_input, grad_weight, grad_bias


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, inplace=False, stochastic=False):
        super(QuantMeasure, self).__init__()
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input):
        q_input = quantize(input, config.activation_preconditioner(),
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.acts is not None:
            config.acts.append(input.detach().cpu().numpy())

        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:  # TODO weight quantization scheme...
            qweight = quantize(self.weight, config.weight_preconditioner())
            qbias = self.bias
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact') or not config.quantize_gradient:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = conv2d_act.apply(qinput, qweight, qbias, self.stride,
                                      self.padding, self.dilation, self.groups)

        self.act = output

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, ):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:
            qweight = quantize(self.weight, config.weight_preconditioner())
            if self.bias is not None:
                qbias = quantize(self.bias, config.bias_preconditioner())
            else:
                qbias = None
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact') or not config.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act.apply(qinput, qweight, qbias)

        return output


class QBatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(QBatchNorm2D, self).__init__(num_features)
        self.quantize_input = QuantMeasure()

    def forward(self, input):  # TODO: weight is not quantized
        self._check_input_dim(input)
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        qweight = self.weight
        qbias = self.bias

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, qweight, qbias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


# if __name__ == '__main__':
#
#     type = "100"
#     torch.set_printoptions(profile="full", linewidth=160)
#     grad_output = torch.load("ckpt/grad_output_linear_{}.pt".format(type))
#     inputs = torch.load("ckpt/input_linear_{}.pt".format(type))
#
#     full_grad_weight = grad_output.t().mm(inputs)
#     grad_output_8_sum = None
#     grad_output_2_sum = None
#     grad_weight_8_sum = None
#     grad_weight_2_sum = None
#     num_sample = 10000
#     for i in trange(num_sample):
#         grad_output_8 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 8), stochastic=True)
#         grad_output_2 = quantize(grad_output, lambda x: TwoLayerWeightPreconditioner(x, 4), stochastic=True)
#
#         # grad_weight_2 = grad_output_2.t().mm(torch.cat([inputs, inputs], dim=0))
#         m1, m2 = twolayer_linearsample(grad_output_2, inputs, epoch=0)
#         grad_weight_2 = m1.t().mm(m2)
#         grad_weight_8 = grad_output_8.t().mm(inputs)
#         try:
#             grad_weight_2_sum += grad_weight_2 / num_sample
#             grad_weight_8_sum += grad_weight_8 / num_sample
#             grad_output_2_sum += grad_output_2 / num_sample
#             grad_output_8_sum += grad_output_8 / num_sample
#         except:
#             grad_weight_2_sum = grad_weight_2 / num_sample
#             grad_weight_8_sum = grad_weight_8 / num_sample
#             grad_output_2_sum = grad_output_2 / num_sample
#             grad_output_8_sum = grad_output_8 / num_sample
#
#     print("full gradient: ", full_grad_weight.mean(), full_grad_weight.abs().mean())
#     print("grad_output:   ", grad_output.mean(), grad_output.abs().mean())
#     print("inputs:        ", inputs.mean(), inputs.abs().mean())
#     bias_weight_8 = grad_weight_8_sum - full_grad_weight
#     bias_output_8 = grad_output_8_sum - grad_output
#     print("bias_weight_8  ", bias_weight_8.mean(), bias_weight_8.abs().mean())
#     print("bias_output_8  ", bias_output_8.mean(), bias_output_8.abs().mean())
#     print("_________________________________________________________________________________")
#     bias_weight_2 = grad_weight_2_sum - full_grad_weight
#     bias_output_2 = grad_output_2_sum[:128] + grad_output_2_sum[128:] - grad_output
#     print("bias_weight_2  ", bias_weight_2.mean(), bias_weight_2.abs().mean())
#     print("bias_output_2  ", bias_output_2.mean(), bias_output_2.abs().mean())


if __name__ == '__main__':

    the_type = "180"
    torch.set_printoptions(profile="full", linewidth=160)
    grad_output = torch.load("ckpt/grad_output_conv_{}.pt".format(the_type))
    inputs = torch.load("ckpt/inputs_conv_{}.pt".format(the_type))
    # grad_weight_fake = torch.load('ckpt/grad_weight_conv_{}.pt'.format(the_type))
    inputt, weight, bias, stride, padding, dilation, groups = inputs['input'], inputs['weight'], inputs['bias'], inputs[
        'stride'], inputs['padding'], inputs['dilation'], inputs['groups']

    _, full_grad_weight = ext_backward_func.cudnn_convolution_backward(
        inputt, grad_output, weight, padding, stride, dilation, groups,
        True, False, False,  # ?
        [False, True])
    grad_output_8_sum = None
    grad_output_2_sum = None
    grad_weight_8_sum = None
    grad_weight_2_sum = None
    num_sample = 1
    for i in trange(num_sample):
        grad_output_8 = quantize(grad_output, lambda x: ScalarPreconditioner(x, 8), stochastic=True)
        grad_output_2 = quantize(grad_output, lambda x: TwoLayerWeightPreconditioner(x, 4), stochastic=True)

        input_sample, grad_output_weight_condi_sample = twolayer_convsample(torch.cat([inputt, inputt], dim=0),
                                                                            grad_output_2, epoch=0)
        input_sample_debug, grad_output_weight_condi_sample_debug = twolayer_convsample_debug(torch.cat([inputt, inputt], dim=0),
                                                                            grad_output_2, epoch=int(the_type))
        # grad_weight_2 = grad_output_2.t().mm(torch.cat([inputs, inputs], dim=0))
        _, grad_weight_2 = ext_backward_func.cudnn_convolution_backward(
            input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])
        _, grad_weight_2_d = ext_backward_func.cudnn_convolution_backward(
            input_sample_debug, grad_output_weight_condi_sample_debug, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])
        # brute force
        # input_2 = torch.cat([inputt, inputt], dim=0)
        # new_grad_weight = full_grad_weight.unsqueeze(0).repeat(input_2.shape[0], 1, 1, 1, 1)
        # for i in range(input_2.shape[0]):
        #     g2i, i2i = grad_output_2[i].unsqueeze(0), input_2[i].unsqueeze(0)
        #     _, grad_weight_2_i = ext_backward_func.cudnn_convolution_backward(
        #         i2i, g2i, weight, padding, stride, dilation, groups,
        #         True, False, False,  # ?
        #         [False, True])
        #     new_grad_weight[i] = grad_weight_2_i
        #
        # new_norm, new_abs_norm = new_grad_weight.sum(dim=(1, 2, 3, 4)), new_grad_weight.abs().sum(dim=(1, 2, 3, 4))
        # index = new_abs_norm.sort()[1]
        # # index = index[input_2.shape[0] // 2:]
        # index = index[100:]
        # grad_weight_2 = new_grad_weight[index].sum(dim=0)
        #
        # print(new_norm.sort()[0], new_abs_norm.sort()[0])
        # exit(0)

        _, grad_weight_8 = ext_backward_func.cudnn_convolution_backward(
            inputt, grad_output_8, weight, padding, stride, dilation, groups,
            True, False, False,  # ?
            [False, True])
        try:
            grad_weight_2_sum += grad_weight_2 / num_sample
            grad_weight_8_sum += grad_weight_8 / num_sample
            grad_output_2_sum += grad_output_2 / num_sample
            grad_output_8_sum += grad_output_8 / num_sample
        except:
            grad_weight_2_sum = grad_weight_2 / num_sample
            grad_weight_8_sum = grad_weight_8 / num_sample
            grad_output_2_sum = grad_output_2 / num_sample
            grad_output_8_sum = grad_output_8 / num_sample

    # print("fake gradient: ", grad_weight_fake.mean(), grad_weight_fake.abs().mean())
    print("full gradient: ", full_grad_weight.mean(), full_grad_weight.abs().mean())
    print("grad_output:   ", grad_output.mean(), grad_output.abs().mean())
    print("inputs:        ", inputt.mean(), inputt.abs().mean())
    bias_weight_8 = grad_weight_8_sum - full_grad_weight
    bias_output_8 = grad_output_8_sum - grad_output
    print("bias_weight_8  ", bias_weight_8.mean(), bias_weight_8.abs().mean())
    print("bias_output_8  ", bias_output_8.mean(), bias_output_8.abs().mean())
    print("_________________________________________________________________________________")
    bias_weight_2 = grad_weight_2_sum - full_grad_weight
    bias_output_2 = grad_output_2_sum[:128] + grad_output_2_sum[128:] - grad_output
    print("bias_weight_2  ", bias_weight_2.mean(), bias_weight_2.abs().mean())
    print("bias_output_2  ", bias_output_2.mean(), bias_output_2.abs().mean())
    bias_weight_2_d = grad_weight_2_d - full_grad_weight
    print("bias_weight_2d ", bias_weight_2_d.mean(), bias_weight_2_d.abs().mean())

