from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

import matplotlib.pyplot as plt
import actnn.cpp_extension.backward_func as ext_backward_func
import numpy as np

try:
    from image_classification.preconditioner import ScalarPreconditioner, DiagonalPreconditioner, \
        BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct, TwoLayerWeightPreconditioner
    from image_classification.utils import twolayer_mm, twolayer_convsample
except:
    from utils import twolayer_mm
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
            if layertype == 'linear':
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
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False):

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

            output = preconditioner.inverse(output)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace)


class conv2d_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.saved = input, weight, bias
        ctx.other_args = stride, padding, dilation, groups
        ctx.inplace = False
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output.mean(), grad_output.max(), grad_output.min())
        grad_output_weight_condi = quantize(grad_output, config.weight_gradient_preconditioner(layertype='conv'),
                                            stochastic=True)
        grad_output_active_condi = quantize(grad_output, config.activation_gradient_preconditioner(), stochastic=True)
        input, weight, bias = ctx.saved
        stride, padding, dilation, groups = ctx.other_args

        if config.twolayer_weight and False:
            input_sample, grad_output_weight_condi_sample = twolayer_convsample(torch.cat([input, input], dim=0),
                                                                                grad_output_weight_condi)
            _, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input_sample, grad_output_weight_condi_sample, weight, padding, stride, dilation, groups,
                True, False, False,  # ?
                [False, True])
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
        grad_output_weight_conditioner = quantize(grad_output,
                                                  config.weight_gradient_preconditioner(layertype='linear'),
                                                  stochastic=True)
        grad_output_active_conditioner = quantize(grad_output, config.activation_gradient_preconditioner(),
                                                  stochastic=True)
        # print(grad_output_weight_conditioner[:, :2])
        # exit(0)
        input, weight, bias = ctx.saved

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
            # grad_weight = grad_output_flatten_weight.t().mm(torch.cat([input_flatten, input_flatten], dim=0))
            grad_weight = twolayer_mm(grad_output_flatten_weight, input_flatten)
        # print(grad_weight.shape, weight.shape)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

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


if __name__ == '__main__':
    x = torch.tensor(
        [[3.2723e-08, 1.8726e-09, 1.8328e-11, 3.7134e-13, 5.7877e-12,
          1.2058e-13, 3.7612e-12, 1.4629e-12, -3.5390e-08, 1.1421e-09],
         [0.0000e+00, 1.2839e-12, 4.2918e-11, 3.7341e-14, 2.5342e-12,
          2.5254e-15, 1.0307e-14, 9.0324e-13, 3.9564e-10, 2.1876e-11],
         [1.8798e-10, 3.9651e-14, 0.0000e+00, 1.6028e-10, 3.9125e-12,
          5.2916e-12, 3.2124e-12, 2.3277e-13, 9.5195e-15, 4.2267e-13],
         [1.5126e-09, 3.4958e-10, 4.6997e-11, 1.0813e-07, 3.3118e-07,
          3.5806e-08, 5.8287e-11, -4.8336e-07, 3.1377e-09, 3.3499e-09],
         [7.2615e-08, 8.7851e-07, 2.6572e-09, 1.6795e-08, 1.8037e-10,
          1.1874e-09, 4.2303e-07, 5.8646e-10, 2.1713e-08, -1.4170e-06],
         [1.3008e-05, 5.7069e-09, 3.9206e-11, 9.4548e-11, 1.3130e-10,
          1.7237e-11, 2.2791e-11, 2.9782e-11, -1.5667e-05, 2.6522e-06],
         [-5.1223e-09, 2.9541e-09, 5.4632e-11, 1.5688e-10, 3.8615e-12,
          2.8110e-12, 3.1621e-12, 3.5082e-13, 1.9681e-10, 1.9551e-09],
         [1.8712e-06, -7.5787e-03, 1.3260e-09, 1.8891e-09, 3.3114e-11,
          9.9149e-12, 4.0474e-11, 2.9205e-09, 2.9921e-08, 7.5767e-03],
         [3.4105e-06, 8.0245e-09, -5.5480e-03, 3.0522e-07, 1.7228e-07,
          3.5074e-06, 5.5147e-03, 9.9850e-06, 1.5778e-05, 9.0882e-08],
         [2.7507e-11, 2.6283e-08, 1.1546e-09, 1.9143e-08, 8.5808e-07,
          6.1092e-09, 1.7360e-09, -9.3272e-07, 2.1882e-09, 1.6838e-08],
         [9.9590e-10, 4.9599e-09, 3.9224e-11, 5.0502e-11, 1.2564e-11,
          5.0534e-12, 4.6298e-10, 9.7998e-11, -2.7940e-08, 2.1139e-08],
         [2.0660e-06, 1.2800e-08, 7.8619e-06, 6.1740e-07, 6.8248e-07,
          1.2011e-07, -2.0275e-05, 4.2000e-08, 8.8415e-06, 3.0385e-08],
         [4.5206e-06, 2.8974e-10, 4.8318e-09, 3.1639e-11, 4.0514e-10,
          2.8113e-12, 5.5335e-12, 4.1724e-11, -4.5262e-06, 5.3821e-10],
         [5.8168e-08, 1.5789e-07, 3.4655e-06, -6.5056e-04, 2.0320e-04,
          4.1046e-04, 3.1934e-06, 2.6926e-05, 8.4722e-07, 2.2470e-06],
         [1.2566e-11, 1.7535e-11, -1.1642e-06, 1.4443e-08, 1.1150e-06,
          5.8749e-09, 2.7656e-08, 5.6945e-10, 1.0238e-11, 2.3773e-10],
         [9.9751e-16, 9.4556e-15, 8.8193e-13, 2.9149e-13, 1.9765e-14,
          3.9629e-14, 0.0000e+00, 1.2785e-15, 6.0114e-15, 3.8604e-12],
         [5.2474e-12, 6.8796e-11, 2.2301e-09, 3.8690e-09, 3.7132e-13,
          3.8277e-11, -5.5879e-09, 5.9053e-11, 4.9590e-11, 5.7124e-11],
         [8.8355e-08, -1.4746e-05, 1.2085e-10, 2.2303e-09, 4.2177e-10,
          2.0597e-09, 1.7613e-10, 1.5814e-09, 1.3774e-07, 1.4514e-05],
         [4.1795e-09, 3.8147e-10, 1.1408e-07, 2.5880e-09, 2.3971e-05,
          2.6568e-06, 2.1013e-08, -2.6771e-05, 7.2227e-10, 4.4999e-10],
         [1.2816e-11, 3.8943e-13, 5.3521e-15, 9.1838e-15, 8.8487e-16,
          1.3006e-15, 5.2105e-16, 1.1094e-15, 0.0000e+00, 8.7471e-13],
         [-4.7032e-08, 2.8804e-12, 4.5609e-08, 3.4153e-10, 2.2310e-11,
          3.4854e-10, 7.2005e-11, 2.6493e-11, 1.8344e-12, 5.7750e-11],
         [1.4035e-07, -3.0851e-05, 2.7087e-10, 5.6505e-09, 3.3612e-09,
          4.9976e-07, 9.5896e-10, 1.6801e-10, 1.6878e-07, 3.0032e-05],
         [4.3150e-15, 9.2942e-11, 6.5900e-13, 8.3012e-11, 5.1093e-12,
          4.1822e-12, -1.8626e-09, 3.1278e-13, 6.0843e-12, 1.9364e-09],
         [1.1060e-05, 1.4192e-08, 9.3268e-08, 6.7292e-09, 3.4639e-07,
          7.2862e-08, 5.2858e-09, 8.3547e-07, 4.2281e-03, -4.2405e-03],
         [9.0488e-11, 1.2532e-08, 2.6308e-08, -8.2093e-04, 7.2265e-09,
          8.2086e-04, 1.0166e-08, 1.0577e-08, 8.1931e-10, 1.3579e-09],
         [1.5102e-08, 1.1514e-10, -8.8839e-06, 1.1900e-07, 8.4278e-06, 9.6340e-09, 2.6999e-08, 2.8193e-07,
          2.7061e-09, 1.0861e-10],
         [7.7470e-07, 1.5499e-07, -4.3828e-06, 1.7573e-06, 5.3837e-07,
          1.3154e-07, 9.2179e-07, 6.8999e-08, 2.9296e-08, 6.2894e-09],
         [1.3287e-12, 1.7570e-11, 2.5773e-09, 6.2978e-09, 1.1524e-09,
          -2.8871e-07, 2.1370e-07, 6.4542e-08, 5.2341e-11, 9.4674e-11],
         [1.3139e-05, 3.9227e-06, 5.9427e-04, 1.0354e-04, 7.5398e-06,
          -1.3582e-03, 4.0604e-04, 3.2208e-05, 1.0245e-07, 1.9744e-04],
         [9.9457e-12, 9.6533e-13, 2.5266e-13, 5.2379e-08, 2.0030e-10,
          -5.3551e-08, 2.4511e-11, 4.5821e-10, 1.0539e-11, 5.2867e-12],
         [5.8134e-07, 1.1426e-08, 4.1661e-11, 3.6363e-10, 3.9546e-12,
          1.4174e-12, 8.8575e-13, 4.6768e-10, 2.2888e-07, -8.2189e-07],
         [3.6404e-04, 5.0109e-08, -6.9433e-04, 2.8890e-04, 3.6239e-06,
          4.4473e-07, 3.9596e-07, 4.8246e-08, 2.8540e-09, 3.6825e-05],
         [3.0862e-07, 7.2414e-08, 1.8173e-07, 2.8618e-07, 1.1326e-07,
          3.4893e-08, 4.7353e-07, -1.6736e-06, 1.3627e-07, 6.7208e-08],
         [1.0061e-07, 1.8013e-09, 3.2646e-05, -9.6984e-04, 9.2275e-04,
          1.4190e-05, 6.4576e-08, 7.6465e-08, 2.1398e-09, 5.9927e-09],
         [2.6529e-12, 1.9250e-08, 2.2852e-13, 1.1762e-11, 5.1711e-12,
          6.9524e-12, 7.2433e-11, 4.9493e-14, 4.8603e-13, -1.9558e-08],
         [3.7548e-09, 6.0504e-09, 1.9341e-11, 2.0521e-10, 2.2831e-11,
          5.5729e-11, 3.1141e-10, 1.1061e-11, -1.2107e-08, 1.8073e-09],
         [2.7016e-07, 2.9293e-09, -3.1712e-07, 1.5920e-08, 1.8449e-08,
          2.8665e-10, 3.4438e-09, 1.9193e-09, 1.7010e-10, 4.2280e-09],
         [5.9529e-11, -2.5937e-07, 6.4353e-13, 2.9490e-11, 1.6072e-12,
          8.1096e-12, 2.5223e-10, 4.3213e-11, 1.7195e-07, 8.6236e-08],
         [1.1767e-12, 6.3715e-14, -9.3132e-10, 3.8359e-13, 5.2663e-10,
          5.3332e-11, 1.0721e-13, 1.6403e-10, 1.7811e-14, 8.6391e-15],
         [6.6934e-12, 5.0276e-12, 1.4831e-11, 1.2240e-10, 2.3562e-08,
          7.9257e-09, 1.5237e-10, -3.2596e-08, 2.1961e-11, 3.2395e-12],
         [7.6671e-05, -1.9064e-04, 1.2151e-06, 1.7113e-08, 5.2118e-09,
          6.4434e-09, 1.4833e-06, 7.4820e-09, 1.0195e-04, 9.2797e-06],
         [4.3097e-05, 7.1098e-08, -4.7144e-05, 2.9838e-06, 4.2770e-07,
          3.3360e-08, 3.7676e-08, 2.5934e-07, 1.6904e-07, 6.3207e-08],
         [5.2659e-06, -6.5118e-06, 8.9056e-08, 3.0167e-08, 9.2418e-09,
          2.3823e-09, 1.9007e-10, 1.4115e-08, 1.2810e-07, 9.7263e-07],
         [9.5119e-15, 6.4914e-13, 5.9506e-14, 7.5868e-12, 3.2898e-07,
          -3.2876e-07, 4.8278e-12, 2.2397e-11, 1.5301e-14, 2.2758e-11],
         [4.1946e-08, 1.6516e-07, 6.8781e-11, 3.3467e-09, 2.5057e-11,
          2.3493e-12, 3.6105e-12, 3.4837e-11, 1.0964e-09, -2.1141e-07],
         [3.9129e-10, 3.5505e-09, 6.6246e-09, 1.3870e-03, 2.4655e-04,
          -1.6369e-03, 3.7571e-08, 3.2947e-06, 1.0356e-08, 1.4547e-09],
         [3.1016e-09, 9.6646e-10, 6.7175e-07, -7.1195e-06, 2.7298e-06,
          3.2278e-06, 4.6994e-08, 3.7033e-07, 8.0453e-09, 6.0603e-08],
         [4.4836e-14, 1.6083e-11, 1.5409e-07, 1.7036e-11, 7.5068e-07,
          5.9836e-10, 2.7455e-10, -9.0618e-07, 1.9386e-13, 1.2629e-13],
         [1.8906e-12, 2.7100e-12, 2.0941e-10, 8.5401e-06, 1.3129e-08,
          -8.5593e-06, 1.7286e-10, 5.4844e-09, 1.3452e-12, 8.2669e-12],
         [4.4393e-09, 2.5051e-08, -2.1141e-07, 5.1675e-08, 1.5158e-08,
          2.4529e-08, 2.0408e-09, 4.0507e-08, 4.8816e-08, 1.7642e-10],
         [3.7929e-14, 9.5760e-14, 1.4104e-07, 9.4840e-10, -1.4901e-07,
          6.1945e-09, 3.3410e-10, 7.5528e-10, 5.1315e-13, 9.0922e-13],
         [5.0851e-12, -5.1223e-09, 1.4699e-16, 3.4101e-15, 2.0726e-12, 3.2163e-15, 9.0246e-14, 2.3109e-12,
          2.8200e-12, 4.6811e-09],
         [4.8036e-07, 3.4005e-10, -3.6792e-06, 1.0336e-08, 3.1847e-06,
          6.8856e-10, 6.9984e-10, 1.6243e-09, 9.7360e-11, 1.8686e-10],
         [0.0000e+00, 3.0212e-13, 3.4199e-13, 4.3121e-15, 5.5533e-14,
          1.5869e-15, 4.8933e-16, 1.7377e-12, 1.8427e-13, 1.2947e-12],
         [-3.1232e-03, 4.3633e-06, 6.5760e-08, 1.6574e-07, 3.4435e-06,
          6.7622e-07, 1.2993e-06, 5.5054e-07, 3.1122e-03, 4.5398e-07],
         [6.8454e-14, 7.0180e-13, 4.5975e-11, 2.5895e-10, 1.5255e-12,
          2.1468e-12, 0.0000e+00, 6.8438e-13, 1.3429e-12, 2.4532e-13],
         [-1.1860e-04, 8.6875e-07, 3.5597e-05, 2.1084e-07, 1.1445e-08,
          3.8255e-09, 7.3938e-09, 4.6039e-06, 7.6882e-05, 4.1244e-07],
         [6.2410e-03, 4.8848e-07, -6.5799e-03, 1.3150e-05, 1.9256e-06,
          7.6160e-06, 2.3665e-06, 8.6161e-07, 3.0770e-04, 4.7140e-06],
         [3.3187e-07, 6.5013e-06, 6.9707e-06, 4.6361e-05, 8.9903e-07,
          -1.3964e-03, 1.3232e-03, 3.3955e-07, 7.8679e-08, 1.1786e-05],
         [1.4085e-09, 4.1443e-07, 2.0089e-09, 8.1338e-08, 6.1016e-09,
          8.7000e-08, 9.0343e-10, 1.0227e-07, 9.0420e-09, -7.0408e-07],
         [8.2217e-12, 1.4316e-11, 1.4887e-10, 5.9599e-11, 4.3070e-12,
          5.0300e-11, 0.0000e+00, 7.5027e-12, 1.4402e-11, 5.4350e-12],
         [1.9508e-09, 2.8600e-08, 8.3786e-07, 1.5340e-04, 6.5938e-06,
          -2.2703e-04, 4.2678e-05, 2.0858e-05, 1.2447e-08, 2.6221e-06],
         [6.9119e-10, 2.0796e-11, -1.4901e-07, 1.3367e-08, 2.8522e-08,
          3.4831e-08, 3.1016e-10, 7.0879e-08, 3.2610e-10, 1.5732e-11],
         [2.1732e-11, 6.6841e-11, 1.5161e-11, 4.7664e-07, 2.9523e-10,
          -4.8056e-07, 1.9682e-11, 7.5573e-10, 1.2783e-09, 5.5290e-10],
         [1.5646e-08, -9.0711e-07, 8.5630e-11, 6.5593e-11, 5.2351e-10,
          2.5121e-10, 5.1565e-10, 4.2009e-10, 7.1082e-09, 8.8175e-07],
         [-1.0906e-06, 4.0843e-09, 3.1378e-09, 1.5235e-11, 2.9963e-13,
          2.8880e-14, 4.0267e-13, 4.2741e-13, 1.0710e-06, 1.2347e-08],
         [5.1792e-10, 1.3675e-09, 1.4064e-07, -5.7528e-06, 5.4549e-06,
          1.3314e-07, 1.3445e-08, 1.1446e-09, 3.0316e-10, 7.1033e-09],
         [2.3330e-10, 9.4273e-10, 2.7791e-09, -6.0350e-07, 5.9732e-07,
          4.5556e-10, 1.9945e-11, 1.1486e-12, 2.1892e-14, 1.5850e-09],
         [1.8931e-12, 3.9303e-12, 4.6279e-11, 4.1342e-09, 1.1346e-09,
          -5.5879e-09, 3.3668e-10, 4.0171e-12, 2.5194e-11, 7.5895e-11],
         [2.7795e-06, 6.2287e-06, -2.8528e-04, 2.5685e-04, 3.6653e-07,
          5.2698e-08, 2.3634e-08, 1.4659e-06, 6.0342e-07, 1.6909e-05],
         [3.6688e-10, 2.7392e-06, 8.1445e-09, -1.4733e-05, 3.6170e-09,
          1.1959e-05, 4.8826e-10, 4.7406e-10, 1.6083e-08, 4.5443e-09],
         [1.8464e-10, -1.2759e-07, 3.4867e-13, 7.5779e-11, 4.9382e-12,
          9.1312e-10, 3.1328e-13, 1.9443e-13, 3.8047e-12, 1.2653e-07],
         [7.6268e-09, 1.5171e-09, 1.4830e-06, -4.2906e-06, 1.0057e-06,
          1.7013e-06, 9.0343e-08, 4.1455e-10, 7.7966e-10, 5.9920e-10],
         [9.8577e-12, 9.1000e-12, -9.3132e-10, 8.6958e-11, 1.9184e-10,
          5.9856e-10, 2.1636e-12, 2.3401e-11, 4.7621e-11, 3.3857e-12],
         [2.7840e-10, 4.9191e-09, 4.0987e-08, 3.0265e-05, 1.7933e-08,
          -3.2434e-05, 3.5154e-09, 2.1009e-06, 1.3052e-10, 8.8623e-11],
         [2.4185e-09, 3.1394e-10, 4.3859e-11, 2.2856e-10, 1.0903e-12,
          2.1793e-09, 4.3983e-12, -5.5879e-09, 3.3126e-11, 9.7235e-11],
         [1.4542e-10, 5.9192e-10, 2.8404e-06, -3.3924e-03, 4.7292e-07,
          3.3877e-03, 1.3041e-07, 1.3270e-06, 7.5350e-10, 2.2300e-10],
         [2.7190e-07, 4.5016e-08, 1.4501e-04, 1.6489e-07, 2.1524e-05, 7.9473e-07, -1.6887e-04, 4.4253e-07,
          1.1960e-07, 4.9952e-07],
         [3.1870e-09, 5.5263e-09, 1.5675e-07, 3.0625e-07, 8.9553e-05,
          8.4265e-07, 7.1801e-10, -9.0880e-05, 5.4722e-09, 5.8423e-09],
         [2.0642e-10, 3.0347e-11, 5.9344e-16, 4.1953e-15, 3.1584e-17,
          5.5697e-15, 4.6146e-16, 6.5532e-15, 1.5927e-13, 0.0000e+00],
         [5.9506e-09, 2.9420e-10, 4.6003e-11, 1.0971e-11, 3.4990e-12,
          2.3554e-11, 8.4568e-12, 1.1838e-11, -6.9849e-09, 8.6419e-10],
         [5.9695e-09, 2.4625e-08, 4.1214e-05, 4.4820e-05, 4.1082e-05,
          3.1773e-04, 1.1253e-07, -4.4518e-04, 1.1185e-07, 8.2666e-08],
         [2.1727e-09, 2.4851e-09, 9.2012e-09, -5.2080e-06, 1.0487e-06,
          1.0568e-06, 8.4339e-09, 3.0271e-06, 3.1830e-09, 5.1493e-08],
         [4.8693e-09, 5.2423e-13, -1.1176e-08, 1.7839e-11, 2.5684e-10,
          5.9799e-09, 1.3882e-12, 1.5237e-11, 5.8655e-14, 3.9123e-13],
         [3.7769e-06, 1.9467e-06, 9.0496e-07, 7.0460e-05, 6.8865e-06,
          -8.9523e-05, 3.2952e-06, 1.2992e-08, 1.6627e-08, 2.2224e-06],
         [2.2258e-10, -9.3132e-09, 4.4973e-13, 2.3656e-11, 2.9561e-12,
          1.2824e-11, 4.0463e-11, 5.2777e-12, 4.2724e-09, 4.7766e-09],
         [-3.9271e-05, 5.9206e-07, 2.7434e-05, 7.6221e-07, 2.1152e-06,
          4.0926e-08, 8.9143e-09, 8.1891e-09, 8.2589e-06, 4.9349e-08],
         [3.8982e-09, 3.3120e-09, 3.4650e-07, 3.1755e-06, 5.1447e-06,
          -5.8936e-05, 2.6777e-06, 4.7582e-05, 9.1914e-10, 1.0474e-09],
         [-2.1402e-03, 7.8364e-06, 1.1178e-05, 2.6970e-06, 1.3339e-07,
          5.8009e-07, 3.7441e-07, 1.2060e-07, 2.1135e-03, 3.7482e-06],
         [6.6596e-12, 1.8862e-10, 4.8303e-07, -4.0450e-04, 3.8036e-05,
          3.6592e-04, 3.0300e-09, 6.4140e-08, 7.3851e-12, 9.3056e-11],
         [9.3077e-12, 7.6225e-11, 1.1607e-09, 1.7211e-09, 5.2251e-12,
          1.6650e-11, -5.5879e-09, 1.7286e-11, 2.2474e-10, 3.0055e-09],
         [1.1289e-10, 3.3414e-08, -6.5984e-07, 5.7999e-08, 1.4461e-08,
          8.0792e-08, 4.6740e-07, 2.8153e-09, 2.1268e-09, 5.3880e-10],
         [1.5422e-07, 1.0457e-07, -4.3444e-05, 1.3328e-05, 2.5200e-06,
          4.4916e-06, 1.8138e-07, 2.1450e-05, 9.2288e-07, 2.9058e-07],
         [9.6707e-11, 3.1082e-11, 5.4546e-05, -5.6200e-05, 2.1126e-07,
          4.7546e-08, 1.3885e-06, 3.8223e-09, 1.1159e-10, 2.4447e-09],
         [2.9384e-09, 2.2558e-10, 2.5940e-09, 7.8352e-10, 3.0223e-08,
          2.1750e-07, 1.0728e-10, -2.7753e-07, 2.2625e-08, 6.0068e-10],
         [4.5645e-10, 2.8784e-11, 1.8176e-11, 1.1054e-11, 1.1297e-10,
          4.8647e-11, 1.1265e-10, 2.1539e-11, -9.3132e-10, 6.0247e-10],
         [1.0120e-12, 2.4696e-10, 2.8686e-06, -1.0041e-05, 6.9399e-06,
          5.1745e-09, 2.2333e-07, 3.7278e-09, 2.0251e-13, 1.5131e-12],
         [3.2049e-11, -7.4506e-09, 1.3312e-10, 9.3154e-11, 4.1041e-13,
          1.0242e-11, 4.4718e-12, 6.0288e-12, 3.4649e-09, 3.8391e-09],
         [5.6580e-12, 1.0530e-11, 4.2929e-10, 1.6035e-11, -5.5879e-09,
          1.7049e-10, 2.3743e-11, 4.8234e-09, 6.3409e-12, 1.0380e-10],
         [1.1593e-11, -2.2004e-04, 7.8678e-11, 1.0258e-10, 1.2098e-10,
          4.8307e-11, 1.1005e-09, 8.6839e-12, 1.0034e-09, 2.2004e-04],
         [4.5498e-07, 8.7311e-09, 6.5583e-10, 6.0718e-10, 5.1461e-08,
          3.5304e-10, 5.4395e-09, -3.3123e-05, 2.2274e-05, 1.0326e-05],
         [4.5095e-09, -1.9558e-08, 4.8668e-12, 2.3043e-09, 7.3919e-12,
          3.8640e-11, 1.6872e-12, 2.1656e-10, 3.6530e-11, 1.2450e-08],
         [2.2490e-08, 5.6673e-12, 6.0541e-07, 2.5771e-08, 1.3527e-06,
          6.6617e-10, -2.0089e-06, 9.5871e-11, 6.4708e-11, 2.0020e-09],
         [1.1273e-07, 5.7621e-10, 6.1123e-05, -1.4993e-03, 1.2961e-06, 1.4367e-03, 7.8690e-08, 4.2979e-08,
          1.9186e-09, 2.8899e-10],
         [5.9468e-13, 1.4475e-11, 2.8940e-10, 1.1450e-11, 1.3913e-10,
          1.8412e-10, 1.3296e-11, 0.0000e+00, 5.3911e-14, 3.2695e-12],
         [6.4241e-11, 2.4023e-10, 2.1466e-13, 9.5261e-14, 4.0693e-14,
          2.4906e-14, 5.0331e-14, 6.2767e-14, 0.0000e+00, 1.6979e-11],
         [-7.5437e-08, 1.2523e-11, 6.8736e-08, 9.2408e-13, 1.8806e-11,
          7.8575e-13, 4.2385e-10, 3.0435e-10, 6.7868e-09, 5.8259e-11],
         [6.2535e-09, 7.8235e-10, 1.3037e-07, 4.1416e-07, 1.3259e-08,
          3.2923e-05, 4.1842e-10, -3.3489e-05, 7.2012e-11, 5.1816e-11],
         [1.0250e-12, 5.9624e-12, 7.3074e-09, 5.6033e-11, 3.3483e-10,
          3.0742e-10, -8.3819e-09, 2.5553e-11, 1.0560e-11, 2.0369e-10],
         [8.8663e-12, 1.1147e-05, 2.5557e-11, 5.6387e-08, 6.1591e-07,
          2.1807e-08, 6.3735e-09, 6.3538e-10, 9.8120e-11, -1.1848e-05],
         [0.0000e+00, 2.0071e-14, 3.7319e-12, 2.7638e-13, 5.0907e-11,
          9.7899e-14, 1.5427e-13, 4.3071e-14, 1.4011e-11, 2.4944e-13],
         [3.2905e-08, -4.0222e-03, 5.8992e-09, 3.3272e-06, 5.4071e-06,
          1.3514e-07, 1.6164e-05, 3.0778e-06, 8.4238e-08, 3.9939e-03],
         [1.1991e-08, -4.1335e-05, 1.0581e-11, 7.5170e-09, 7.7859e-11,
          5.1520e-09, 7.5360e-09, 4.2411e-10, 4.0928e-05, 3.7345e-07],
         [6.8469e-10, 2.6020e-10, -3.5707e-06, 3.2767e-06, 2.7656e-07,
          6.4589e-09, 1.4103e-09, 7.8626e-09, 7.1916e-12, 5.2973e-11],
         [1.1980e-09, 6.3664e-10, -4.2655e-07, 1.5638e-10, 1.6770e-07,
          9.6225e-08, 5.0043e-12, 1.4242e-07, 1.7320e-08, 4.0191e-10],
         [7.4812e-09, 4.1178e-12, 2.1591e-13, -4.3213e-07, 1.3678e-12,
          4.1836e-07, 3.7950e-13, 4.5864e-10, 1.7453e-11, 6.3509e-09],
         [4.7143e-15, 1.0769e-12, 1.9529e-10, 6.7088e-14, 1.2205e-13,
          2.1459e-12, 0.0000e+00, 2.6324e-15, 1.5894e-14, 2.1385e-12],
         [3.1387e-05, 1.6667e-04, 1.8864e-08, 3.9119e-07, 4.6965e-10,
          7.6732e-09, 4.8848e-07, 1.0531e-09, 3.8142e-07, -1.9935e-04],
         [1.4399e-14, 0.0000e+00, 2.1893e-17, 8.0288e-17, 7.5061e-17,
          1.8246e-16, 1.7449e-18, 9.2100e-17, 2.2292e-16, 3.9279e-12],
         [7.9963e-11, 7.7316e-11, 2.6841e-09, 8.7140e-09, 3.9448e-07,
          1.8533e-11, -4.3819e-07, 2.4810e-11, 3.8519e-10, 3.1317e-08],
         [5.7766e-11, -9.3132e-10, 2.9035e-11, 4.8084e-12, 6.1087e-11,
          7.2756e-11, 1.0504e-11, 3.6761e-12, 5.4000e-11, 8.5912e-10],
         [1.0904e-12, 2.3713e-13, 3.6111e-10, -1.3970e-08, 1.3765e-08,
          7.4757e-11, 2.0344e-11, 4.8014e-14, 4.3650e-15, 1.1790e-14],
         [2.7009e-14, 1.2168e-08, 4.9602e-17, 1.0228e-15, 3.0059e-16,
          9.9693e-16, 3.4195e-16, 1.7301e-15, 1.1059e-13, -1.2107e-08],
         [8.4607e-10, 8.6721e-09, 3.7249e-06, 3.7815e-05, 1.9184e-05,
          -1.0357e-04, 4.2805e-05, 2.5672e-08, 1.4537e-09, 5.4067e-09],
         [5.8709e-05, 7.4178e-06, 1.4064e-05, 3.0014e-03, 9.6473e-06,
          -3.1228e-03, 7.5226e-06, 1.8305e-06, 1.3809e-05, 8.4017e-06],
         [5.4157e-10, 1.4607e-09, 7.5473e-11, 2.2592e-08, -1.1758e-06,
          4.0005e-07, 1.6278e-10, 7.5104e-07, 5.0492e-10, 2.7208e-10],
         [-4.5169e-08, 9.5584e-11, 2.4388e-08, 2.8037e-11, 1.1194e-09,
          6.6400e-11, 2.2376e-09, 1.3059e-09, 1.2491e-08, 2.7177e-09],
         [5.1373e-07, 1.0858e-08, 1.1244e-12, 4.6535e-11, 5.6030e-06,
          6.4383e-11, 1.4114e-09, 4.9996e-10, -6.1584e-06, 2.8417e-08]])
    print(x.shape)
    x_two = quantize(x, lambda x: TwoLayerWeightPreconditioner(x, 4))
    x_ptq_8 = quantize(x, lambda x: ScalarPreconditioner(x, 8))
    # print(x)
    print(x_two)
    print(x_ptq_8)
    # print(torch.abs(x - x_two[:x.shape[0], :] - x_two[-x.shape[0]:, :]).sum())
    # print(torch.abs(x - x_ptq_8).sum())
