import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


# ********************* 二值(+-1) ***********************
# A
class Binary_a(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # *******************ste*********************
        grad_input = grad_output.clone()
        # ****************saturate_ste***************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        '''
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        '''
        return grad_input


# W
class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* 三值(+-1、0) ***********************
class Ternary(Function):

    @staticmethod
    def forward(self, input):
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(
            torch.add(torch.sign(torch.add(input, threshold)), torch.sign(torch.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* A(特征)量化(二值) ***********************
class activation_bin(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = Binary_a.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
            # ******************** A —— 1、0 *********************
            # a = torch.clamp(a, min=0)
        else:
            output = self.relu(input)
        return output


# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean)  # W中心化(C方向)
    w.data.clamp(-1.0, 1.0)  # W截断
    return w


class weight_tnn_bin(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W
        self.mask_flag = False

    def binary(self, input):
        output = Binary_w.apply(input)
        return output

    def ternary(self, input):
        output = Ternary.apply(input)
        return output

    def cal_mask_ternary(self, weights, rate=70):  # 暂时未使用
        mask = []
        origin_weights = list(weights.cpu().data.abs().numpy().flatten())  #
        if len(origin_weights) != len(origin_weights):
            print("维度不匹配不予剪枝")
            return torch.ones(weights.size())
        threshold_weight = np.percentile(np.array(origin_weights), rate)
        data_len = len(origin_weights)
        count = 0
        for i in range(data_len):
            prun_val = 1
            if origin_weights[i] < threshold_weight and self.atten_mask[i] == 1:
                prun_val = 0
                count += 1
            mask.append(prun_val)
        # print("true cut rate:", count / data_len)
        mask = torch.tensor(mask)
        mask = to_var(mask)  # Variable(mask, requires_grad=False, volatile=False)
        mask = mask.view(weights.size())
        return mask

    def set_atten_mask(self, mask):
        self.atten_mask = mask
        # self.mask = self.cal_mask_ternary(output, 50)
        # self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def set_mask(self, mask):
        self.mask = mask
        self.mask_flag = True

    def forward(self, input):
        if self.W == 2 or self.W == 3:
            # **************************************** W二值 *****************************************
            if self.W == 2:
                output = meancenter_clampConvParams(input)  # W中心化+截断
                # output = input
                # **************** channel级 - E(|W|) ****************
                E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
                # **************** α(缩放因子) ****************
                alpha = E
                # ************** W —— +-1 **************
                if hasattr(self, "mask_flag") and self.mask_flag:
                    # self.mask = self.cal_mask_ternary(output, 50)
                    # print("output device:",output.device,"mask device:",mask.device)
                    output = output * self.mask
                output = self.binary(output)  # 加个if判断有没有flag
                # ************** W * α **************
                output = output * alpha  # 若不需要α(缩放因子)，注释掉即可
                # **************************************** W三值 *****************************************
            elif self.W == 3:
                output_fp = input.clone()
                # ************** W —— +-1、0 **************
                output, threshold = self.ternary(input)
                # **************** α(缩放因子) ****************
                output_abs = torch.abs(output_fp)
                mask_le = output_abs.le(threshold)
                mask_gt = output_abs.gt(threshold)
                output_abs[mask_le] = 0
                output_abs_th = output_abs.clone()
                output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
                mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                alpha = output_abs_th_sum / mask_gt_sum  # α(缩放因子)
                # *************** W * α ****************
                # output = output * alpha # 若不需要α(缩放因子)，注释掉即可
        else:
            output = input
        return output


# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class Conv2d_Q(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            A=2,
            W=2
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        # self.activation_quantizer = activation_bin(A=A)
        self.weight_quantizer = weight_tnn_bin(W=W)
        self.mask_flag = False

    def forward(self, input):
        # 量化A和W
        # bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)
        # print(bin_input)
        if self.mask_flag:
            tnn_bin_weight = tnn_bin_weight * self.mask

        # print(tnn_bin_weight[0][0][0][:])
        # 用量化后的A和W做卷积
        output = F.conv2d(
            input=input,
            weight=tnn_bin_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return output

    def set_mask(self, mask):
        # self.mask = mask
        self.weight_quantizer.set_mask(mask)  # set_atten_mask(mask)
        # self.weight.data = self.weight.data * self.mask.data
        # self.mask_flag = True

    def get_mask(self):
        print(self.weight_quantizer.mask_flag)
        return self.weight_quantizer.atten_mask
