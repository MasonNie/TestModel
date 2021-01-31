import torch
import numpy as np
from torch.autograd import Variable


def atten_cal_mask(atten_matrix, origin_matrix, pruning_perc):
    print("=========>calculate mask")
    masks = []
    campare_matrix = atten_matrix.cpu() - origin_matrix.cpu()
    atten_cal_weights = list(campare_matrix.cpu().data.numpy().flatten())
    origin_weights = list(origin_matrix.cpu().data.abs().numpy().flatten())
    threshold_weight = np.percentile(np.array(origin_weights), 70)
    threshold_atten = np.percentile(np.array(atten_cal_weights), pruning_perc)
    data_len = len(atten_cal_weights)
    count = 0
    for i in range(data_len):
        prun_val = 1
        if origin_weights[i] < threshold_weight and atten_cal_weights[i] < threshold_atten:
            prun_val = 0
            count += 1
        masks.append(prun_val)
    print("true cut rate:", count / data_len)
    masks = torch.tensor(masks)
    masks = to_var(masks)
    masks = masks.view(origin_matrix.size())
    print("=========>mask calculate finish")
    return masks


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


atten_cal_mask(torch.randn((102, 32, 32)), torch.randn((102, 32, 32)), 50)
