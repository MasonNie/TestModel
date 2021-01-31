import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import time, math, glob
import scipy.io as sio
from PIL import Image
import torch
from torch import nn
from torch.autograd import Variable
from pytorch_msssim import ssim
from modules import *


def showimg(img, fake_img, subrate=0.1):
    #fake_img = getimg("resImages/babytmpc22hn1_5.png")
    # plt.axis("off")
    # subrate = 0.2
    img = img.convert('L')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    # plt.axis('off')
    plt.imshow(fake_img,cmap='gray')
    # plt.savefig("result_images" + '/butterfly_{}.jpg'.format(subrate), bbox_inches='tight')
    # print("image saved as butterfly_{}.jpg".format(subrate))
    plt.show()


parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args([])

opt.cuda = False
opt.model =  "./models/0.5/full_db3_10_38.64.pth" #"CSNetPlus_model_t/model_epoch_435_17.98640433730782.pth" #
# opt.model = "CSNetPlus_model_t/0.2_denseBlock3*10.pth"#0.2_888.6.pth"#-1+1_noPurning_depthwise_0.2.pth"  #loss:600
# opt.model = "saved_models/0.2/rrdb4_32.45.pth"
subrate = 0.5

cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model,map_location=torch.device('cpu'))["model"]
model.eval()
image_list = glob.glob("dataSet/Test/" + opt.dataset + "_mat/*.*") 

avg_psnr_predicted = 0.0
avg_elapsed_time = 0.0
avg_ssim_val = 0.0

pos = 0
for image_name in image_list:
    # pos += 1
    # if pos != 1:
    #     continue
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)['im_gt_y']

    im_gt_y = im_gt_y.astype(float)
    X = Variable(torch.from_numpy(im_gt_y).float()).view(1, -1, im_gt_y.shape[0], im_gt_y.shape[1])
    im_input = im_gt_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])


    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    res, _ = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    from torchvision import transforms

    unloader = transforms.ToPILImage()
    image = res.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    showimg(Image.fromarray(im_gt_y), image, subrate)

    res = res.cpu()
    im_res_y = res.data[0].numpy().astype(np.float32)

    im_res_y = im_res_y*255.
    im_res_y[im_res_y<0] = 0
    im_res_y[im_res_y>255.] = 255. 
    im_res_y = im_res_y[0,:,:]
    Y = res*255.
    Y[Y<0] = 0
    Y[Y>255.] = 255

    psnr_predicted = PSNR(im_gt_y, im_res_y,shave_border=0)
    ssim_val = ssim( X, Y, data_range=255, size_average=True) # return (N,)
    # ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)
    # print(ssim_val.item())
    
    print( + psnr_predicted, + ssim_val.item())
    avg_psnr_predicted += psnr_predicted
    avg_ssim_val += ssim_val.item()



print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted/len(image_list), avg_ssim_val/len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))