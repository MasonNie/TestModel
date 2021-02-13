import math
import time
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, \
    ToTensor, Resize, Grayscale

from modules import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def showimg(img, fake_img, subrate=0.1):
    # fake_img = getimg("resImages/babytmpc22hn1_5.png")
    # plt.axis("off")
    # subrate = 0.2
    img = img.convert('L')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    # plt.axis('off')
    plt.imshow(fake_img, cmap='gray')
    # plt.savefig("result_images" + '/butterfly_{}.jpg'.format(subrate), bbox_inches='tight')
    # print("image saved as butterfly_{}.jpg".format(subrate))
    plt.show()


class test_dataset_for_folder(Dataset):
    def __init__(self, dataset_dir, crop_size=256):
        super(test_dataset_for_folder, self).__init__()
        # self.blocksize = blocksize
        self.high_res_length = crop_size
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.test_compose = Compose([
            # CenterCrop(crop_size),
            Grayscale(),
            ToTensor(),
            Resize(crop_size),
            # transforms.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])

        hr_image = self.test_compose(hr_image)

        return hr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def up2ssim255(y):
    y = y * 255.
    y[y < 0] = 0
    y[y > 255.] = 255
    return y

def image_in_model_test(data_path, model_path, crop_size=256, cuda=False, ):
    image_filenames = [join(data_path, x) for x in listdir(data_path) if is_image_file(x)]
    test_compose = Compose([
        # CenterCrop(crop_size),
        Grayscale(),
        # Resize((crop_size, crop_size)),
        ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    model = torch.load(model_path, map_location=torch.device('cpu'))["model"]  #
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    #  print(low_res_sample.size(0))
    total_elapsed_time = 0
    avg_psnr_predicted = 0
    avg_ssim_val = 0
    for image_name in image_filenames:
        origin_image = Image.open(image_name)
        origin_image_tensor = test_compose(origin_image)
        x = origin_image_tensor.view(1, -1, origin_image_tensor.shape[1], origin_image_tensor.shape[2])
        if cuda:
            x = x.cuda()

        start_time = time.time()
        fake_image, _ = model(x)
        elapsed_time = time.time() - start_time
        total_elapsed_time += elapsed_time

        from torchvision import transforms

        unloader = transforms.ToPILImage()
        output_image = fake_image.cpu().clone()  # clone the tensor
        output_image = output_image.squeeze(0)  # remove the fake batch dimension
        output_image = unloader(output_image)
        origin_image_gray = unloader(origin_image_tensor)
        # a,b = np.array(input_image),np.array(output_image)
        showimg(origin_image_gray, output_image, 1)
        X = x.cpu()
        X = up2ssim255(X)
        Y = fake_image.cpu()
        Y = up2ssim255(Y)

        psnr_predicted = PSNR(np.array(origin_image_gray), np.array(output_image), shave_border=0)
        ssim_val = ssim(X, Y, data_range=255, size_average=True)  # return (N,)
        # ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)
        # print(ssim_val.item())

        print(+ psnr_predicted, + ssim_val.item())
        avg_psnr_predicted += psnr_predicted
        avg_ssim_val += ssim_val.item()

    print("Dataset=", data_path)
    print("PSNR_predicted=", avg_psnr_predicted / len(image_filenames), avg_ssim_val / len(image_filenames))
    print("It takes {}s for processing".format(total_elapsed_time))


image_in_model_test('./dataSet/Test/Test5/',
                    "./models/0.1/f_ten_db3_10_28.03.pth")