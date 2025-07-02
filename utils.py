import argparse
import glob
import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import torchvision.transforms as T_func

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='/home/data')
    parser.add_argument('--data_name', type=str, default='rain100L')
    parser.add_argument('--save_path', type=str, default='result_gauss')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8],
                        help='number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[48, 96, 192, 384],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=30000, help='iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[16, 10, 8, 4, 2, 2])

    parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 160, 192, 256, 320, 384],
                        help='patch size of each image for progressive learning')
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[9200, 15600, 20400, 24000, 27600],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (-1 for no manual seed)')
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')
    parser.add_argument('--task_type', type=str, default='denoising', 
                        help="Select task type: 'denoising' for Gaussian denoising or 'rain' for rain removal")
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo', 'mpi'], help='Distributed backend')

    return init_args(parser.parse_args())

def get_default_args():
    class Args:
        data_path = '/home/data'
        data_name = 'rain100L'
        save_path = 'result_gauss'
        num_blocks = [4, 6, 6, 8]
        num_heads = [1, 2, 4, 8]
        channels = [48, 96, 192, 384]
        expansion_factor = 2.66
        num_refinement = 4
        num_iter = 30000
        batch_size = [16, 10, 8, 4, 2, 2]
        patch_size = [128, 160, 192, 256, 320, 384]
        lr = 0.0003
        milestone = [9200, 15600, 20400, 24000, 27600]
        workers = 8
        seed = -1
        model_file = None
        task_type = 'denoising'
    return init_args(Args())

class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.task_type = args.task_type
        self.seed=args.seed
        # self.backend=args.backend


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)





def pad_image_needed(img, size):
    
    # Chuyển từ tensor sang numpy array
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).numpy()  # Chuyển từ [C,H,W] sang [H,W,C]
    
    h, w = img.shape[:2]
    h_pad = max(0, size[0] - h)
    w_pad = max(0, size[1] - w)
    
    if h_pad == 0 and w_pad == 0:
        return torch.from_numpy(img).permute(2, 0, 1)  # Chuyển lại về tensor [C,H,W]
    
    # Pad ảnh
    img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    
    # Xử lý ảnh grayscale
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    
    # Chuyển lại về tensor
    return torch.from_numpy(img).permute(2, 0, 1)  # Chuyển từ [H,W,C] sang [C,H,W]

def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)

class GaussianDenoisingDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, patch_size=None, length=None, sigma_type='random', sigma_range=(0, 50), use_y_channel=False):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        if self.data_type == 'train':
            self.gt_images = sorted(glob.glob('{}/{}/{}/DFWB/*.*'.format(data_path, data_name, data_type)))
        if self.data_type == 'test':
            self.gt_images = sorted(glob.glob('{}/{}/{}/CBSD68/*.*'.format(data_path, data_name, data_type)))

        self.num = len(self.gt_images)
        self.sample_num = length if data_type == 'train' else self.num
        self.sigma_type = sigma_type
        self.sigma_range = sigma_range
        self.use_y_channel = use_y_channel

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        # Load Ground Truth image
        gt_image = Image.open(self.gt_images[idx % self.num])
        gt = T.to_tensor(gt_image)
        h, w = gt.shape[1:]

        if self.data_type == 'train':
            # Ensure image is large enough for cropping
            gt = pad_image_needed(gt, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(gt, (self.patch_size, self.patch_size))
            gt = T.crop(gt, i, j, th, tw)

            # Convert to tensor first before adding noise
            lq = gt.clone()

            # Add Gaussian noise
            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            noise = torch.randn_like(lq).mul_(noise_level)
            lq.add_(noise)
            lq.clamp_(0, 1)  # Ensure values stay in [0,1] range

            # Augmentations (flip, rotate, etc.)
            if random.random() < 0.5:
                gt = T.hflip(gt)
                lq = T.hflip(lq)
            if random.random() < 0.5:
                gt = T.vflip(gt)
                lq = T.vflip(lq)
        else:
            # For validation/test, add noise and pad image
            new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            pad_h = new_h - h if h % 8 != 0 else 0
            pad_w = new_w - w if w % 8 != 0 else 0
            gt = F.pad(gt, (0, pad_w, 0, pad_h), 'reflect')

            # Convert to tensor first before adding noise
            lq = gt.clone()
            
            # Add noise for testing
            noise_level = torch.FloatTensor([self.sigma_range[0]])/255.0
            noise = torch.randn_like(lq).mul_(noise_level)
            lq.add_(noise)
            lq.clamp_(0, 1)  # Ensure values stay in [0,1] range

        # Convert to Y channel if requested
        if self.use_y_channel:
            gt = rgb_to_y(gt)
            lq = rgb_to_y(lq)

        # Extract the image name for future use
        image_name = os.path.basename(self.gt_images[idx % self.num])

        return lq, gt, image_name, h, w

class RainDataset(Dataset):
    def __init__(self, task, data_path, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        
        if task == 'single_image_deblur':
            if data_type == 'train':
                self.rain_images = sorted(glob.glob('{}/{}/{}/DPDD/inputC_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/{}/DPDD/target_crops/*.*'.format(data_path, data_name, data_type)))
            else:
                self.rain_images = sorted(glob.glob('{}/{}/val/DPDD/inputC_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/val/DPDD/target_crops/*.*'.format(data_path, data_name, data_type)))
        
        elif task == 'motion_deblur':
            if data_type == 'train':
                self.rain_images = sorted(glob.glob('{}/{}/{}/GoPro/input_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/{}/GoPro/target_crops/*.*'.format(data_path, data_name, data_type)))
            else:
                self.rain_images = sorted(glob.glob('{}/{}/val/GoPro/input_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/val/GoPro/target_crops/*.*'.format(data_path, data_name, data_type)))
        
        elif task == 'derain':
            if data_type == 'train':
                self.rain_images = sorted(glob.glob('{}/{}/{}/Rain13K/input/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/{}/Rain13K/target/*.*'.format(data_path, data_name, data_type)))
            else:
                self.rain_images = sorted(glob.glob('{}/{}/{}/Rain100L/input/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/{}/Rain100L/target/*.*'.format(data_path, data_name, data_type)))
        
        elif task == 'real_denoise':
            if data_type == 'train':
                self.rain_images = sorted(glob.glob('{}/{}/{}/SIDD/input_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/{}/SIDD/target_crops/*.*'.format(data_path, data_name, data_type)))
            else:
                self.rain_images = sorted(glob.glob('{}/{}/val/SIDD/input_crops/*.*'.format(data_path, data_name, data_type)))
                self.norain_images = sorted(glob.glob('{}/{}/val/SIDD/target_crops/*.*'.format(data_path, data_name, data_type)))

        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]))
        norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
        h, w = rain.shape[1:]

        if self.data_type == 'train':
            # Ensure image is large enough for cropping
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            
            # Data augmentation
            if random.random() < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if random.random() < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        else:
            # For validation/test, pad image to multiple of 8
            new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            pad_h = new_h - h if h % 8 != 0 else 0
            pad_w = new_w - w if w % 8 != 0 else 0
            rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
            norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')
            
        return rain, norain, image_name, h, w


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()
