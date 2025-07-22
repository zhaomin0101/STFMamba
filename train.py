import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import PatchSet
from loss import GeneratorLoss
from utils import AverageMeter
import torchvision.models as models
import torch.distributed
from model import model_STF

torch.cuda.set_device(0)
def_device = torch.device('cuda:0')

def get_features(image, model):
    features = {}
    x1 = model(image[:,0:3,:,:])
    x2 = model(image[:,3:6,:,:])
    features = torch.cat((x1,x2),dim=1)    
    return features


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE):
    train_set = PatchSet(opt.train_dir, train_dates, IMAGE_SIZE, PATCH_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=10, shuffle=True)
    model = model_STF().cuda()
   
    cri_pix = GeneratorLoss()
    criterion = torch.nn.MSELoss()
    cri_pix.cuda()

    trainable_params = list(model.parameters())
    optimizer = torch.optim.Adam(trainable_params, 5*1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    save_dir = 'experiment_model_pth'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vgg19 = models.vgg19(pretrained=True).features.cuda()
    vgg19.eval()
   
    for epoch in tqdm(range(opt.num_epochs)):
        model.train()
        g_loss, down_loss, vgg_loss, batch_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        batches = len(train_loader)

        for item, (data, target, ref_lr, ref_target, gt_mask) in tqdm(enumerate(train_loader)):
            t_start = timer()
            data = data.cuda()
            target = target.cuda()
            ref_lr = ref_lr.cuda()
            ref_target = ref_target.cuda()
            gt_mask = gt_mask.float().cuda() 
            SR1,SR2,x1,x2,fusion = model(ref_lr, data, ref_target, def_device)
            optimizer.zero_grad()
            l_re = cri_pix(fusion * gt_mask, target * gt_mask, is_ds=False)
            l_re_2 = cri_pix(x2 * gt_mask, target * gt_mask, is_ds=False)
            l_re_1 = cri_pix(x1 * gt_mask, target * gt_mask, is_ds=False)
            
            features_pre = get_features(fusion* gt_mask, vgg19)
            features_targrt = get_features(target* gt_mask, vgg19)
            l_vgg = criterion(features_pre, features_targrt)

            l_sr = cri_pix(SR2 * gt_mask, target * gt_mask, is_ds=False)
            + cri_pix(SR1 * gt_mask, ref_target * gt_mask, is_ds=False) 
            l_total = l_sr + l_re + (l_re_1 + l_re_2) + 1e-3*l_vgg 

            l_total.backward()
            optimizer.step()

            g_loss.update(l_re.cpu().item())
            down_loss.update(l_sr.cpu().item())
            vgg_loss.update(l_vgg.cpu().item())

            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))
        print('[%d/%d][%d/%d] G-Loss: %.4f down-Loss: %.4f vgg-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, batches, batches, g_loss.avg, down_loss.avg, vgg_loss.avg, batch_time.avg,
        ))
        scheduler.step()
        if epoch%5==0:
            torch.save(model.state_dict(), f'{save_dir}model_{epoch}.pth')

def main():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train STFMamba Models')
    parser.add_argument('--image_size', default=[1000, 1000], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='', help='Datasets root directory')
    parser.add_argument('--train_dir', default='', help='Datasets train directory')
    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size

    train_dates = []
    test_dates = []
    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE)

if __name__ == '__main__':
    main()


