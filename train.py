import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from sewar import rmse, sam
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import PatchSet, load_image_pair, transform_image
from loss import GeneratorLoss
from utils import AverageMeter
import torchvision.models as models
import torch.distributed
from model import model_STF

torch.cuda.set_device(4)
def_device = torch.device('cuda:4')

def get_features(image, model):
    features = {}
    x1 = model(image[:,0:3,:,:])
    x2 = model(image[:,3:6,:,:])
    features = torch.cat((x1,x2),dim=1)    
    return features

def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE):
    cur_result = {}
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    for i in range(0,1):
        cur_date = test_dates[0]
        cur_day = int(cur_date.split('_')[1])
        if cur_day == 363:
            for ref_date in test_dates:
                ref_day = int(ref_date.split('_')[1])
                if ref_day != cur_day:
                    images = load_image_pair(opt.root_dir, cur_date, ref_date)

                    output_image = np.zeros(images[1].shape)
                    image_mask = np.ones(images[1].shape)
                    for i in range(4):
                        negtive_mask = np.where(images[i] < 0)
                        inf_mask = np.where(images[i] > 10000.)
                        image_mask[negtive_mask] = 0
                        image_mask[inf_mask] = 0

                    for i in range(len(h_index_list)):
                        for j in range(len(w_index_list)):
                            h_start = h_index_list[i]
                            w_start = w_index_list[j]

                            input_lr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_lr = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_hr = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

                            flip_num = 0
                            rotate_num0 = 0
                            rotate_num = 0
                            input_lr, im_mask = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                            ref_lr, im_mask = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                            ref_hr, im_mask = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                            input_lr = input_lr.unsqueeze(0).cuda()
                            ref_lr = ref_lr.unsqueeze(0).cuda() 
                            ref_hr = ref_hr.unsqueeze(0).cuda()
                            _,_,_,_,output = model(ref_lr, input_lr, ref_hr, def_device)
                            output = output.squeeze()

                            h_end = h_start + PATCH_SIZE
                            w_end = w_start + PATCH_SIZE
                            cur_h_start = 0
                            cur_h_end = PATCH_SIZE
                            cur_w_start = 0
                            cur_w_end = PATCH_SIZE

                            if i != 0:
                                h_start = h_start + PATCH_SIZE // 4
                                cur_h_start = PATCH_SIZE // 4

                            if i != len(h_index_list) - 1:
                                h_end = h_end - PATCH_SIZE // 4
                                cur_h_end = cur_h_end - PATCH_SIZE // 4

                            if j != 0:
                                w_start = w_start + PATCH_SIZE // 4
                                cur_w_start = PATCH_SIZE // 4

                            if j != len(w_index_list) - 1:
                                w_end = w_end - PATCH_SIZE // 4
                                cur_w_end = cur_w_end - PATCH_SIZE // 4

                            output_image[:, h_start: h_end, w_start: w_end] = \
                                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()

                    real_im = images[1] * 0.0001 * image_mask
                   
                    real_output = (output_image + 1) * 0.5 * image_mask
                    
                    for real_predict in [real_output]:
                        cur_result['rmse'] = []

                        for i in range(6):
                            cur_result['rmse'].append(rmse(real_im[i], real_predict[i]))

                        cur_result['sam'] = sam(real_im.transpose(1, 2, 0), real_predict.transpose(1, 2, 0)) * 180 / np.pi
                        print('[%s/%s] RMSE: %.4f SAM: %.4f' % (
                            cur_date, ref_date, np.mean(np.array(cur_result['rmse'])),
                             cur_result['sam']))


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE):
    train_set = PatchSet(opt.train_dir, train_dates, IMAGE_SIZE, PATCH_SIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=8, shuffle=True)
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
            down_loss.update(l_re.cpu().item())
            vgg_loss.update(l_vgg.cpu().item())

            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))
        print('[%d/%d][%d/%d] G-Loss: %.4f down-Loss: %.4f vgg-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, batches, batches, g_loss.avg, down_loss.avg, vgg_loss.avg, batch_time.avg,
        ))
        scheduler.step()
        if epoch%10==0:
            test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE)
            #torch.save(model.state_dict(), f'{save_dir}model_{epoch}.pth')

def main():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train STFMamba Models')
    parser.add_argument('--image_size', default=[2720, 3200], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='/home/data/zhaomin/A_ST_fusion/Mamba_test/swinstfm-main/datasets/data_LGC', help='Datasets root directory')
    parser.add_argument('--train_dir', default='/home/data/zhaomin/A_ST_fusion/Mamba_test/swinstfm-main/datasets/train_LGC_128', help='Datasets train directory')
    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size

    # Loading Datasets
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        cur_day = int(dir_name.split('_')[1])
        if cur_day not in [347, 363]:
            train_dates.append(dir_name)
        else:
            test_dates.append(dir_name)
    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE)

if __name__ == '__main__':
    main()


