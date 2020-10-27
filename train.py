# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataloader
import torch.nn.functional as F
from util import *
from model import *
from modules.cross_entropy_loss_2d import *
import matplotlib.pyplot as plt


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Parameter
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    mode = args.mode
    train_continue = args.train_continue


    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir


    net = ColorizationNetwork(device=device)
    net = net.to(device)

    # 로스함수 정의
    fn_loss = CrossEntropyLoss2d().to(device)

    # optimizer 정의
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.99))

    createFolder(os.path.join(log_dir, 'train'))
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    if mode == 'train':
        loader_train = get_dataloader(dataset="yumi", phase="train", batch_size=batch_size, processed_dir=".")
        data_iter = iter(loader_train)
        if train_continue == "on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        else:
            st_epoch = 0  # start epoch number

        net.train()
        for epoch in range(st_epoch + 1, num_epoch + 1):
            try:
                sketch_img, color_img = next(data_iter)
            except:
                data_iter = iter(loader_train)
                sketch_img, color_img = next(data_iter)

            sketch_img = sketch_img.to(device)
            color_img = color_img.to(device)
            color_img = rgb_to_lab(color_img.permute(0,2,3,1))
            color_img = torch.from_numpy(color_img).permute(0,3,1,2)
            q_pred, q_actual = net(sketch_img, color_img)

            assert False
            optim.zero_grad()
            loss = fn_loss(q_pred, q_actual)
            loss.backward()
            optim.step()

            # if lr_scheduler is not None:
            #     lr_scheduler.step()


            # lr = optim.param_groups[0]['lr']
            writer_train.add_scalar('loss', loss, epoch)
            print('training %d iters, loss is %.4f' % (epoch, loss))

            # Learning rate decay
            if (epoch+1) == 200000:
                lr = 1e-5
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                print ('Decayed learning rates, lr: %4f' % (lr))
            elif (epoch+1) == 375000:
                lr = 3e-6
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                print('Decayed learning rates, lr: %4f' % (lr))



            if epoch % 1 == 0 or epoch == num_epoch:
                createFolder(ckpt_dir)
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        writer_train.close()


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Parameter
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    mode = args.mode
    train_continue = args.train_continue

    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir


    createFolder(result_dir)

    net = ColorizationNetwork(device=device)
    net = net.to(device)

    # 로스함수 정의
    fn_loss = CrossEntropyLoss2d().to(device)

    # optimizer 정의
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.99))


    if mode == "test":
        loader_test = get_dataloader(dataset="yumi", phase="test", batch_size=batch_size, processed_dir=".")

        net, optim, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        net = net.to(device)

        with torch.no_grad():
            net.eval()

            for batch_idx, data in enumerate(loader_test):
                # forward pass
                sketch_img, color_img = data
                sketch_img = sketch_img.to(device)

                prediction = net(sketch_img, color_img).cpu()
                if(sketch_img.shape[2] != prediction.shape[2] or sketch_img.shape[3] != prediction.shape[3]):
                    pred_resize = F.interpolate(prediction, size=sketch_img.shape[2:], mode='bilinear', align_corners=False)

                pred_lab = torch.cat((sketch_img, pred_resize), dim=1)
                pred_rgb = lab_to_rgb(pred_lab.cpu().permute(0,2,3,1))

                for index,img in enumerate(pred_rgb):
                    img_num = batch_idx * batch_size + index + 1
                    img = (255*np.clip(img, 0, 1)).astype('uint8')
                    new_image_path = os.path.join(result_dir, str(img_num)+".png")
                    plt.imsave(new_image_path, img)

                # calculate PSNR
                pred_rgb = torch.from_numpy(pred_rgb).permute(0,3,1,2)
                mse = torch.mean((pred_rgb - color_img) ** 2)
                PSNR = 10 * torch.log10(1.0 / mse)

                print("%03d.png :  PSNR %.4f" %
                      (img_num + 1, PSNR))





