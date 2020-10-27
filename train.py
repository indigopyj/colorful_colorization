# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataloader
from skimage import color
from util import *
from model import *
from modules.cross_entropy_loss_2d import *


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Parameter
    lr = args.lr
    lr_update_step = args.lr_update_step
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    use_gpu = args.use_gpu


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
    load_opt = args.load_opt

    mode = args.mode
    train_continue = args.train_continue

    # task = args.task
    # opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    # nker = args.nker

    network = args.network

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    # log_dir = args.log_dir + "/" + network + date
    # result_dir = args.result_dir + "/" + network + date

    #    createFolder(result_dir)

    # 로스함수 정의
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    fn_loss = nn.CrossEntropyLoss().to(device)

    if network == "resnet":
        # net = ResNet(in_channels=nch, out_channels=nch, norm="bnorm").to(device)
        # net = torchvision.models.resnet50(pretrained=True)
        net = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        # for param in net.parameters():
        #    param.requires_grad=False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net = net.to(device)
        params = net.parameters()
    elif network == "mobilenet":
        # net = torchvision.models.mobilenet_v2(pretrained=True)
        # net.classifier[1] = nn.Linear(net.last_channel, 2)
        # net = net.to(device)
        # net = mobilenet_v2(pretrained=True, input_size=11).to(device)
        # net.classifier = nn.Linear(net.last_channel, 2)
        # net = net.to(device)
        net = mobilenetv2(width_mult=0.5)
        net.load_state_dict(torch.load('./mobilenetv2/pretrained/mobilenetv2_0.5-eaa6f9ad.pth'))
        net.classifier = nn.Linear(net.output_channel, 2)
        net = net.to(device)
        params = net.parameters()
    elif network == "efficientnet":
        net = EfficientNet.from_pretrained('efficientnet-b0')
        net._fc = nn.Linear(in_features=net._fc.in_features, out_features=2, bias=True)
        params = net.parameters()
        net = net.to(device)

    # optimizer 정의
    optim = torch.optim.Adam(params, lr=lr)

    if mode == "test":
        transform = transforms.Compose([transforms.Resize([ny, nx]), transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        dataset_test = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
        # print(dataset_test.classes)
        # 부수적인 변수들 정의
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    if mode == "test":
        ############# Test ################

        #        createFolder(os.path.join(result_dir, "female"))
        #       createFolder(os.path.join(result_dir, "male"))

        net, optim, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim, load_opt=load_opt)
        net = net.to(device)

        with torch.no_grad():
            net.eval()
            avg_loss = []
            avg_acc = []

            loss_meter_test = tnt.meter.AverageValueMeter()
            acc_meter_test = tnt.meter.ClassErrorMeter(accuracy=True)

            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input, label = data
                label = label.to(device)
                input = input.to(device)

                output = net(input)

                loss = fn_loss(output.squeeze(), label)
                loss_meter_test.add(loss.item())
                acc_meter_test.add(output.data.cpu().numpy(), label.cpu().numpy().squeeze())

                avg_loss += [loss_meter_test.value()[0]]
                avg_acc += [acc_meter_test.value()[0]]

                print("TEST :  BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" %
                      (batch, num_batch_test, loss_meter_test.value()[0], acc_meter_test.value()[0]))

        print("\nAVERAGE TEST PERFORMANCE: LOSS %.4f | ACCURACY %.4f" %
              (np.mean(avg_loss), np.mean(avg_acc)))

