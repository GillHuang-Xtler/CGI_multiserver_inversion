import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from utils.data_processing import set_idx, label_mapping
import numpy as np
from .game import NashMSFL
from utils.data_processing import total_variation
from torchmetrics.image import TotalVariation
from utils.save import save_img, early_stop, save_final_img, save_eval
from torch.nn.utils import clip_grad_norm_
from .method_utils import gradient_closure, gradient_closure2
from .method_utils import defense_alg


def mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    for imidx in range(num_dummy):

        # get random idx or
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)

        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )

        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    d_mean, d_std = mean_std
    dm = torch.as_tensor(d_mean, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()
    ds = torch.as_tensor(d_std, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()

    original_dy_dxs = []
    _label_preds = []

    for i in range(args.num_servers):
        if args.defense_method == 'none':
            out = nets[i](gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, nets[i].parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        else:
            original_dy_dx = defense_alg(nets[i], gt_data, gt_label, criterion, device, args)

        original_dy_dxs.append(original_dy_dx)

        # predict the ground-truth label
        _label_preds.append(torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape((1,)).requires_grad_(False))

    label_preds = []
    for i in _label_preds:
        j = i.repeat(args.num_dummy)
        label_preds.append(j)

    # initialize random image
    dummy_data = torch.randn(gt_data.size(), dtype=next(nets[0].parameters()).dtype).to(device).requires_grad_(True)

    with torch.no_grad():
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    if args.num_dummy > 1:
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 12.0], gamma=0.0001)
        else:
            optimizer = optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 1.5], gamma=0.1)
    else:
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 12.0],
                                                             gamma=0.0001)
        else:
            optimizer = torch.optim.Adam([dummy_data, ], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 1.5], gamma=0.1)

    history = []
    history_iters = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            # new
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = nets[i](dummy_data)
                if args.num_dummy > 1:
                    dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=1))
                else:
                    dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

            grad_diff = 0
            for i in range(args.num_servers):
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    if args.inv_loss == 'l1':
                        grad_diff += (torch.abs(gx - gy)).sum()
                    else:
                        grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        if args.inv_loss == 'l2' or args.inv_loss == 'l1':
            current_loss = optimizer.step(closure)
        else:  # 'sim'
            current_loss = optimizer.step(gradient_closure2(optimizer, dummy_data, original_dy_dxs,
                                                        label_preds, nets, args, criterion))
        if args.scheduler:
            scheduler.step()

        with torch.no_grad():
            # Project into image space
            dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if (iters + 1 == args.iteration) or iters % 500 == 0:
                print(f'It: {iters}. Rec. loss: {current_loss.item():2.4f}.')

        train_iters.append(iters)
        if iters % args.log_metrics_interval == 0 or iters in [10 * i for i in range(10)]:
            result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
            res = [iters]
            res.extend(result)
            results.append(res)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0 or (args.log_interval == 1 and iters == Iteration-1):
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results

def mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    tmp_labels = []
    gt_labels = []

    d_mean, d_std = mean_std
    dm = torch.as_tensor(d_mean, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()
    ds = torch.as_tensor(d_std, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()

    for imidx in range(num_dummy):
        tmp_labels = []
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_labels.append(torch.Tensor([dst[idx][1]]).long().to(device))

        '''get new mapping label for the same data sample'''
        if args.num_servers > 1:
            for i in range(args.num_servers-1):
                tmp_labels.append(label_mapping(origin_label = dst[idx][1], idx = i).to(device))

        for i in range(args.num_servers):
            tmp_labels[i] = tmp_labels[i].view(1, )

        if imidx == 0:
            gt_data = tmp_datum
            for i in range(args.num_servers):
                gt_labels.append(tmp_labels[i])
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            for i in range(args.num_servers):
                gt_labels[i] = torch.cat((gt_labels[i], tmp_labels[i]), dim=0)

    original_dy_dxs = []
    label_preds = []
    for i in range(args.num_servers):
        if args.defense_method == 'none':
            out = nets[i](gt_data)
            y = criterion(out, gt_labels[i])
            dy_dx = torch.autograd.grad(y, nets[i].parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        else:
            original_dy_dx = defense_alg(nets[i], gt_data, gt_labels[i], criterion, device, args)

        original_dy_dxs.append(original_dy_dx)

        # predict the ground-truth label
        label_preds.append(torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape((1,)).requires_grad_(False))
        label_preds[i] = label_preds[i].repeat(args.num_dummy)
    dummy_data = torch.randn(gt_data.size(), dtype=next(nets[0].parameters()).dtype).to(device).requires_grad_(True)

    with torch.no_grad():
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

    if args.optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)
    else:
        optimizer = torch.optim.Adam([dummy_data, ], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[args.iteration // 2.0], gamma=0.1)

    history = []
    history_iters = []
    # losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            single_alpha = torch.FloatTensor([0,1])
            _ = torch.rand(1)
            random_alpha = torch.FloatTensor([_, 1 - _])

            # new
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = (nets[i](dummy_data))
                dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))
            losses = []
            for i in range(args.num_servers):
                _loss = 0
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    _loss += ((gx - gy) ** 2).sum()
                losses.append(_loss)

            if args.diff_task_agg == 'game':
                game = NashMSFL(n_tasks=args.num_servers)
                _, _, game_alpha = game.get_weighted_loss(losses=losses, dummy_data=dummy_data)
                game_alpha = [game_alpha[i] / sum(game_alpha) for i in range(len(game_alpha))]
                grad_diff = sum([losses[i] * game_alpha[i] for i in range(len(game_alpha))])
            elif args.diff_task_agg == 'single':
                grad_diff = sum([losses[i] * single_alpha[i] for i in range(len(single_alpha))])
            elif args.diff_task_agg == 'random':
                grad_diff = sum([losses[i] * random_alpha[i] for i in range(len(random_alpha))])

            grad_diff.backward()
            return grad_diff

        if args.inv_loss == 'l2':
            current_loss = optimizer.step(closure)
        else:  # 'sim'
            current_loss = optimizer.step(gradient_closure2(optimizer, dummy_data, original_dy_dxs,
                                                        label_preds, nets, args, criterion))
        if args.scheduler:
            scheduler.step()

        with torch.no_grad():
            # Project into image space
            dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if (iters + 1 == args.iteration) or iters % 500 == 0:
                print(f'It: {iters}. Rec. loss: {current_loss.item():2.4f}.')

        train_iters.append(iters)
        # losses.append(current_loss)
        if iters % args.log_metrics_interval == 0 or iters in [10 * i for i in range(10)]:
            result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
            res = [iters]
            res.extend(result)
            results.append(res)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results