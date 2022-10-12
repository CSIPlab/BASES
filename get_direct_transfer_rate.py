"""
This code is for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Models from torchvision 0.12.0

It calculates the direct transfer rate of single model (e.g. vgg16_bn, vgg13), to vgg19

test results:
vgg16_bn -> vgg19:  2%
vgg13 -> vgg19:     17%
"""


import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from tqdm import tqdm

from utils_bases import load_imagenet_1000, load_model, get_adv_np, get_label_loss, get_logits_probs


def main():
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", nargs="?", default=16, help="perturbation level out of 255")
    parser.add_argument("--iters", nargs="?", default=10, help="number of inner iterations: 10, 20...")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0, 1...")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--algo", nargs="?", default='pgd', help="the attack algorithm. fgm, pgd, or mim")
    parser.add_argument("--victim", nargs="?", default='vgg19', help="victim model")
    parser.add_argument("--x", nargs="?", default=3, help="times alpha by x")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss: cw, ce")
    parser.add_argument("--n_im", nargs="?", default=100, help="number of images")
    args = parser.parse_args()
    
    eps = int(args.eps)
    n_iters = int(args.iters)
    algo = args.algo
    device = f'cuda:{int(args.gpu)}'
    victim_name = args.victim
    x_alpha = int(args.x)
    loss_name = args.loss_name
    n_im = int(args.n_im)

    # load imagenet dataset, NeurIPS17 
    im_root = 'imagenet1000'
    img_paths, gt_labels, tgt_labels = load_imagenet_1000(im_root)

    # load victim model
    victim_name = 'vgg19'
    victim_model = load_model(victim_name, device=device)

    if algo == 'fgm':
        alpha = 1000
    elif algo in ['pgd', 'mim']:
        alpha = eps / n_iters
        if x_alpha > 1:
            alpha = alpha * x_alpha

    # load surrogate models
    wb_names = ['vgg16_bn', 'vgg13', 'resnet18', 'squeezenet1_1', 'googlenet', \
                    'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                    'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                    'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                    'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    for model_name in wb_names:
        print(f"load: {model_name}")
        wb = [load_model(model_name, device)]

        success_idx_list = set()
        success_idx_list_top5 = set()
        acc = 0
        for im_idx in tqdm(range(n_im)):
            im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
            gt_label = gt_labels[im_idx]
            tgt_label = tgt_labels[im_idx]

            logits, probs = get_logits_probs(im_np, wb[0])
            pred_label = logits.argmax().item()
            if pred_label == gt_label:
                acc += 1
            
            w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
            adv_np, _ = get_adv_np(im_np, tgt_label, w_np, wb, eps, n_iters, alpha, loss_name=loss_name, adv_init=None)
            label_idx, _, top5 = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)

            if not args.untargeted and label_idx == tgt_label:
                success_idx_list.add(im_idx)
            if not args.untargeted and tgt_label in top5:
                success_idx_list_top5.add(im_idx)
        
        print(f"top1 acc on clean: {acc/n_im*100:.2f}%")
        print(f"transfer from {model_name} -> {victim_name}")
        print(f"top1 fooling rate: {len(success_idx_list)/n_im*100:.2f}%")
        print(f"top5 fooling rate: {len(success_idx_list_top5)/n_im*100:.2f}%")

if __name__ == '__main__':
    main()
