"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Attack blackbox victim model via querying weight space of ensemble models. 

Whitebox setting
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from class_names_imagenet import lab_dict as imagenet_names
from utils_bases import load_imagenet_1000, load_model
from utils_bases import normalize, relu, get_loss_fn, get_adv


def main():
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", nargs="?", default=16, help="perturbation level in a scale of 0-255")
    parser.add_argument("--iters", nargs="?", default=10, help="number of inner iterations: 5,10,20...")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    parser.add_argument("--algo", nargs="?", default='pgd', help="the attack algorithm. fgm, pgd, or mim")
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--victim", nargs="?", default='vgg19', help="victim model")
    parser.add_argument("--x", nargs="?", default=3, help="multiply step-size by x")
    parser.add_argument("--n_wb", nargs="?", default=10, help="number of models in the ensemble")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss: cw, ce")
    parser.add_argument("--lr", nargs="?", default=1e-5, help="learning rate of w")
    parser.add_argument("--iterw", nargs="?", default=50, help="iterations of updating w")
    parser.add_argument("--l2", nargs="?", default=0, help="l2 bound")
    parser.add_argument("--n_im", nargs="?", default=1000, help="number of images")
    args = parser.parse_args()
    

    eps = int(args.eps)
    n_iters = int(args.iters)
    algo = args.algo
    fuse = args.fuse
    device = f'cuda:{int(args.gpu)}'
    victim_name = args.victim
    x_alpha = int(args.x)
    n_wb = int(args.n_wb)
    loss_name = args.loss_name
    lr_w = float(args.lr)
    iterw = int(args.iterw)
    l2_bound = float(args.l2)
    n_im = int(args.n_im)
    loss_fn = get_loss_fn(loss_name, targeted = not args.untargeted)
    

    # load surrogate models
    surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    wb = []
    for model_name in surrogate_names[:n_wb]:
        print(f"load: {model_name}")
        wb.append(load_model(model_name, device))

    # load victim model
    victim_model = load_model(victim_name,device=device)
    # load images
    img_paths, gt_labels, tgt_labels = load_imagenet_1000(dataset_root='imagenet1000')

    # create folders
    if l2_bound > 0:
        exp = f'{n_wb}wb_fuse_{fuse}_algo_{algo}_l2_{l2_bound}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_loss_{loss_name}_lr{lr_w}_iterw{iterw}'
    else:
        exp = f'{n_wb}wb_fuse_{fuse}_algo_{algo}_eps_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_loss_{loss_name}_lr{lr_w}_iterw{iterw}'
    if args.untargeted:
        exp = 'untargeted_' + exp


    exp_root = Path(f"learned_w_logs/") / exp
    exp_root.mkdir(parents=True, exist_ok=True)
    print(exp)
    adv_root = Path(f"learned_w_adv_images/") / exp
    adv_root.mkdir(parents=True, exist_ok=True)

    if algo == 'fgm':
        alpha = 1000
    elif algo in ['fgsm', 'mi-fgsm']:
        alpha = eps / n_iters
        if x_alpha > 1:
            alpha = alpha * x_alpha


    success_idx_list = set()
    for im_idx in tqdm(range(n_im)):

        im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
        im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
        gt_label = gt_labels[im_idx]
        gt_label_name = imagenet_names[gt_label].split(',')[0]
        tgt_label = tgt_labels[im_idx]
        if args.untargeted:
            tgt_label = gt_label
        target = torch.LongTensor([tgt_label]).to(device)
        exp_name = f"idx{im_idx}_f{gt_label}_t{tgt_label}"
        
        # initial W, lr_w
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        w = torch.from_numpy(w_np).to(device)
        adv = torch.clone(im) # adversarial image

        w_list = []
        loss_wb_list = []
        loss_list = []
        adv_label_list = []
        for iter_w in range(iterw):
            w.requires_grad=True

            # lower-level, generate perturbation
            adv, loss_wb = get_adv(im, adv, target, w, ensemble=wb, eps=eps, n_iters=n_iters, alpha=alpha, algo=algo, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name)

            # upper-level, update w
            output = victim_model(normalize(adv/255))
            loss = loss_fn(output,target)
            
            # record info to txt
            w_list.append(w.tolist())
            loss_list.append(loss.item())
            victim_label_idx = output.argmax().item()
            adv_label_list.append(victim_label_idx)
            loss_wb_list += loss_wb

            # record successful index
            if (not args.untargeted and victim_label_idx == tgt_label) or (args.untargeted and victim_label_idx != tgt_label):
                success_idx_list.add(im_idx)

            # save adv in folder
            adv_np = adv.squeeze().cpu().numpy().transpose(1, 2, 0)
            adv_path = adv_root / f"{img_paths[im_idx].stem}_iter{iter_w:02d}.png"
            adv_png = Image.fromarray(adv_np.astype(np.uint8))
            adv_png.save(adv_path)

            info = ""
            info += f"n_iters: {n_iters}, iter_w: {iter_w}, loss: {loss.item()}\n"
            info += f"w: {w.tolist()}\n"
            info += f"label victim: {victim_label_idx, imagenet_names[victim_label_idx]}\n"
            for idx_model, model in enumerate(wb):
                out = model(normalize(adv/255))
                wb_label_idx = out.argmax().item()
                info += f"label wb {idx_model}: {wb_label_idx, imagenet_names[wb_label_idx]}\n"
            # save to txt
            file = open(exp_root / f'{exp_name}.txt', 'a')
            file.write(f"{info}\n\n")
            file.close()

            loss.backward()
            with torch.no_grad():
                w -= lr_w * w.grad
                w = relu(w)
                w = w / w.sum()

        print(f"im_idx: {im_idx}; total_success: {len(success_idx_list)}")
        

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_wb_list)
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        ax[1].imshow(im_np)
        ax[1].set_title(f"clean image:\n{gt_label_name}")
        adv_np = adv.cpu().numpy().squeeze().transpose(1,2,0)
        adv_label_name = imagenet_names[victim_label_idx].split(',')[0]
        ax[2].imshow(adv_np/255)
        ax[2].set_title(f"adv image:\n{adv_label_name}")
        ax[3].plot(loss_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        # plot circles on true adversarial W
        for idx, label in enumerate(adv_label_list):
            if (not args.untargeted and label == tgt_label) or (args.untargeted and label != tgt_label):
                ax[3].scatter(idx, loss_list[idx], s=10, c='red', marker='o')
        ax[4].plot(w_list)
        ax[4].legend(surrogate_names, shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        # plt.show()
        plt.tight_layout()
        plt.savefig(exp_root / f"{exp_name}.png")
        plt.close()


if __name__ == '__main__':
    main()
