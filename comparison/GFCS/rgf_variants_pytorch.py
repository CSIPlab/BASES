"""
This code is used to produce P-RGF comparison with
NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Code modified for new datasets and models
"""

# This is a PyTorch port of the main P-RGF attack file originally released and documented in this repo:
#   https://github.com/thu-ml/Prior-Guided-RGF. This implements the method described in "Improving Black-box
#   Adversarial Attacks with a Transfer-based Prior" (https://arxiv.org/abs/1906.06919). Note that this port does not
#   necessarily cover every code path of the original, only the portions required for the relevant experimental
#   comparisons.
# Additionally, we have added a '--method ods' option that implements the ODS-RGF method of "Diversity can be
#   transferred: Output diversification for white-and black-box attacks" (https://arxiv.org/abs/2003.06878). There is
#   otherwise no public implementation of ODS-RGF as of this writing; the implementation of the method was confirmed
#   through correspondence with the first author of that paper.

# This only implements the untargeted attack variant, as with the original.

# Note that reference PyTorch versions of certain networks, including InceptionV3, differ from TensorFlow versions in
#   that they output 1000 rather than 1001 (including "background") classes. Certain portions of code have been omitted
#   accordingly.

# Note also that this code is only intended to function on ImageNet as written.

# Be aware that in the original code, the value of max_queries is not strictly enforced as it can be exceeded in the
#   final run of the outer iteration, and will in that case be counted as a success by this code.
# This means that results logged as in the original code must be pruned for these cases to align with the results
#   reported for other methods.

import argparse
from argparse import Namespace
import random
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
# import torchvision.datasets as datasets

import gfcs_util
# import eval_sets
# load module from BASES
sys.path.append('../../')
from utils_bases import load_imagenet_1000


@torch.no_grad()
def p_rgf_pytorch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Target model to use.')
    # parser.add_argument('--model_name', required=True, choices=['inception_v3', 'vgg16_bn', 'resnet50'],
    #                     help='Model to be attacked. Assumed to be a torchvision net at present.')
    # parser.add_argument('--smodel_name', type=str, nargs='+',
    #                     help='One or more surrogate models to use for applicable methods. At least one must be '
    #                          'provided. All model names are currently assumed to refer to torchvision nets.')
    # parser.add_argument('--output', required=True, help='Output filename. Will be under output folder.')
    # parser.add_argument('--output_dir', default='experimental_results', help='Output folder.')
    parser.add_argument('--norm', choices=['l2', 'linfty'], default='l2', help='The norm used in the attack')
    parser.add_argument('--norm_bound', type=float, default=float('inf'),
                        help='Radius of norm ball onto which solution will be maintained through PGD-type '
                             'optimisation. If not supplied, is effectively infinite (norm is unconstrained).')
    parser.add_argument('--step_size', default=0.2, type=float, help='step size per iteration')
    parser.add_argument('--method', choices=['uniform', 'biased', 'average', 'fixed_biased', 'fixed_average', 'ods'],
                        default='biased', help='Methods used in the attack.')
    parser.add_argument('--fixed_const', type=float, default=0.5,
                        help='Value of lambda used in fixed_biased, or value of mu used in fixed_average')
    parser.add_argument('--use_larger_step_size', action='store_true',
                        help='Determines the value of initial sigma. In the reference code, the value of this option '
                             'depends on the selected model.')
    parser.add_argument('--dataprior', action='store_true', help='Whether to use data prior in the attack.')
    parser.add_argument('--show_true', action='store_true', help='Whether to print statistics about the true gradient.')
    parser.add_argument('--show_loss', action='store_true', help='Whether to print loss in some given step sizes.')
    parser.add_argument('--data_index_set', type=str,
                        choices=['vgg16_bn_mstr', 'vgg16_bn_batch0', 'vgg16_bn_batch1',
                                 'vgg16_bn_batch2', 'vgg16_bn_batch3', 'vgg16_bn_batch4', 'vgg16_bn_batch0_2',
                                 'resnet50_mstr', 'resnet50_batch0', 'resnet50_batch1',
                                 'resnet50_batch2', 'resnet50_batch3', 'resnet50_batch4', 'resnet50_batch0_2',
                                 'inceptionv3_mstr', 'inceptionv3_batch0', 'inceptionv3_batch1',
                                 'inceptionv3_batch2', 'inceptionv3_batch3', 'inceptionv3_batch4',
                                 'inceptionv3_batch0_2', 'imagenet_val_random'],
                        default='imagenet_val_random',
                        help='The indices from the ImageNet val set to use as inputs. Most options represent '
                             'predefined randomly sampled batches. imagenet_val_random samples from the val set '
                             'randomly, and may not necessarily give images that are correctly classified by the '
                             'target net.')
    parser.add_argument('--samples_per_draw', type=int, default=50, help='Number of samples to estimate the gradient.')
    parser.add_argument('--number_images', type=int, default=1000, help='Number of images for evaluation.')
    parser.add_argument('--max_queries', type=int, default=10000, help='Maximum number of queries.')
    parser.add_argument('--device', default='cuda:0', help='Device for attack.')
    parser.add_argument('--net_specific_resampling', action='store_true',
                        help='If specified, resizes input images to match expectations of target net (as always), but '
                             'adds a linear interpolation step to each surrogate network to match its expected '
                             'resolution. Gradients are thus effectively computed in the native surrogate resolutions '
                             'and returned to the target net''s own resolution via the reverse interpolation.')
    parser.add_argument("--n_wb", nargs="?", default=10, help="number of models in the ensemble")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    parser.add_argument('--targeted', action='store_true', help='If true, perform targeted attack; else, untargeted.')
    args = parser.parse_args()
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = f'cuda:{int(args.gpu)}'
    n_wb = int(args.n_wb)

    if args.dataprior and args.method == 'ods':
        sys.exit("You cannot specify dataprior and ods simultaneously.")

    target_model = getattr(models, args.model_name)(pretrained=True)
    target_model = torch.nn.Sequential(
        gfcs_util.Normalise(gfcs_util.imagenet_mean, gfcs_util.imagenet_std),
        target_model
    )
    target_model.to(device).eval()


    surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    wb_names = surrogate_names[:n_wb]
    norm_bound = 0.0625
    algo = 'P-RGF'
    step_size = args.step_size
    
    exp = f"{algo}_victim_{args.model_name}_{n_wb}wb_{norm_bound}_step{step_size}"
    if args.norm == 'linfty':
        exp = f"{algo}_victim_{args.model_name}_{n_wb}wb_linf_{norm_bound}_step{step_size}"
    if not args.targeted:
        exp = "untargeted_" + exp

    root = Path(f"results")
    adv_root = root / exp
    adv_root.mkdir(parents=True, exist_ok=True)

    surrogate_model_list = []
    for model_name in wb_names:
    # for s in range(len(args.smodel_name)):
        # surrogate_model = getattr(models, args.smodel_name[s])(pretrained=True)
        surrogate_model = getattr(models, model_name)(pretrained=True)
        if args.net_specific_resampling:
            image_width = 299 if model_name == 'inception_v3' else 224
            surrogate_model = torch.nn.Sequential(
                gfcs_util.Interpolate(torch.Size([image_width, image_width]), 'bilinear'),
                gfcs_util.Normalise(gfcs_util.imagenet_mean, gfcs_util.imagenet_std),
                surrogate_model
            )
        else:
            surrogate_model = torch.nn.Sequential(
                gfcs_util.Normalise(gfcs_util.imagenet_mean, gfcs_util.imagenet_std),
                surrogate_model
            )
        surrogate_model_list.append(surrogate_model.to(device).eval())

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    data_transform, image_width = gfcs_util.generate_data_transform(
        "imagenet_inception_299" if args.model_name == "inception_v3" else "imagenet_common_224"
    )

    # Set your ImageNet folder path here. Consult the documentation for torchvision.datasets.ImageNet to understand what
    # files must be placed where initially. Only the val set is required here.
    # imagenet_path = '/your/imagenet/dataset/path'
    # imagenet_dir = '../datasets/imagenet_data'
    # dataset = datasets.ImageNet(imagenet_dir, split='val', transform=data_transform)

    # if args.data_index_set == 'imagenet_val_random':
    #     input_index_list = torch.randperm(len(dataset))[:args.number_images]
    # else:
    #     input_index_list = getattr(eval_sets, args.data_index_set)[:args.number_images]

    # The following is hardcoded into the original, and is being replicated here as found:
    if args.norm == 'l2':
        learning_rate = 2.0 / np.sqrt(image_width * image_width * 3)
    else:  # linf
        learning_rate = 0.005
    # Note that in the reference, the above values of the learning rate were hard-coupled with choices of the norm limit
    #   "epsilon" (l2: sqrt(0.001 * image_size^2 * 3); linf: 0.05).
    # Here, the norm limit is taken as a parameter, leaving the user free to match or vary vs. the reference.

    # In the original, whether or not to "use larger step size" was bound to each network using a wrapper. Here, if you
    #   want this behaviour, you must specify it at the command line, but it is otherwise equivalent. Values are again
    #   taken from the reference code.
    if args.use_larger_step_size:
        ini_sigma = 1e-3
    else:
        ini_sigma = 1e-4

    success = 0
    queries = []
    correct = 0

    # Some more structures to line the outputs of this method up with our formats across the project:
    l2_norm_list = []
    linf_norm_list = []
    success_list = []
    all_queries_list = []  # This differs from "queries" in that it contains entries for failed attempts as well.

    img_paths, gt_labels, tgt_labels = load_imagenet_1000('../../imagenet1000')
    # for cnt, data_ind in enumerate(input_index_list):
    correct = len(img_paths)
    for i, im_path in tqdm(enumerate(img_paths)):
        im_idx = i
        # (image, label) = dataset[s]
        filename = im_path.name
        image = data_transform(Image.open(im_path).convert('RGB'))
        im_np = np.array(Image.open(im_path).convert('RGB'))
        label = gt_labels[i]
        # if args.targeted:
        #     label = tgt_labels[i]
        image = image.numpy()
        label = torch.tensor([label], device=device)
    # for cnt, data_ind in enumerate(input_index_list):
    #     (image, label) = dataset[data_ind]
    #     image = image.numpy()

        output_logging = open(adv_root / 'rgf_logging', 'a')
        # output_logging = open(os.path.join(args.output_dir, 'rgf_logging'), 'a')
        # filename = f"val_image_{data_ind:07d}.png"

        label = torch.tensor([label], device=device)

        # Note that this is assuming the untargeted attack setting.
        logits = target_model(torch.tensor(image, device=device, dtype=torch.float32).unsqueeze_(0))
        correctly_classified = (torch.argmax(logits, dim=1) == label)

        if correctly_classified:
            correct += 1

            sigma = ini_sigma

            np.random.seed(0)

            adv_image = image.copy()
            l = loss_func(logits, label).item()

            lr = learning_rate
            total_q = 0
            ite = 0

            while total_q <= args.max_queries:
                total_q += 1

                surrogate_model = random.choice(surrogate_model_list)

                if ite % 2 == 0 and sigma != ini_sigma:
                    print("sigma has been increased before; checking if sigma could be set back to ini_sigma")
                    rand = np.random.normal(size=adv_image.shape)
                    rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
                    rand_loss = loss_func(
                        target_model(torch.tensor(adv_image + ini_sigma * rand,
                                                  device=device, dtype=torch.float32).unsqueeze_(0)),
                        label
                    )
                    total_q += 1
                    rand = np.random.normal(size=adv_image.shape)
                    rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
                    rand_loss2 = loss_func(
                        target_model(torch.tensor(adv_image + ini_sigma * rand,
                                                  device=device, dtype=torch.float32).unsqueeze_(0)),
                        label
                    )
                    total_q += 1

                    if (rand_loss - l) != 0 and (rand_loss2 - l) != 0:
                        print("set sigma back to ini_sigma")
                        sigma = ini_sigma

                if args.method != 'uniform':
                    adv_image_pt = torch.tensor(
                        adv_image, device=device, dtype=torch.float32).unsqueeze_(0).requires_grad_()
                    with torch.enable_grad():
                        surrogate_loss = loss_func(surrogate_model(adv_image_pt), label)
                    surrogate_loss.backward()
                    prior = np.squeeze(adv_image_pt.grad.cpu().numpy())
                    adv_image_pt.requires_grad = False
                    adv_image_pt.grad.zero_()
                    prior = prior / np.maximum(1e-12, np.sqrt(np.mean(np.square(prior))))

                if args.method in ['biased', 'average']:
                    start_iter = 3
                    if ite % 10 == 0 or ite == start_iter:
                        s = 10
                        pert = np.random.normal(size=(s,) + adv_image.shape)
                        for i in range(s):
                            pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
                        eval_points = adv_image + sigma * pert
                        losses = loss_func(target_model(torch.tensor(eval_points, device=device, dtype=torch.float32)),
                                           torch.tensor(np.repeat(label.item(), s), device=device, dtype=torch.int64)
                                           ).cpu().numpy()
                        total_q += s
                        norm_square = np.average(((losses - l) / sigma) ** 2)

                    while True:
                        prior_loss = loss_func(
                            target_model(torch.tensor(adv_image + sigma * prior, device=device, dtype=torch.float32
                                                      ).unsqueeze_(0)),
                            label
                        ).item()
                        total_q += 1
                        diff_prior = prior_loss - l
                        if diff_prior == 0:
                            # Avoid the numerical issue in finite difference
                            sigma *= 2
                            print("multiply sigma by 2")
                        else:
                            break

                    est_alpha = diff_prior / sigma / np.maximum(np.sqrt(np.sum(np.square(prior)) * norm_square), 1e-12)
                    print("Estimated alpha =", est_alpha)
                    alpha = est_alpha
                    if alpha < 0:
                        prior = -prior
                        alpha = -alpha

                q = args.samples_per_draw
                n = image_width * image_width * 3
                d = 50 * 50 * 3
                gamma = 3.5
                A_square = d / n * gamma

                return_prior = False
                if args.method == 'average':
                    if args.dataprior:
                        alpha_nes = np.sqrt(A_square * q / (d + q - 1))
                    else:
                        alpha_nes = np.sqrt(q / (n + q - 1))
                    if alpha >= 1.414 * alpha_nes:
                        return_prior = True
                elif args.method == 'biased':
                    if args.dataprior:
                        best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                                A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
                    else:
                        best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                                alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
                    print('best_lambda = ', best_lambda)
                    if best_lambda < 1 and best_lambda > 0:
                        lmda = best_lambda
                    else:
                        if alpha ** 2 * (n + 2 * q - 2) < 1:
                            lmda = 0
                        else:
                            lmda = 1
                    if np.abs(alpha) >= 1:
                        lmda = 1
                    print('lambda = ', lmda)
                    if lmda == 1:
                        return_prior = True
                elif args.method == 'fixed_biased':
                    lmda = args.fixed_const

                if not return_prior:
                    if args.dataprior:
                        pert = np.random.normal(size=(q, 3, 50, 50))
                        pert = torch.nn.functional.interpolate(
                            torch.tensor(pert, device=device, dtype=torch.float32),
                            mode='nearest', size=adv_image.shape[-2:]).cpu().numpy()
                    elif args.method == 'ods':
                        adv_image_pt_stack = torch.tensor(
                            adv_image, device=device, dtype=torch.float32
                        ).unsqueeze_(0).repeat(q, 1, 1, 1).requires_grad_()
                        ods_directions = torch.rand((q, 1000), device=device, dtype=torch.float32) * 2. - 1.
                        with torch.enable_grad():
                            # NOTE that as it stands, too high a value of samples_per_draw (q) can lead to running out
                            #   of memory here. You could work around that by batching (doing fwd/bwd passes on smaller
                            #   batches of ODS directions, and stacking the sampled gradients into a final result
                            #   structure as you go).
                            loss = (surrogate_model(adv_image_pt_stack) * ods_directions).sum()
                        loss.backward()
                        pert = adv_image_pt_stack.grad.cpu().numpy()
                    else:
                        pert = np.random.normal(size=(q,) + adv_image.shape)
                    for i in range(q):
                        if args.method in ['biased', 'fixed_biased']:
                            pert[i] = pert[i] - np.sum(pert[i] * prior) * prior / np.maximum(1e-12,
                                                                                             np.sum(prior * prior))
                            pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
                            pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior
                        else:
                            pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))

                    while True:
                        eval_points = adv_image + sigma * pert
                        losses = loss_func(target_model(torch.tensor(eval_points, device=device, dtype=torch.float32)),
                                           torch.tensor(np.repeat(label.item(), q), device=device, dtype=torch.int64)
                                           ).cpu().numpy()
                        total_q += q

                        grad = (losses - l).reshape(-1, 1, 1, 1) * pert
                        grad = np.mean(grad, axis=0)
                        norm_grad = np.sqrt(np.mean(np.square(grad)))
                        if norm_grad == 0:
                            sigma *= 5
                            print("estimated grad == 0, multiply sigma by 5")
                        else:
                            break
                    grad = grad / np.maximum(1e-12, np.sqrt(np.mean(np.square(grad))))

                    if args.method == 'average':
                        while True:
                            diff_prior = loss_func(
                                target_model(torch.tensor(adv_image + sigma * prior, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            total_q += 1
                            diff_nes = loss_func(
                                target_model(torch.tensor(adv_image + sigma * grad, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            total_q += 1
                            diff_prior = max(0, diff_prior)
                            if diff_prior == 0 and diff_nes == 0:
                                sigma *= 2
                                print("multiply sigma by 2")
                            else:
                                break
                        final = prior * diff_prior + grad * diff_nes
                        final = final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
                        print("diff_prior = {}, diff_nes = {}".format(diff_prior, diff_nes))
                    elif args.method == 'fixed_average':
                        diff_prior = loss_func(
                            target_model(torch.tensor(adv_image + sigma * prior, device=device, dtype=torch.float32
                                                      ).unsqueeze_(0)),
                            label).item() - l
                        total_q += 1
                        if diff_prior < 0:
                            prior = -prior
                        final = args.fixed_const * prior + (1 - args.fixed_const) * grad
                        final = final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
                    else:
                        final = grad

                    def print_loss(model, direction):
                        length = [1e-4, 1e-3]
                        les = []
                        for ss in length:
                            les.append(
                                loss_func(
                                    model(torch.tensor(adv_image + ss * direction, device=device, dtype=torch.float32
                                                       ).unsqueeze_(0)),
                                    label
                                ).item() - l
                            )
                        print("losses", les)

                    if args.show_loss:
                        if args.method in ['average', 'fixed_average']:
                            lprior = loss_func(
                                target_model(torch.tensor(adv_image + lr * prior, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            print_loss(target_model, prior)
                            lgrad = loss_func(
                                target_model(torch.tensor(adv_image + lr * grad, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            print_loss(target_model, grad)
                            lfinal = loss_func(
                                target_model(torch.tensor(adv_image + lr * final, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            print_loss(target_model, final)
                            print(lprior, lgrad, lfinal)
                        elif args.method in ['biased', 'fixed_biased']:
                            lprior = loss_func(
                                target_model(torch.tensor(adv_image + lr * prior, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            print_loss(target_model, prior)
                            lgrad = loss_func(
                                target_model(torch.tensor(adv_image + lr * grad, device=device, dtype=torch.float32
                                                          ).unsqueeze_(0)),
                                label).item() - l
                            print_loss(target_model, grad)
                            print(lprior, lgrad)
                else:
                    final = prior

                if args.norm == 'l2':
                    adv_image = adv_image + lr * final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
                    norm = max(1e-12, np.linalg.norm(adv_image - image))
                    factor = min(1, args.norm_bound / norm)
                    # ^ Note that args.norm_bound is what was called "eps" in the reference code. It was hardcoded
                    #   there, whereas for us, it's an input parameter.
                    adv_image = image + (adv_image - image) * factor
                else:
                    adv_image = adv_image + lr * np.sign(final)
                    adv_image = np.clip(adv_image, image - args.norm_bound, image + args.norm_bound)
                    # ^ See above, re: "eps" and args.norm_bound.
                adv_image = np.clip(adv_image, 0, 1)

                adv_logits = target_model(torch.tensor(adv_image, device=device, dtype=torch.float32).unsqueeze_(0))
                adv_label = torch.argmax(adv_logits, dim=1)
                l = loss_func(adv_logits, label).item()

                print('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:',
                      adv_label.item(), 'distortion:', np.max(np.abs(adv_image - image)),
                      np.linalg.norm(adv_image - image))

                ite += 1

                # if adv_label != label:
                #     print('Stop at queries:', total_q)
                #     success += 1
                #     queries.append(total_q)

                #     output_logging.write(filename + ' succeed; queries: ' + str(total_q) + '\n')

                if (not args.targeted and (adv_label != label)) or (args.targeted and (adv_label == label)):
                    print('Stop at queries:', total_q)
                    success += 1
                    queries.append(total_q)
                    linf_ = np.max(np.abs(adv_image - image))
                    l2_ = np.linalg.norm(adv_image - image)
                    info = f"im_idx: {im_idx}, queries: {total_q}, linf: {linf_}, l2: {l2_}\n"
                    output_logging.write(info)

                    # Required for our outputs which enable comparisons across methods:
                    l2_norm_list.append(np.linalg.norm(adv_image - image))
                    linf_norm_list.append(np.max(np.abs(adv_image - image)))
                    success_list.append(True)
                    all_queries_list.append(total_q)

                    break
            else:
                output_logging.write(filename + ' fail.\n')

                # Required for our outputs which enable comparisons across methods:
                l2_norm_list.append(np.linalg.norm(adv_image - image))
                linf_norm_list.append(np.max(np.abs(adv_image - image)))
                success_list.append(False)
                all_queries_list.append(total_q)

            output_logging.close()
        
            # save the adversarial image
            print(f"adv_img: {adv_image.shape}, {adv_image.min()}, {adv_image.max()}")
            adv_np = (adv_image*255).astype(np.uint8).transpose(1, 2, 0)
            im = Image.fromarray(adv_np)
            im.save(adv_root / f'{img_paths[im_idx].name}')
            diff = adv_np - im_np
            print(f"im_idx: {im_idx}, min: {diff.min()}, max: {diff.max()}")
        # else:
        #     output_logging.write(filename + ' original misclassified.\n')
        #     output_logging.close()
        #     print(f'image {cnt+1}: already adversary')

    # Note that in our experimental protocol, all input images are assumed to be correctly classified by the target
    #   net, and thus for 'total' to always equal the input set size. This logging code (kept from the original) won't
    #   output the correct rates if that assumption is violated. It is not used in our results, but, be aware.
    total = correct
    print('Success rate:', success / total, 'Queries', queries)
    output_logging = open(os.path.join(args.output_dir, 'rgf_logging'), 'a')
    output_logging.write('Success rate: ' + str(success / total) + ', Queries on success: ' + str(np.mean(queries)) +
                         '\n\n')
    output_logging.close()

    # output_file_path = os.path.join(args.output_dir, args.output)
    output_file_path = adv_root/f"{exp}.pt"
    print("Saving to file", output_file_path)
    torch.save({
        "succs": torch.BoolTensor(success_list),
        "queries": torch.IntTensor(all_queries_list),
        "l2_norms": torch.as_tensor(l2_norm_list),
        "linf_norms": torch.as_tensor(linf_norm_list),
        "canonical_adv": None,
        "input_args": Namespace(attack_mode="rgf", args=args),
    }, output_file_path)


if __name__ == '__main__':
    p_rgf_pytorch()
