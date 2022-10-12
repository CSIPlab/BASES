"""
This code is used to produce TREMBA comparison with
NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Code modified for new datasets and models
"""

import sys
import argparse
import os
import json
import csv
# import DataLoader
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from utils import Function, MarginLoss_Single
from FCN import Imagenet_Encoder, Imagenet_Decoder
from Normalize import Normalize, Permute
from imagenet_model.Resnet import resnet152_denoise, resnet101_denoise

sys.path.append('../../')
from utils_bases import load_imagenet_1000, get_label


preprocess = transforms.Compose([
    transforms.Resize(224), 
    # transforms.Resize(256), 
    # transforms.CenterCrop(224)
    ])

softmax = torch.nn.Softmax(dim=1)


def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
    device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size']), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']
    for iter in range(config['num_iters']+1):

        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])
        logit, loss = function(torch.clamp(image+perturbation, 0, 1), label)
        if config['target']:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) !=label
        last_loss.append(loss.item())

        if function.current_counts > 50000:
            break
        
        if bool(success.item()):
            return True, torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        if iter % config['log_interval'] == 0 and config['print_log']:
            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, origin_image


parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='config.json', help='config file')
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--model_name', default=None)
parser.add_argument('--eps', type=float, default=0.0625, help='max linf norm')
parser.add_argument("-targeted", action='store_true', help="run targeted attack")
parser.add_argument('--target_class', type=int, default=0, help='targeted class 0, 20, ...')
parser.add_argument("-random_generator", action='store_true', help="use a random generator")
args = parser.parse_args()

if args.targeted:
    config_file = "config/attack_target.json"
else:
    config_file = "config/attack_untarget.json"
with open(config_file) as config_file:
    state = json.load(config_file)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
if args.model_name is not None:
    state['model_name'] = args.model_name

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
state['epsilon'] = args.eps
state['target'] = args.targeted
state['target_class'] = args.target_class

state['generator_name'] = f"Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_target_{args.target_class}"
weight = torch.load(os.path.join("G_weight", state['generator_name']+".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val

# _, dataloader, nlabels, mean, std = DataLoader.imagenet(new_state)
if 'defense' in new_state and new_state['defense']:
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
else:               
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
nlabels = 1000

if 'OSP' in state:
    if state['source_model_name'] == 'Adv_Denoise_Resnet152':
        s_model = resnet152_denoise()
        loaded_state_dict = torch.load(os.path.join('weight', state['source_model_name']+".pytorch"))
        s_model.load_state_dict(loaded_state_dict)
    if 'defense' in state and state['defense']:
        source_model = nn.Sequential(
            Normalize(mean, std),
            Permute([2,1,0]),
            s_model
        )
    else:
        source_model = nn.Sequential(
            Normalize(mean, std),
            s_model
        )


if args.targeted:
    target_class = state['target_class']
    adv_root = f"adv_images_victim_{args.model_name}_targetClass_{target_class}_eps_{args.eps}/"
else:
    adv_root = f"untargeted_adv_images_victim_{args.model_name}_eps_{args.eps}/"
if args.random_generator:
    adv_root = "random_" + adv_root

adv_root = "adv_images_new/" + adv_root
adv_root = Path(adv_root)
adv_root.mkdir(parents=True, exist_ok=True)
log_root = "output_new/"
log_root = Path(log_root)
log_root.mkdir(parents=True, exist_ok=True)

if state['model_name'] == 'Resnet34':
    pretrained_model = models.resnet34(pretrained=True)
elif state['model_name'] == 'VGG19':
    pretrained_model = models.vgg19_bn(pretrained=True)
elif state['model_name'] == 'Densenet121':
    pretrained_model = models.densenet121(pretrained=True)
elif state['model_name'] == 'Mobilenet':
    pretrained_model = models.mobilenet_v2(pretrained=True)
elif state['model_name'] == 'Adv_Denoise_Resnext101':
    pretrained_model = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', state['model_name']+".pytorch"))
    pretrained_model.load_state_dict(loaded_state_dict, strict=True)
else:
    # pretrained_model = models.resnet34(pretrained=True)
    pretrained_model = getattr(models, args.model_name)(pretrained=True)
if 'defense' in state and state['defense']:
    model = nn.Sequential(
        Normalize(mean, std),
        Permute([2,1,0]),
        pretrained_model
    )
else:
    model = nn.Sequential(
        Normalize(mean, std),
        pretrained_model
    )

encoder = Imagenet_Encoder()
decoder = Imagenet_Decoder()
if not args.random_generator:
    encoder.load_state_dict(encoder_weight)
    decoder.load_state_dict(decoder_weight)

model.to(device)
model.eval()
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

if 'OSP' in state:
    source_model.to(device)
    source_model.eval()

F = Function(model, state['batch_size'], state['margin'], nlabels, state['target'])

count_success = 0
count_total = 0
if not os.path.exists(state['save_path']):
    os.mkdir(state['save_path'])


img_paths, gt_labels, tgt_labels = load_imagenet_1000('../../imagenet1000')
# for i, (images, labels) in enumerate(dataloader):
for i in tqdm(range(len(img_paths))):
    img_path = img_paths[i]
    im_np = np.array(Image.open(img_path).convert('RGB'))
    if not args.targeted:
        label = get_label(im_np, model=model)
        print(f"pred of im: {label}, range: {im_np.min()} - {im_np.max()}")
    else:
        label = tgt_labels[i] # get target label
    image = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0)
    image = preprocess(image/255)

    images = image.to(device)
    labels = int(label)
    logits = model(images)
    print(f"pred: {torch.argmax(logits, dim=1).item()}, gt: {gt_labels[i]}, tgt: {label}")
    # correct = torch.argmax(logits, dim=1) == labels
    correct = 1
    if correct:
        torch.cuda.empty_cache()
        if state['target']:
            labels = state['target_class']

        if 'OSP' in state:
            hinge_loss = MarginLoss_Single(state['white_box_margin'], state['target'])
            images.requires_grad = True
            latents = encoder(images)
            print(f"latent: {latents.shape}")
            for k in range(state['white_box_iters']):     
                perturbations = decoder(latents)*state['epsilon']
                logits = source_model(torch.clamp(images+perturbations, 0, 1))
                loss = hinge_loss(logits, labels)
                grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - state['white_box_lr'] * grad

            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state, latents.view(-1))

        else:
            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state)

        count_success += int(success)
        count_total += int(correct)
        
        # calculate info and save images
        adv_np = adv.squeeze().cpu().numpy().transpose(1, 2, 0)
        im_np = images.squeeze().cpu().numpy().transpose(1, 2, 0)
        diff = adv_np - im_np
        linf = np.max(np.abs(diff))
        l2 = np.sqrt(np.sum(diff**2))
        adv_path = str(adv_root/f"{img_path.name}")
        cv2.imwrite(adv_path, adv_np[:, :, [2, 1, 0]]*255)

        # load from png
        adv_np = np.array(Image.open(adv_path).convert('RGB'))
        pred_label = get_label(adv_np, model=model)
        print(f"pred of adv: {pred_label}, range: {adv_np.min()} - {adv_np.max()}")

        info = f"image: {i}, eval_count: {F.current_counts}, success: {success}, linf: {linf:.4f}, l2: {l2:.4f}, average_count: {F.get_average()}, success_rate: {float(count_success) / float(count_total):.2f}\n"
        print(info)
        if args.targeted:
            target_class = state['target_class']
            file = open(log_root / f'targeted_victim_{args.model_name}_targetClass_{target_class}_eps{args.eps}.txt', 'a')
        else:
            file = open(log_root / f'untargeted_victim_{args.model_name}_eps{args.eps}.txt', 'a')
        file.write(f"{info}")
        file.close()
        F.new_counter()


success_rate = float(count_success) / float(count_total)

print("success rate {}".format(success_rate))

print("average eval count {}".format(F.get_average()))
