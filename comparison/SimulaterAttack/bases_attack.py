"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"
to compare results with SimulatorAttack

This file goes under folder SimulatorAttack

Here we use the same datasets and models as SimulatorAttack

Datasets: tiny-imagenet-200 http://cs231n.stanford.edu/tiny-imagenet-200.zip
Put it in folder: SimulatorAttack/dataset/tinyImageNet/
and unzip to tiny-imagenet-200/

Trained models: https://cloud.tsinghua.edu.cn/d/a11beb12358b416199b9/?p=%2Freal_image_model&mode=list
Put in folder: SimulatorAttack/train_pytorch_model/real_image_model/

"""


import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from config import IN_CHANNELS, CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

sys.path.append("../..")
from utils_bases import get_loss_fn

preprocess = transforms.Compose([
        transforms.ToTensor(),
        ])

relu = torch.nn.ReLU()
softmax = torch.nn.Softmax(dim=1)
softmax_np = lambda x: np.exp(x) / np.sum(np.exp(x))
loss_ce_np = lambda x,y: -np.log(softmax_np(x)[y])  # x: list - logits, y: int - the label


def get_logits_probs(im_np, model):
    """Get the logits of a model given an image as input
    Args:
        im (PIL.Image or np.ndarray): uint8 image (read from PNG file), for 0-1 float image
        model (torchvision.models): the model returned by function load_model
    Returns:
        logits (numpy.ndarray): direct output of the model
        probs (numpy.ndarray): logits after softmax function
    """
    device = next(model.parameters()).device
    im_pil = Image.fromarray(im_np.astype(np.uint8))
    im_tensor = preprocess(im_pil).unsqueeze(0).to(device)
    logits = model(im_tensor)
    probs = softmax(logits)
    logits = logits.detach().cpu().numpy().squeeze() # convert torch tensor to numpy array
    probs = probs.detach().cpu().numpy().squeeze()
    return logits, probs


def get_label_loss(im, model, tgt_label, loss_name, targeted=True):
    """Get the loss (ce)
    Args:
        im (PIL.Image): uint8 image (read from PNG file)
        tgt_label (int): target label
        loss_name (str): 'cw', 'ce'
    """
    loss_fn = get_loss_fn(loss_name, targeted=targeted)
    logits, _ = get_logits_probs(im, model)
    pred_label = logits.argmax()
    logits = torch.tensor(logits)
    tgt_label = torch.tensor(tgt_label)
    loss = loss_fn(logits, tgt_label)
    return pred_label, loss


def main():
    parser = argparse.ArgumentParser(description="BASES attack")
    parser.add_argument("--victim", nargs="?", default='densenet121', help="victim model")
    parser.add_argument("--n_wb", type=int, default=10, help="number of models in the ensemble: 4,10,20")
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=8, help="perturbation bound: 8 for linf")
    parser.add_argument("--iters", type=int, default=50, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--x", type=int, default=1, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--n_im", type=int, default=1000, help="number of images")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    args = parser.parse_args()
    
    arch = args.victim
    n_iters = args.iters
    # eps = args.eps
    # alpha = 1e-2
    bound = args.bound

    if bound == 'linf':
        eps = 8
        alpha = args.x * eps / n_iters
    else: # bound = 'l2'
        eps = 1173 # 4.6*255
        alpha = 1e-1
        
    # n_iters = args.iters
    loss_name = args.loss_name
    lr_w = float(args.lr)
    iterw = args.iterw
    untargeted = False


    root = Path('results')
    exp = f'{arch}_{bound}'
    exp_root = root / f"logs" / exp
    exp_root.mkdir(parents=True, exist_ok=True)
    adv_root = root / f"adv_images" / exp
    adv_root.mkdir(parents=True, exist_ok=True)
    clean_root = root / f"clean_images" / exp
    clean_root.mkdir(parents=True, exist_ok=True)
    log_file = root / f"{exp}.txt"

    dataset = "TinyImageNet"
    victim_model = StandardModel(dataset, arch, no_grad=True).cuda()
    victim_model.eval()

    wb_names = ["vgg13","densenet169","vgg11_bn","resnet34","vgg19","vgg13_bn","vgg11","resnet18","vgg16", \
        "vgg19_bn","densenet201","resnet101","densenet161","resnet50","vgg16_bn","resnet152"]
    wb = []
    for model_name in wb_names:
        print(f"load: {model_name}")
        model = StandardModel(dataset, arch, no_grad=False).cuda()
        model.eval()
        wb.append(model)
    n_wb = len(wb)

    dataset_loader = DataLoaderMaker.get_img_label_data_loader(dataset, batch_size=1, is_train=False, image_size=None)
    total_images = len(dataset_loader.dataset)
    print(total_images)


    def get_adv_np(w_np, adv_np, im_np, tgt_label, bound=bound, eps=eps, alpha = alpha):
        """Get the adversarial image by attacking the perturbation machine
            w_np, adv_np, im_np (np.ndarray): 
            tgt_label (int): 
            bound (str): choices=['linf','l2'], bound in linf or l2 norm ball
            eps, n_iters, alpha (float/int): perturbation budget, number of steps, step size
        """
        w = torch.from_numpy(w_np).float().cuda()
        im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().cuda()
        adv = torch.from_numpy(adv_np).permute(2,0,1).unsqueeze(0).float().cuda()
        loss_fn = get_loss_fn(loss_name='cw', targeted = not untargeted)
        target = torch.LongTensor([tgt_label]).cuda()

        for it in range(n_iters):
            adv.requires_grad=True
            outputs = [model(adv/255) for model in wb]
            logits_weighted = sum([w[idx] * outputs[idx] for idx in range(n_wb)])
            loss = loss_fn(logits_weighted,target)
            adv_np = adv.detach().squeeze().cpu().numpy().transpose(1, 2, 0)

            # pred_adv = get_label(adv_np, victim_model)
            # pred_label, _ = get_label_loss(adv_np, victim_model, tgt_label, loss_name='cw')
            # print(f"loss: {loss}, adv label: {pred_label}")
            
            loss.backward()
            with torch.no_grad():
                grad = adv.grad
                adv = adv - alpha * torch.sign(grad)

                if bound == 'linf':
                    # project to linf ball
                    adv = (im + (adv - im).clamp(min=-eps,max=eps)).clamp(0,255)
                else:
                    # project to l2 ball
                    pert = adv - im
                    l2_norm = torch.norm(pert, p=2)
                    if l2_norm > eps:
                        pert = pert / l2_norm * eps
                    adv = (im + pert).clamp(0,255)

        adv_np = adv.squeeze().cpu().numpy().transpose(1, 2, 0)
        return adv_np


    success_idx_list = set()
    success_idx_list_pretend = set() # pretend targeted to be untargeted
    query_list = []
    query_list_pretend = []
    w_np = np.array([1 for _ in range(len(wb))]) / len(wb)

    cnt = 0
    for im_idx, data_tuple in tqdm(enumerate(dataset_loader)):
        cnt += 1

        images, true_labels = data_tuple[0], data_tuple[1]
        im_np = images.numpy().squeeze().transpose([1,2,0])*255
        print(im_np.shape)
        im_pil = Image.fromarray(im_np.astype(np.uint8))
        im_pil.save(clean_root / f'{im_idx}.png')

        images, true_labels = images.cuda(), true_labels.cuda()
        target_labels = torch.fmod(true_labels + 1, CLASS_NUM[dataset])
        gt_label = true_labels.cpu().numpy()[0]
        tgt_label = target_labels.cpu().numpy()[0]
        target = torch.LongTensor([tgt_label]).cuda()
        print(f"gt_label: {gt_label}, tgt_label: {tgt_label}")

        pred, loss = get_label_loss(im_np, victim_model, tgt_label, loss_name, targeted = not untargeted)
        print(f"labels: gt_label - {gt_label}, pred - {pred}")
    
        lr_w = float(1e-2)
        # start from equal weights
        n_query = 0
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        adv_np = get_adv_np(w_np, adv_np = im_np, im_np = im_np, tgt_label = tgt_label)
        

        label_idx, loss = get_label_loss(adv_np, victim_model, tgt_label, loss_name, targeted = not untargeted)
        n_query += 1
        w_list = []
        loss_bb_list = []       # loss of victim model
        print(f"{label_idx}, loss: {loss}")
        print(f"w: {w_np.tolist()}")

        # pretend
        if not untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)

        if (not untargeted and label_idx == tgt_label) or (untargeted and label_idx != tgt_label):
            # originally successful
            print('success')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
            w_list.append(w_np.tolist())
            iters = 0 # if successful at 1st round
            loss_bb_list.append(loss)
        else: 
            idx_w = 0         # idx of wb in W
            last_idx = 0    # if no changes after one round, reduce the learning rate
            
            while n_query < iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w
                adv_np_plus = get_adv_np(w_np_temp_plus, adv_np = adv_np, im_np = im_np, tgt_label = tgt_label)
                label_plus, loss_plus = get_label_loss(adv_np_plus, victim_model, tgt_label, loss_name, targeted = not untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} +, {label_plus}, loss: {loss_plus}")
                
                # pretend
                if (not untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not untargeted)*(tgt_label == label_plus) or untargeted*(tgt_label != label_plus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w
                adv_np_minus = get_adv_np(w_np_temp_minus, adv_np = adv_np, im_np = im_np, tgt_label = tgt_label)
                label_minus, loss_minus = get_label_loss(adv_np_minus, victim_model, tgt_label, loss_name, targeted = not untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} -, {label_minus}, loss: {loss_minus}")

                # pretend
                if (not untargeted and label_minus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not untargeted)*(tgt_label == label_minus) or untargeted*(tgt_label != label_minus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    break

                # update
                if loss_plus < loss_minus:
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    print(f"{idx_w} +")
                    last_idx = idx_w
                else:
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    print(f"{idx_w} -")
                    last_idx = idx_w
                    
                idx_w = (idx_w+1)%n_wb
                if n_query > 5 and last_idx == idx_w:
                    lr_w /= 2 # half the lr if there is no change
                    print(f"lr_w: {lr_w}")

                w_list.append(w_np.tolist())
                loss_bb_list.append(loss)

        if im_idx in success_idx_list:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"im_idx: {im_idx}; total_success: {len(success_idx_list)}")

        if im_idx in success_idx_list_pretend:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list_pretend[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}_pretend.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"pretend - im_idx: {im_idx}; total_success: {len(success_idx_list_pretend)}")


        # save adv image
        adv_path = adv_root / f"{im_idx}.png"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)


        print(f"query_list: {query_list}")
        print(f"avg queries: {np.mean(query_list)}")
        print(f"query_list_pretend: {query_list_pretend}")
        print(f"avg queries pretend: {np.mean(query_list_pretend)}")


        
        with open(log_file, 'a') as f:
            f.write(f"im_id: {im_idx}, targeted: {query_list}, mean {np.mean(query_list)}; untargeted: {query_list_pretend}, mean: {np.mean(query_list_pretend)}\n")

        if cnt >= 1000:
            break


if __name__ == '__main__':
    main()
