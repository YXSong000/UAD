import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random
from sklearn.metrics import confusion_matrix
from networks.models import DenseNet121
from temperature_scaling import ModelWithTemperature

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['target_'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=1, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def train_target(args):
    dset_loaders = data_load(args)
    net = pick_n_adapt(args)
    param_group = []
    for k, v in net.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            net.eval()
            pseudo_label = obtain_label(args, dset_loaders['target_'])
            pseudo_label = torch.from_numpy(pseudo_label).cuda()
            net.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        _, outputs_test, _ = net(inputs_test)

        if args.cls_par > 0:
            pred = pseudo_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            net.eval()
            acc_s_te, _ = cal_acc(dset_loaders['test'], net, False)
            log_str = 'Task: {}, Iter:{}/{}; Trained Model Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n\n')
            args.out_file.flush()
            print(log_str+'\n')
            net.train()
  
    torch.save(net.state_dict(), osp.join(args.output_dir, "target_" + args.savename + ".pt"))
        
    return net



def cal_confidence_acc(args, loader, net, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs, _ = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        
    scaled_model = ModelWithTemperature(args, net)
    temperature_value = scaled_model.set_temperature(loader)
    temperature = torch.tensor(temperature_value).reshape(-1).unsqueeze(1).expand(all_output.size(0), all_output.size(1)).float().cpu()

    all_output = nn.Softmax(dim=1)(all_output/temperature)

    all_confidence_measures = torch.max(all_output, dim=1)[0] - torch.topk(all_output, k=2)[0][:, -1]
    mean_confidence_measure = torch.mean(all_confidence_measures)

    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return mean_confidence_measure.item(), accuracy*100, temperature_value, mean_ent


def pick_n_adapt(args):
    log_str = '======Select Source Model...======'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    dset_loaders = data_load(args)
    
    net_list = [DenseNet121(out_size=args.class_num, mode='U-Ones').cuda() for i in range(len(args.src))]

    conf_list = []
    acc_list = []

    for i in range(len(args.src)):
        args.modelpath = args.output_dir_src[i] + '/source_lr0.01_no-sch.pt'   
        net_list[i].load_state_dict(torch.load(args.modelpath))
        net_list[i].eval()

        confidence_measure, acc, temperature_value, _ = cal_confidence_acc(args, dset_loaders['test'], net_list[i], False)
        conf_list.append(confidence_measure)
        acc_list.append(acc)
        log_str = 'Model: {}, Temperature = {:.5f}, Confidence = {:.5f}'.format(args.src[i], temperature_value, confidence_measure)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')

    ind = conf_list.index(max(conf_list))
    log_str = 'Best Model: {}, Accuracy = {:.2f}%'.format(args.src[ind], acc_list[ind])
    args.name = args.src[ind]
    args.out_file.write(log_str + '\n\n')
    args.out_file.flush()
    print(log_str+'\n')

    return net_list[ind]


def obtain_label(args, loader):
    log_str = '======Obtain Test Set Pseudo-Label...======'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')


    net_list = [DenseNet121(out_size=args.class_num, mode='U-Ones').cuda() for i in range(len(args.src))]

    temperature_value_list = []

    for i in range(len(args.src)):
        args.modelpath = args.output_dir_src[i] + '/source_lr0.01_no-sch.pt'   
        net_list[i].load_state_dict(torch.load(args.modelpath))
        net_list[i].eval()
        scaled_model = ModelWithTemperature(args, net_list[i])
        temperature_value = scaled_model.set_temperature(loader)
        temperature_value_list.append(temperature_value)
        log_str = 'Model: {}, Overall temperature = {:.5f}'.format(args.src[i], temperature_value)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

    args.out_file.write('\n')
    args.out_file.flush()

    log_str = 'Rescaled temperature for pseudo-labelling: {}'.format(temperature_value_list)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            conf_list = []
            scr_outputs = []
            data = iter_test.next()
            input = data[0]
            label = data[1]
            input = input.cuda()
            label = label.float()
            # print(label)
            for i in range(len(args.src)):
                _, output, _ = net_list[i](input)
                output = output.float().cpu()
                
                temperature = torch.tensor(temperature_value_list[i]).reshape(-1).unsqueeze(1).expand(output.size(0), output.size(1)).float().cpu()
                output = nn.Softmax(dim=1)(output/temperature)
                scr_outputs.append(output)
                
                confidence_measure = torch.max(output, dim=1)[0] - torch.topk(output, k=2)[0][:, -1]
                conf_list.append(confidence_measure)

            ind = conf_list.index(max(conf_list))
            
            if start_test:
                all_output = scr_outputs[ind].float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, scr_outputs[ind].float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)
      
    _, predict = torch.max(all_output, 1)
    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Pseudo-Label Accuracy = {:.2f}%'.format(accuracy * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.numpy().astype('int')


def cal_acc(loader, net, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs, _ = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='HAM10000', choices=['DR', 'HAM10000'])
    parser.add_argument('--lr', type=float, default=1*1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/adapt_ours')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    args = parser.parse_args()
    
    if args.dset == 'DR':
        names = ['APTOS-2019', 'DDR', 'IDRiD']
        args.class_num = 5
    if  args.dset == 'HAM10000':
        names = ['back', 'face', 'lowerExtremity', 'upperExtremity']
        args.class_num = 2

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = './dataset/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset + '_densenet', args.src[i][0].upper()))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())


    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)

    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_target(args)

