import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import myTransform
from functions import TET_loss, seed_all, get_logger
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from models import VGGSNN, VGGPSN, MaskedSlidingPSN
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j',
                    '--workers',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=100,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb',
                    default=1e-3,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('-out_dir', default='./logs/', type=str, help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-method', type=str, default='VGGSNN', help='use which network')
parser.add_argument('-opt', type=str, default='SGD0.1', help='optimizer method')
parser.add_argument('-tau', type=float, default=0.25, help='tau of LIF')
parser.add_argument('-TET', action='store_true', help='use the tet loss')
parser.add_argument('-fixed', action='store_true')
parser.add_argument('-loss_lambda', type=float, default=1e-7)
parser.add_argument('--test_only', action='store_true', help='test only')
args = parser.parse_args()
# print(args)


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    firing_loss = 0
    train_firing_num = 0
    train_firing_num_t = 0
    train_firing_rate = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)

        outputs, firing_num_t, firing_rate = model(images, epoch)

        mean_out = outputs.mean(1) # mean over time

        firing_num = firing_num_t.sum().item()
        
        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        # _, predicted = output.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
        
        # pdb.set_trace()
        train_firing_num += firing_num
        train_firing_num_t += firing_num_t
        train_firing_rate += firing_rate * images.shape[0]
    end_time = time.time()
    print(f'train a epoch time: {end_time - start_time:.2f}s')
        
    running_loss = running_loss / total
    acc = 100 * correct / total
    train_firing_num = train_firing_num / total
    train_firing_num_t = train_firing_num_t / total
    train_firing_rate = train_firing_rate / total

    return running_loss, firing_loss, acc, train_firing_num, train_firing_num_t, train_firing_rate 

@torch.no_grad()
def test(model, test_loader, device, epoch=None):
    correct = 0
    total = 0
    test_firing_num = 0
    test_firing_num_t = 0
    test_firing_rate = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs, firing_num_t, firing_rate = model(inputs, epoch)

        mean_out = outputs.mean(1)  # mean over time

        _, predicted = mean_out.cpu().max(1)

        firing_num = firing_num_t.sum().item()

        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        test_firing_num += firing_num
        test_firing_num_t += firing_num_t
        test_firing_rate += firing_rate * inputs.shape[0]
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    test_firing_num = test_firing_num / total
    test_firing_num_t = test_firing_num_t / total
    test_firing_rate = test_firing_rate / total
    print(f"test firing_num_t: {test_firing_num_t}, test_firing_rate: {test_firing_rate}")
    return final_acc, test_firing_num, test_firing_num_t, test_firing_rate

def build_dvscifar():
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48), antialias=None),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(),])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
        # Cutout(n_holes=1, length=16)])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48), antialias=None)])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    train_set = CIFAR10DVS(root='/mnt/ssd/share/dfxue/datasets/CIFAR10-DVS', train=True, data_type='frame', frames_number=args.T, split_by='number', transform=transform_train) # '/mnt/data/dfxue/datasets/CIFAR10-DVS'
    test_set = CIFAR10DVS(root='/mnt/ssd/share/dfxue/datasets/CIFAR10-DVS/', train=False, data_type='frame', frames_number=args.T, split_by='number', transform=transform_test)

    return train_set, test_set

if __name__ == '__main__':
    seed_all(args.seed)
    train_dataset, val_dataset = build_dvscifar()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    print(len(train_loader))
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    print(len(test_loader))


    if args.method == 'PSN':
        model = VGGPSN()
    else:
        model = VGGSNN(tau=args.tau)
    print(model)

    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'SGD0.1':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    elif args.opt == 'SGD0.02':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    log_file_name = f'T{args.T}_opt_{args.opt}_tau_{args.tau}_method_{args.method}_b_{args.batch_size}'
    if args.TET:
        log_file_name += '_TET'

    num_gpus = torch.cuda.device_count()
    log_file_name += f'_{num_gpus}gpu' 

    log_file_name += f'_mi-trqr'

    if args.seed != 100:
        log_file_name += f'_s{args.seed}'
    print(log_file_name)

    start_epoch = 0
    best_acc = 0
    fr_ba = 0
    best_epoch = 0
    out_dir = os.path.join(args.out_dir, log_file_name)

    if args.resume:
        print('load resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        # fr_ba = checkpoint['fr_ba']
        print(f"resume from epoch: {start_epoch}, best_acc: {best_acc}, fr_ba: {fr_ba}")

    if args.test_only:
        facc, firing_num, firing_num_t, firing_rate = test(parallel_model, test_loader, device, epoch=start_epoch)
        print(f"Test acc: {facc}, firing_num: {firing_num}, firing_num_t: {firing_num_t}, fire_rate: {firing_rate}")
        exit()

    logger = get_logger(log_file_name + '.log')
    logger.info('start training!')

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):
        # pdb.set_trace()
        start_time = time.time()
        loss, train_firing_loss, acc, train_firing_num, train_firing_num_t, train_firing_rate = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f'Epoch:[{epoch}/{args.epochs}]\t loss={loss:.5f}\t firing_rate={train_firing_rate}\t acc={acc:.4f}\t training_time={training_time} s')
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar('train_firing_num', train_firing_num, epoch)
        writer.add_scalar('train_firing_rate', train_firing_rate, epoch)
        for i in range(train_firing_num_t.shape[0]):
            writer.add_scalar(f'train_firing_num_t{i}', train_firing_num_t[i], epoch)
        scheduler.step()

        facc, test_firing_num, test_firing_num_t, test_firing_rate = test(parallel_model, test_loader, device, epoch)
        logger.info('Epoch:[{}/{}]\t Test acc={:.4f}'.format(epoch, args.epochs, facc))
        writer.add_scalar('test_acc', facc, epoch)
        writer.add_scalar('test_firing_num', test_firing_num, epoch)
        writer.add_scalar('test_firing_rate', test_firing_rate, epoch)
        for i in range(test_firing_num_t.shape[0]):
            writer.add_scalar(f'test_firing_num_t{i}', test_firing_num_t[i], epoch)

        save_max = False
        if best_acc < facc:
            best_acc = facc
            fr_ba = test_firing_rate
            save_max = True
            best_epoch = epoch + 1
            # torch.save(parallel_model.module.state_dict(), 'VGGSNN_woAP.pth')
        logger.info('Best Test acc={:.3f}, fire_rate={:.3f}, epoch={}'.format(best_acc, fr_ba, best_epoch))
        print('\n')

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'fr_ba': test_firing_rate
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

    print(log_file_name)

