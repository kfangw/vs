# -*- coding:utf-8 -*-
from __future__ import print_function

import argparse
from glob import glob

import torch
import torch.nn.functional as F
from pymongo import MongoClient
from torchvision import transforms

# Training settings
from vs.data import image_loader
from vs.models import resnet

parser = argparse.ArgumentParser(description='PyTorch Visual Shopping')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Visual_Search', type=str,
                    help='name of experiment')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--workers', type=int, default=8, metavar='N',
                    help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--anchor', default='', type=str,
                    help='path to anchor image folder')
parser.add_argument('--feature-size', default=1000, type=int,
                    help='fully connected layer size')
parser.add_argument('--save-db', action='store_true', default=False,
                    help='save inferencing result to mongodb')

best_acc = 0


def vs(*pargs, **kwargs):
    # 기본 설정 부분
    global args, best_acc
    args = parser.parse_args(*pargs)
    data_path = args.data
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    pin = True
    if args.cuda:
        print("GPU Mode")
        torch.cuda.manual_seed(args.seed)
    else:
        pin = False
        print("CPU Mode")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 이미지 로딩
    image_data = torch.utils.data.DataLoader(
        image_loader.ImageFolder(data_path, transforms.Compose([
            transforms.Scale(400),
            transforms.CenterCrop(400),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )

    # model 생성
    model = resnet.resnet18(pretrained=True, feature_size=args.feature_size)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()

    # Inferencing dataset
    print("Analyzing Data")
    paths = glob(data_path + "/*")
    imageN = len(paths)

    inferenced_data = inference(image_data, model)
    # dist = sorted(dist, key = lambda x: x[1])
    print("Analyzed %s images" % imageN)

    # Save result to db
    # if args.save_db:
    #     inf_data = inferenced_data.data.cpu().numpy().tolist()
    #     inf_dict = {paths[idx]: item for idx, item in enumerate(inf_data)}
    #     client = MongoClient('10.214.35.36', 27017)
    #
    #     db = client.db
    #     posts = db.posts
    #     print(inf_dict)
    #     result = posts.insert_one(inf_dict)
    #     print(posts.count())

    anchor_path = args.anchor
    print (anchor_path)
    try:
        anchor_image = torch.utils.data.DataLoader(
            image_loader.ImageFolder(anchor_path, transforms.Compose([
                transforms.Scale(400),
                transforms.CenterCrop(400),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin,
        )

        inferenced_anchor = inference(anchor_image, model)
        inferenced_anchor = inferenced_anchor.expand(imageN, args.feature_size)
        distances = F.pairwise_distance(inferenced_data, inferenced_anchor, 2).data.cpu().numpy().tolist()
        result = []

        for idx, dist in enumerate(distances):
            result.append((paths[idx], dist))

        result = sorted(result, key=lambda x: x[1])
        print(result[:3])
        return result
    except:
        print("Please input correct file path")


def inference(image_data, model):
    # Progress bar setting
    counter = 0
    imageN = len(image_data) * args.batch_size
    printProgressBar(counter, imageN, prefix='Progress:', suffix='Complete', length=100)

    # data inference
    inferenced_data = torch.autograd.Variable(torch.randn(1, 1), volatile=True)
    for idx, (path, input) in enumerate(image_data):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        if idx == 0:
            inferenced_data = output
        else:
            inferenced_data = torch.cat([inferenced_data, output], 0)

        counter = counter + args.batch_size
        printProgressBar(counter, imageN, prefix='Progress:', suffix='Complete', length=100)

    return inferenced_data


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
