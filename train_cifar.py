import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--net', type=str, default='qat', help='Type of Net of cifar10')
parser.add_argument('--bbits', type=int, default=8, help='backward number of bits')
args = parser.parse_args()
method = args.net + '/' + args.net + '_' + str(args.bbits)
model = "results/{}/models/checkpoint-100.pth.tar".format(method)
workplace = "results/" + method
if not os.path.exists(workplace):
    os.mkdir(workplace)

if args.net == 'qat':
    arg = "-c quantize --qa=True --qw=True --qg=False"

    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
              "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
elif args.net == 'ptq':
    arg = "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False --bbits={}".format(args.bbits)

    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
              "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
              # "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))

elif args.net == 'psq':
    arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits={}".format(args.bbits)

    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
              "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
              # "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))

elif args.net == 'bhq':
    arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits={}".format(args.bbits)

    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
              "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
              # "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))




