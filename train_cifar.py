import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_cifar.')


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayersweight', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')
parser.add_argument('--awbits', type=int, default=8, help='weight number of bits')
parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'gradually', 'checkpoint_from_zero'])
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
args = parser.parse_args()
method = 'twolayer'
model = "results/{}/models/checkpoint-195.pth.tar".format(method)
workplace = "results/" + method
if not os.path.exists(workplace):
    os.mkdir(workplace)

# if args.net == 'qat':
#     arg = "-c quantize --qa=True --qw=True --qg=False"
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
# elif args.net == 'ptq':

arg = " -c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False"

if args.lsqforward:
    arg = " -c quantize --qa=True --qw=True --qg=False --persample=False --hadamard=False"
    method = 'lsq/lsq_' + str(args.awbits)
    model = "results/{}/models/checkpoint-93.3.pth.tar".format(method)
    workplace = "results/" + method

if args.training_strategy == 'checkpoint' or args.training_strategy == "checkpoint_from_zero":
    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--twolayersweight {} --lsqforward {} {} ~/data/cifar10 --abits {} --wbits {} --weight-decay {} "
              "--training-strategy {} "
              "--resume {}".format(method,
                                   args.twolayersweight,
                                   args.lsqforward, arg,
                                   args.awbits,
                                   args.awbits,
                                   args.weight_decay,
                                   args.training_strategy,
                                   model))
else:
    os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
              "--twolayersweight {} --lsqforward {} {} ~/data/cifar10 --abits {} --wbits {} --weight-decay {} "
              "--training-strategy {}".format(method, args.twolayersweight,
                                              args.lsqforward, arg, args.awbits,
                                              args.awbits, args.weight_decay,
                                              args.training_strategy))
    # {} ~/data/cifar10 --resume {}".format(args.twolayersweight, args.lsqforward, method, arg, model))

# elif args.net == 'psq':
#     arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits={}".format(args.bbits)
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))
#               # "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
#
#
# elif args.net == 'bhq':
#     arg = "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits={}".format(args.bbits)
#
#     os.system("python main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace results/{} "
#               "--lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 "
#               "--weight-decay 1e-4 {} ~/data/cifar10 --resume {}".format(method, arg, model))
#               # "--weight-decay 1e-4 {} ~/data/cifar10".format(method, arg))
