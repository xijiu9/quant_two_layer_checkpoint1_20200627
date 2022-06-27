import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--net', type=str, default='qat', help='Type of Net of cifar10')
args = parser.parse_args()

for i in [4, 5, 6, 7, 8]:
    os.system("python train_cifar.py --net {} --bbits {}".format(args.net, i))
