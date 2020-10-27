import argparse
from train import train, test

## Parser
parser = argparse.ArgumentParser(description='Training models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=3.16e-5, type=float, dest="lr")
parser.add_argument("--num_epoch", default=10000000, type=int, dest="num_epoch")
parser.add_argument("--lr_update_step", default=215000, type=int, dest="lr_update_step")
parser.add_argument("--batch_size", default=32, type=int, dest="batch_size")
parser.add_argument("--use_gpu", default=False, type=bool, dest="use_gpu")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)