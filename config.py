import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
    parser.add_argument('--data', default='../WebVision/dataset/',
                        help='path to WebVision dataset')
    parser.add_argument('--exp-dir', default='exp', type=str,
                        help='experiment directory')  # 实验记录路径
    parser.add_argument('--data_type', type=str, default="bird",
                        help='data type')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[20, 40, 60], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_true', default=True,
                        help='use cosine lr schedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='exp', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num_class', default=1000, type=int)
    parser.add_argument('--low-dim', default=128, type=int,
                        help='embedding dimension')
    parser.add_argument('--moco_queue', default=8192, type=int,
                        help='queue size; number of negative samples')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='momentum for updating momentum encoder')
    parser.add_argument('--proto_m', default=0.999, type=float,
                        help='momentum for computing the moving average of prototypes')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='contrastive temperature')
    parser.add_argument('--w_inst', default=1, type=float,
                        help='weight for instance contrastive loss')
    parser.add_argument('--w_proto', default=1, type=float,
                        help='weight for prototype contrastive loss')

    parser.add_argument('--start_clean_epoch', default=100, type=int,
                        help='epoch to start noise cleaning')
    parser.add_argument('--pseudo_th', default=0.75, type=float,
                        help='threshold for pseudo labels')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='weight to combine model prediction and prototype prediction')
    args = parser.parse_args()

    return args
