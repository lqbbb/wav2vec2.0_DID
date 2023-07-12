import torch
import dateutil.tz
import time
import logging
import os


import numpy as np
from sklearn.metrics import roc_curve
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple

import copy

plt.switch_backend('agg')

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_pretrained_weights(model, checkpoint):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    checkpoint_file = torch.load(checkpoint)
    pretrain_dict = checkpoint_file['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value  计算并存储平均值和当前值"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()     #  reset()代表初始化

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def compute_eer(distances, labels):
    # Calculate evaluation metrics
    fprs, tprs, _ = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return eer


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k  计算指定k值的k个top预测的准确性"""
    with torch.no_grad():   #评估阶段我们不需要梯度
        maxk = max(topk)   #maxk = max((1,))  取top1准确率，若取top1和top5准确率改为max((1,5))
        # size函数：总元素的个数
        batch_size = target.size(0)   #数组size第一个元素的值赋给变量


       # '''
       # #torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
       #  input：一个tensor数据
       #  k：指明是得到前k个数据以及其index
       #  dim： 指定在哪个维度上排序， 默认是最后一个维度
       #  largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
       #  sorted：返回的结果按照顺序返回
       #  out：可缺省，不要
       # '''
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置
        #print("pred=",pred.shape)    #pred= torch.Size([5, 256])
        #view()函数作用是将一个多行的Tensor,拼接成一行
        #.expand_as作用是将输入tensor的维度扩展为与指定tensor相同的size
        #输出最大值的索引位置，这个索引位置和真实值的索引位置比较相等的做统计
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print("correct=", correct.shape)


        res = []
        for k in topk:
            #correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)  #自己加了.contiguous()
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_logger(log_dir, phase='train'):
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    # log_file = '{}_{}.log'.format(time_str, phase)

    log_file = '{}.log'.format(phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'


     # 清理已有 handlers (本来并行的) 参看 (python logging 模块配置咋不起作用了？)
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)


    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def set_path(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)   #没有GPU
    # mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if torch.isnan(ret).sum():
        import ipdb
        ipdb.set_trace()
        raise OverflowError(f'gumbel softmax output: {ret}')
    return ret

#后面自己加的
#
# def pad_list(xs, pad_value):
#     # From: espnet/src/nets/e2e_asr_th.py: pad_list()
#     n_batch = len(xs)
#
#     if model.train():
#         max_len = 1000
#     else:
#         max_len = max(x.size(0) for x in xs)   #这个如果batch够大，就应该不会出现问题
#
#     pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
#     for i in range(n_batch):
#         pad[i, :xs[i].size(0)] = xs[i]
#     return pad



def pad_list(xs, pad_value):

    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)

    max_len = 1000
    # max_len = max(x.size(0) for x in xs)   #这个如果batch够大，就应该不会出现问题
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
        # print("pad=",pad[i, :xs[i].size(0)].shape)
    return pad



def assign_params_dict(default_params:dict, params:dict, force_check=False, support_unknow=False):
    default_params = copy.deepcopy(default_params)
    default_keys = set(default_params.keys())

    # Should keep force_check=False to use support_unknow
    if force_check:
        for key in params.keys():
            if key not in default_keys:
                raise ValueError("The params key {0} is not in default params".format(key))

    # Do default params <= params if they have the same key
    params_keys = set(params.keys())
    for k, v in default_params.items():
        if k in params_keys:
            if isinstance(v, type(params[k])):
                if isinstance(v, dict):
                    # To parse a sub-dict.
                    sub_params = assign_params_dict(v, params[k], force_check, support_unknow)
                    default_params[k] = sub_params
                else:
                    default_params[k] = params[k]
            elif isinstance(v, float) and isinstance(params[k], int):
                default_params[k] = params[k] * 1.0
            elif v is None or params[k] is None:
                default_params[k] = params[k]
            else:
                raise ValueError("The value type of default params [{0}] is "
                "not equal to [{1}] of params for k={2}".format(type(default_params[k]), type(params[k]), k))

    # Support unknow keys
    if not force_check and support_unknow:
        for key in params.keys():
            if key not in default_keys:
                default_params[key] = params[key]

    return default_params


