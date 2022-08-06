import time, torch, logging
import matplotlib.pyplot as plt
import numpy as np
from utils.core_utils import reduce_tensor, AverageMeter
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def running(cfg, loader, model, criterion, optimizer):
    # AverageMeter for outputting training information
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    reduced_eva = torch.zeros([1, 2]).cuda()
    num_images = torch.Tensor([0]).cuda()

    # switch to train mode
    model.train()
    # test learning rate
    num = 100
    beta = 0.98
    avg_loss = 0
    lr_mult = (0.1 / cfg['TRAIN']['LR']) ** (1 / num)
    lr = []
    ls = []
    best_loss = 1e9
    batch_num = 0
    optimizer = ScheduledOptim(optimizer)

    # get data_loader
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        torch.cuda.nvtx.range_push("forward")
        outputs = model(input)
        torch.cuda.nvtx.range_pop()

        # compute loss
        torch.cuda.nvtx.range_push("loss")
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        loss = criterion(outputs, target, target_weight)
        # if isinstance(outputs, list):
        #     loss = criterion(outputs[0], target, target_weight)
        #     for output in outputs[1:]:
        #         loss += criterion(output, target, target_weight)
        # else:
        #     output = outputs
        #     loss = criterion(output, target, target_weight)
        torch.cuda.nvtx.range_pop()

        # torch.Tensor默认不需要自动微分
        num_images[0] = input.size(0)
        torch.cuda.nvtx.range_push("evaluation")
        _, avg_acc, cnt, pred = accuracy(outputs[-1].detach().cpu().numpy(), target.detach().cpu().numpy())
        torch.cuda.nvtx.range_pop()

        # The time cost by reduce is nearly 0(e-05)
        torch.cuda.nvtx.range_push("reduce_tensor")
        if cfg.WORLD_SIZE>1:
            reduced_num_images = reduce_tensor(num_images, cfg.WORLD_SIZE)
            reduced_loss = reduce_tensor(loss, cfg.WORLD_SIZE)
            reduced_eva[0][0] = torch.Tensor([avg_acc * cnt])
            reduced_eva[0][1] = cnt
            reduced_eva = reduce_tensor(reduced_eva)
        else:
            reduced_num_images = num_images
            reduced_loss = loss
            reduced_eva[0][0] = torch.Tensor([avg_acc * cnt])
            reduced_eva[0][1] = cnt
        torch.cuda.nvtx.range_pop()

        # compute gradient and do SGD step
        torch.cuda.nvtx.range_push("backward")
        optimizer.zero_grad()
        reduced_loss.backward()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0:
            batch_time.update(time.time() - end)
            losses.update(reduced_loss.item(), reduced_num_images.item())
            if reduced_eva[0][1] == 0:
                acc.update(0, 0)
            else:
                acc.update(reduced_eva[0][0].item() / reduced_eva[0][1].item(), reduced_eva[0][1].item())

            msg = 'Epoch: [{0}/{1}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time,
                speed=cfg.WORLD_SIZE * input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

        batch_num += 1
        avg_loss = beta * avg_loss + (1 - beta) * loss.data.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        if smoothed_loss > 4 * best_loss or optimizer.lr > 0.1:
            break
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        lr.append(optimizer.lr)
        ls.append(smoothed_loss)
        optimizer.set_learning_rate(lr[-1] * lr_mult)
    
    if cfg.RANK==0:
        plt.figure(0)
        plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.plot(np.log(lr), ls)
        plt.show()
        plt.figure(1)
        plt.xlabel('num iterations')
        plt.ylabel('learning rate')
        plt.plot(lr)
        plt.show()


class ScheduledOptim(object):
    '''A wrapper class for learning rate scheduling'''

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr

