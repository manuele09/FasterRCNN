import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

n = 100
net = nn.Conv2d(3, 3, 1)
opt = SGD(net.parameters(), 0.1)
warmup_factor = 0.1234
warmup_iters = 345
s = torch.optim.lr_scheduler.LinearLR(opt, start_factor=warmup_factor, total_iters=warmup_iters)

ckpt = "s.pt"

s_lr = []
for _ in range(s.last_epoch, n):
    opt.step()
    lr = s.get_last_lr()
    s_lr.extend(lr)
    s.step()
    if _ == 10:
        torch.save(s.state_dict(), ckpt)
        print("Epoch 10")
        print(f"Last learing rate: {s.get_last_lr()}, start_factor: {s.start_factor}, end_factor: {s.end_factor}, total_iters: {s.total_iters}, last_epoch: {s.last_epoch}")
        break


plt.plot(s_lr)
opt_sd = opt.state_dict()


s_lr = np.array(s_lr)

# Method 1
n = 100
net = nn.Conv2d(3, 3, 1)
opt = SGD(net.parameters(), 0.1)
opt.load_state_dict(opt_sd)

ckpt = "s.pt"
state_dict = torch.load(ckpt)

warmup_factor = 1.0 / 1000
warmup_iters = min(1000, 100 - 1)

# s = torch.optim.lr_scheduler.LinearLR(opt, start_factor=warmup_factor, total_iters=warmup_iters, last_epoch=state_dict["last_epoch"]-1)
s = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, total_iters=warmup_iters)
print(state_dict["start_factor"])
print(state_dict["total_iters"])
for i in range(0, state_dict["last_epoch"]):
    s.step()
# s = torch.optim.lr_scheduler.ConstantLR(opt, factor=0.1)
print("\nResumed Scheduler")
print(f"Last learing rate: {s.get_last_lr()}, start_factor: {s.start_factor}, end_factor: {s.end_factor}, total_iters: {s.total_iters}, last_epoch: {s.last_epoch}")
s_lr_res = []
print(opt.state_dict)
for _ in range(s.last_epoch, n):
    opt.step()
    lr = s.get_last_lr()
    s_lr_res.extend(lr)
    # print(lr)
    s.step()


s_lr_res = np.array(s_lr_res)
# plt.plot(np.arange(100), s_lr)
plt.plot(np.arange(100), np.concatenate((np.zeros(11), s_lr_res)))
plt.show()

print(s_lr[11:] - s_lr_res)