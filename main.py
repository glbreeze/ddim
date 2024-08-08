
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from loss import *
from data import *
from networks import UNet, PrecondUnet
from torch.utils.data import DataLoader
from generate import sampling

# K: num of Gaussian components
# d: rank of U
# n: dim of x

K=2; d=1; n=2
N = 500
batch_size = 32
arch = 'prec_unet'
num_epoch = 100

device = "cuda" if torch.cuda.is_available() else 'cpu'

# ==================== load data ====================
train_set = MoGData(K=K, d=d, n=n, N=N, alpha=1)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# =============== construct network ==================
if arch == 'unet':
    model = UNet(in_dim=n, embed_dim=32)
elif arch == 'prec_unet':
    model = PrecondUnet(in_dim=n, embed_dim=32)
model = model.to(device)

# ================= setup optimizer ==================
loss_fn = EDMLoss(P_mean=-1.5, P_std=0.25)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epoch)

# ================== start training ==================

for epoch in range(num_epoch):
    epoch_loss, steps = 0, 0
    model.train()
    for i, batch in enumerate(train_loader):
        input, label = batch
        input = input.to(device)

        loss = loss_fn(model, input).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        steps += 1
    print(f"Epoch {epoch} loss = {epoch_loss / steps:.4f}")
    scheduler.step()


# ================== sampling ==================
N_test = 100
sampler_kwargs = {}
seeds = list(np.arange(100))
gen_samples = sampling(model, list(np.arange(N_test)), 10, device=device, **sampler_kwargs)

# new testing data
test_set = MoGData(K=K, d=d, n=n, N=N_test, alpha=1, U_dt=train_set.U_dt)
test_samples = torch.tensor(test_set.inputs)

from geomloss import SamplesLoss
Loss = SamplesLoss("energy")

# 3D tensors
distance = Loss(gen_samples, test_samples).item()

print('Given size of training sample {}, distance between generated sample and test samples: {:.4f}'.format(N, distance))

