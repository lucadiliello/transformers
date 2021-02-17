steps = 20

from torch.optim.adamw import AdamW
from transformers_lightning.schedulers import PolynomialLayerwiseDecaySchedulerWithWarmup
from torch.optim import Adam
import torch

def _get_layer_lrs(n_layers):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = []
    key_to_depths.append({"depth": 0, 'params': torch.nn.Linear(1,1).parameters()})
    key_to_depths.append({"depth": 4, 'params': torch.nn.Linear(1,1).parameters()})

    return key_to_depths

group = _get_layer_lrs(2)

sched = PolynomialLayerwiseDecaySchedulerWithWarmup(
        Adam(group, lr=1.0),
        num_training_steps=steps,
        end_learning_rate=0.2,
        lr_decay_power=1.2,
        layerwise_lr_decay_power=0.8,
        cycle=False,
        warmup_steps=5
)

print("Depths:", sched.depths)
print("Lrs:", sched.base_lrs)



for i in range(steps):
    print(sched.get_last_lr(), sched.last_epoch)
    sched.step()


print("\n\n\n\nOptimizers comparison")


print("Pytorch version")

import sys
sys.path.append('..')
from optim import ElectraAdamW

net = torch.nn.Linear(10, 10, bias=True)
net.weight.data.fill_(1.0)
net.bias.data.fill_(0.0)

opt = ElectraAdamW(net.parameters(), lr=1.0, weight_decay=0.1, amsgrad=False)

for i in range(steps):
    opt.zero_grad()
    data = torch.tensor([[2.0]*10]*2)
    loss = net(data).sum()
    loss.backward()
    opt.step()
    print(loss)
