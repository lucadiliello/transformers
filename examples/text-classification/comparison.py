steps = 20

#########


import tensorflow.compat.v1 as tf


def _get_layer_lrs(learning_rate, layer_decay, n_layers=2):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = {}
    
    key_to_depths["layer_0/"] = 0
    key_to_depths["layer_1/"] = 4

    print(key_to_depths)

    return {
        key: learning_rate * (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }

def create_optimizer(
    learning_rate, num_train_steps,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None):
    """Creates an optimizer and training op."""
    
    global_step = tf.train.get_or_create_global_step()
    increment_global_step_op = tf.assign(global_step, global_step+1)

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.2,
        power=lr_decay_power,
        cycle=False)

    warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)

    learning_rate *= tf.minimum(
        1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

    if layerwise_lr_decay_power > 0:
        learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power, n_layers=n_transformer_layers)

    return learning_rate, increment_global_step_op, global_step



learning_rate, increment_global_step_op, global_step = create_optimizer(
    tf.constant(1.0),
    num_train_steps=steps,
    warmup_steps=5,
    lr_decay_power=1.2,
    layerwise_lr_decay_power=0.8,
    n_transformer_layers=2
)

sess = tf.Session()
init = tf.global_variables_initializer()   
sess.run(init)

for i in range(steps):
    print(sess.run([learning_rate, global_step]))
    sess.run(increment_global_step_op)



###################################################################################################
print("\n\n\n")

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

