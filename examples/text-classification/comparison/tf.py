steps = 20
import re

#########

print("Schedulers comparison")

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


print("\n\n\n\n\nOptimizers comparison")
print("TF version")
class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def _apply_gradients(self, grads_and_vars, learning_rate):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self.weight_decay_rate > 0:
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param

      update_with_lr = learning_rate * update
      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return assignments

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if isinstance(self.learning_rate, dict):
      key_to_grads_and_vars = {}
      for grad, var in grads_and_vars:
        update_for_var = False
        for key in self.learning_rate:
          if key in var.name:
            update_for_var = True
            if key not in key_to_grads_and_vars:
              key_to_grads_and_vars[key] = []
            key_to_grads_and_vars[key].append((grad, var))
        if not update_for_var:
          raise ValueError("No learning rate specified for variable", var)
      assignments = []
      for key, key_grads_and_vars in key_to_grads_and_vars.items():
        assignments += self._apply_gradients(key_grads_and_vars,
                                             self.learning_rate[key])
    else:
      assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

weight_initer = tf.constant(1., shape=[10,10], dtype=tf.float32)
input_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)
lin = tf.keras.layers.Dense(10, input_shape=(10,), kernel_initializer='ones',
    bias_initializer='zeros')
loss = tf.math.reduce_sum(
    lin(input_)
)

opt = AdamWeightDecayOptimizer(
    learning_rate=1.0,
    weight_decay_rate=0.1
)

solver = opt.minimize(loss, var_list=[lin.trainable_weights])

sess = tf.Session()
init = tf.global_variables_initializer()   
sess.run(init)


for i in range(steps):
    print(sess.run([loss, solver], feed_dict={input_: [[2.0]*10]*2 }))

