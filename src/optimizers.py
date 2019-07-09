import re

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class AdamWeightDecayOptimizer(tf.keras.optimizers.Adam):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 amsgrad=False,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer",
                 **kwargs):
        """Constructs a AdamWeightDecayOptimizer."""

        super(AdamWeightDecayOptimizer, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        weight_decay_rate_t = math_ops.cast(self.weight_decay_rate, var.dtype)

        if not self.amsgrad:
            return self.update(v_t, m_t, epsilon_t, var, weight_decay_rate_t, lr)
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(var,
                                              lr * m_t / (v_hat_sqrt + epsilon_t),
                                              use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def update(self, v_t, m_t, epsilon_t, var, weight_decay_rate_t, lr):
        v_sqrt = math_ops.sqrt(v_t)

        update = m_t / (v_sqrt + epsilon_t),

        if self._do_use_weight_decay(var.name):
            update += math_ops.multiply(var, weight_decay_rate_t)

        var_update = math_ops.subtract(var, lr * update)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class PolynomialDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a polynomial decay schedule with additional warmup steps."""

    def __init__(self,
                 initial_learning_rate,
                 grow_steps,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 name=None):
        super(PolynomialDecayWithWarmup, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.grow_steps = grow_steps
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "PolynomialDecayWithWarmup") as name:
            initial_learning_rate = ops.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)
            power = math_ops.cast(self.power, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            decay_steps_recomp = math_ops.cast(self.decay_steps, dtype)
            grow_steps_recomp = math_ops.cast(self.grow_steps, dtype)
            if self.cycle:
                # Find the first multiple of decay_steps that is bigger than
                # global_step. If global_step is zero set the multiplier to 1
                multiplier = control_flow_ops.cond(
                    math_ops.equal(global_step_recomp, 0), lambda: 1.0,
                    lambda: math_ops.ceil(global_step_recomp / self.decay_steps))
                decay_steps_recomp = math_ops.multiply(decay_steps_recomp, multiplier)
            else:
                # Make sure that the global_step used is not bigger than decay_steps.
                global_step_recomp = math_ops.minimum(global_step_recomp, self.decay_steps)

            def grow_lr():
                global_steps_int = math_ops.cast(global_step_recomp, tf.int32)
                warmup_steps_int = tf.constant(self.grow_steps, dtype=tf.int32)

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                warmup_percent_done = math_ops.divide(global_steps_float, warmup_steps_float)

                warmup_learning_rate = math_ops.multiply(self.initial_learning_rate, warmup_percent_done, name=name)

                return warmup_learning_rate

            def decay_lr():
                p = math_ops.div(global_step_recomp, decay_steps_recomp)
                return math_ops.add(
                    math_ops.multiply(initial_learning_rate - end_learning_rate, math_ops.pow(1 - p, power)),
                    end_learning_rate)

            return tf.cond(tf.less_equal(global_step_recomp, grow_steps_recomp), true_fn=grow_lr, false_fn=decay_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "grow_steps": self.grow_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name
        }
