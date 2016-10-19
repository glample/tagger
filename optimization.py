import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX
device = theano.config.device


class Optimization:

    def __init__(self, clip=None):
        """
        Initialization
        """
        self.clip = clip

    def get_gradients(self, cost, params):
        """
        Compute the gradients, and clip them if required.
        """
        if self.clip is None:
            return T.grad(cost, params)
        else:
            assert self.clip > 0
            return T.grad(
                theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
                params
            )

    def get_updates(self, method, cost, params, *args, **kwargs):
        """
        Compute the updates for different optimizers.
        """
        if method == 'sgd':
            updates = self.sgd(cost, params, **kwargs)
        elif method == 'sgdmomentum':
            updates = self.sgdmomentum(cost, params, **kwargs)
        elif method == 'adagrad':
            updates = self.adagrad(cost, params, **kwargs)
        elif method == 'adadelta':
            updates = self.adadelta(cost, params, **kwargs)
        elif method == 'adam':
            updates = self.adam(cost, params, **kwargs)
        elif method == 'rmsprop':
            updates = self.rmsprop(cost, params, **kwargs)
        else:
            raise("Not implemented learning method: %s" % method)
        return updates

    def sgd(self, cost, params, lr=0.01):
        """
        Stochatic gradient descent.
        """
        lr = theano.shared(np.float32(lr).astype(floatX))

        gradients = self.get_gradients(cost, params)

        updates = []
        for p, g in zip(params, gradients):
            updates.append((p, p - lr * g))

        return updates

    def sgdmomentum(self, cost, params, lr=0.01, momentum=0.9):
        """
        Stochatic gradient descent with momentum. Momentum has to be in [0, 1)
        """
        # Check that the momentum is a correct value
        assert 0 <= momentum < 1

        lr = theano.shared(np.float32(lr).astype(floatX))
        momentum = theano.shared(np.float32(momentum).astype(floatX))

        gradients = self.get_gradients(cost, params)
        velocities = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, velocity in zip(params, gradients, velocities):
            new_velocity = momentum * velocity - lr * gradient
            updates.append((velocity, new_velocity))
            updates.append((param, param + new_velocity))
        return updates

    def adagrad(self, cost, params, lr=1.0, epsilon=1e-6):
        """
        Adagrad. Based on http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
        """
        lr = theano.shared(np.float32(lr).astype(floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(floatX))

        gradients = self.get_gradients(cost, params)
        gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, gsum in zip(params, gradients, gsums):
            new_gsum = gsum + gradient ** 2.
            updates.append((gsum, new_gsum))
            updates.append((param, param - lr * gradient / (T.sqrt(gsum + epsilon))))
        return updates

    def adadelta(self, cost, params, rho=0.95, epsilon=1e-6):
        """
        Adadelta. Based on:
        http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
        """
        rho = theano.shared(np.float32(rho).astype(floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(floatX))

        gradients = self.get_gradients(cost, params)
        accu_gradients = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]
        accu_deltas = [theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(floatX)) for param in params]

        updates = []
        for param, gradient, accu_gradient, accu_delta in zip(params, gradients, accu_gradients, accu_deltas):
            new_accu_gradient = rho * accu_gradient + (1. - rho) * gradient ** 2.
            delta_x = - T.sqrt((accu_delta + epsilon) / (new_accu_gradient + epsilon)) * gradient
            new_accu_delta = rho * accu_delta + (1. - rho) * delta_x ** 2.
            updates.append((accu_gradient, new_accu_gradient))
            updates.append((accu_delta, new_accu_delta))
            updates.append((param, param + delta_x))
        return updates

    def adam(self, cost, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam. Based on http://arxiv.org/pdf/1412.6980v4.pdf
        """
        updates = []
        gradients = self.get_gradients(cost, params)

        t = theano.shared(np.float32(1.).astype(floatX))

        for param, gradient in zip(params, gradients):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m = beta1 * m_prev + (1. - beta1) * gradient
            v = beta2 * v_prev + (1. - beta2) * gradient ** 2.
            m_hat = m / (1. - beta1 ** t)
            v_hat = v / (1. - beta2 ** t)
            theta = param - (lr * m_hat) / (T.sqrt(v_hat) + epsilon)

            updates.append((m_prev, m))
            updates.append((v_prev, v))
            updates.append((param, theta))

        updates.append((t, t + 1.))
        return updates

    def rmsprop(self, cost, params, lr=0.001, rho=0.9, eps=1e-6):
        """
        RMSProp.
        """
        lr = theano.shared(np.float32(lr).astype(floatX))

        gradients = self.get_gradients(cost, params)
        accumulators = [theano.shared(np.zeros_like(p.get_value()).astype(floatX)) for p in params]

        updates = []

        for param, gradient, accumulator in zip(params, gradients, accumulators):
            new_accumulator = rho * accumulator + (1 - rho) * gradient ** 2
            updates.append((accumulator, new_accumulator))

            new_param = param - lr * gradient / T.sqrt(new_accumulator + eps)
            updates.append((param, new_param))

        return updates
