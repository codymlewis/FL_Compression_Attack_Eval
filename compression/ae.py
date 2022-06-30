"""
Autoencoder compression scheme from `https://arxiv.org/abs/2108.05670 <https://arxiv.org/abs/2108.05670>`_
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


def mseloss(net):

    @jax.jit
    def _apply(params, x):
        z = net.apply(params, x)
        return jnp.mean(0.5 * (x - z)**2)

    return _apply


def _update(opt, loss):

    @jax.jit
    def _apply(params, opt_state, x):
        grads = jax.grad(loss)(params, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return _apply

class AutoEncoder(nn.Module):
    input_len: int

    def setup(self):
        self.encoder = nn.Sequential([
            nn.Dense(64), nn.relu,
            nn.Dense(32), nn.relu,
            nn.Dense(16), nn.relu,
        ])
        self.decoder = nn.Sequential([
            nn.Dense(16), nn.relu,
            nn.Dense(32), nn.relu,
            nn.Dense(64), nn.relu,
            nn.Dense(self.input_len), nn.sigmoid,
        ])

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

# Autoencoder compression


class Coder:
    """Store the per-endpoint autoencoders and associated variables."""

    def __init__(self, gm_params, num_clients):
        """
        Construct the Coder.
        Arguments:
        - gm_params: the parameters of the global model
        - num_clients: the number of clients connected to the associated controller
        """
        gm_params = jax.flatten_util.ravel_pytree(gm_params)[0]
        param_size = len(gm_params)
        model = AutoEncoder(param_size)
        self.model = model
        loss = mseloss(model)
        opt = optax.adam(1e-3)
        self.updater = _update(opt, loss)
        params = model.init(jax.random.PRNGKey(0), gm_params)
        self.params = [params for _ in range(num_clients)]
        self.opt_states = [opt.init(params) for _ in range(num_clients)]
        self.datas = [[] for _ in range(num_clients)]
        self.num_clients = num_clients

    def encode(self, grad, i):
        """Encode the updates of the client i."""
        return self.model.apply(self.params[i], grad, method=self.model.encode)

    def decode(self, all_grads):
        """Decode the updates of the clients."""
        return jnp.array([
            self.model.apply(self.params[i], grad, method=self.model.decode)
            for i, grad in enumerate(all_grads)
        ])

    def add_data(self, grad, i):
        """Add the updates of the client i to the ith dataset."""
        self.datas[i].append(grad)

    def update(self, i):
        """Update the ith client's autoencoder."""
        grads = jnp.array(self.datas[i])
        self.params[i], self.opt_states[i] = self.updater(self.params[i], self.opt_states[i], grads)
        self.datas[i] = []


class Encode:
    """Encoding update transform."""

    def __init__(self, coder):
        """
        Construct the encoder.
        
        Arguments:
        - coder: the autoencoders used for compression
        """
        self.coder = coder

    def __call__(self, all_grads):
        encoded_grads = []
        for i, g in enumerate(all_grads):
            self.coder.add_data(g, i)
            self.coder.update(i)
            encoded_grads.append(self.coder.encode(g, i))
        return encoded_grads


class Decode:
    """Decoding update transform."""

    def __init__(self, params, coder):
        """
        Construct the decoder.
        
        Arguments:
        - params: the parameters of the global model, used for structure information
        - coder: the autoencoders used for decompression
        """
        self.coder = coder

    def __call__(self, all_grads):
        return self.coder.decode(all_grads)
