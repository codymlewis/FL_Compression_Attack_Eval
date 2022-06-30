"""
Federate learning on-off toggle attack, this is not an attack by itself but it toggles an already existing attack.
"""

from functools import partial

import jax
import numpy as np

import ymir


class GradientTransform:
    """
    Gradient transform for determining when to toggle the attack.
    """

    def __init__(
        self,
        params,
        opt,
        network,
        alg,
        adversaries,
        max_alpha,
        sharp,
        beta=1.0,
        gamma=0.85,
        timer=False,
        rng=np.random.default_rng(),
        **kwargs
    ):
        """
        Construct the gradient transform.
        Arguments:
        - params: the parameters of the starting model
        - opt: the optimizer to use
        - network: the network of the FL environment
        - alg: the FL aggregation algorithm to use
        - adversaries: the list of adversaries
        - max_alpha: the maximum value of the scale function from aggregation
        - sharp: whether to use the sharp version of the toggle
        - beta: the beta parameter for the toggle
        - gamma: the gamma parameter for the toggle
        - timer: whether to use a timer to determine when to toggle
        - rng: the random number generator to use
        """
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma
        self.server = alg.Server(network, params, opt=opt, rng=rng, **kwargs)
        self.adversaries = adversaries
        self.num_adv = len(adversaries)
        self.timer_mode = timer
        if timer:
            self.timer = 0

    def should_toggle(self, alpha):
        """Return whether the attack should be toggled based on the current scale value."""
        if self.timer_mode:
            self.timer += 1
            if self.timer % 30 == 0:
                return True
            return False
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, all_updates):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        self.server.update(all_updates)
        alpha = self.server.scale(all_updates)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.adversaries:
                a.toggle()
        return all_updates


def convert(client, opt, loss):
    """Convert a client into an on-off toggle adversary."""
    client.shadow_step = client.step
    client._train_step = ymir.client.client.train_step(opt, loss)
    client.toggle = toggle.__get__(client)


def toggle(self):
    """Toggle the attack."""
    self._train_step, self.shadow_step = self.shadow_step, self.step
