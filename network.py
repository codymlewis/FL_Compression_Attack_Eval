import numpy as np
import ymir


class Network(ymir.utils.network.Network):
    def __init__(self, C=1.0):
        super().__init__(C)
        self.update_transforms = []

    def add_update_transform(self, transform):
        self.update_transforms.append(transform)

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        updates, losses, data = super().__call__(params, rng, return_weights)
        for transform in self.update_transforms:
            updates = transform(updates)
        return updates, losses, data
