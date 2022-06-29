import flax.linen as nn
import einops


class LeNet(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, act=False):
        x = nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), nn.relu,
                nn.Dense(100), nn.relu,
            ]
        )(x)
        if act:
            return x
        return nn.softmax(nn.Dense(self.classes)(x))


class CNN(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, act=False):
        x = nn.Sequential(
            [
                nn.Conv(32, (3, 3)),
                nn.Conv(64, (3, 3)),
                lambda x: nn.max_pool(x, (2, 2), strides=2),
                nn.relu,
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(100), nn.relu,
            ]
        )(x)
        if act:
            return x
        return nn.softmax(nn.Dense(self.classes)(x))
