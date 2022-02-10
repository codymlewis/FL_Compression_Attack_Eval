"""
Evaluation of heterogeneous techniques applied to viceroy.
"""

import haiku as hk
import hkzoo
import jax
import logging
import numpy as np
import optax
import tenjin
from tqdm import trange
import ymir


def main():
    dataset_name = "mnist"  # "mnist" or "kddcup99"
    aggregation = "fedavg"  # ["contra", "flame", "fedavg"]
    attack = "labelflip"  # "labelflip" or "onoff"
    attack_from, attack_to = (0, 11) if dataset_name == "kddcup99" else (0, 1)
    percent_adv = 0.0  # 0.3 or 0.5
    compression = "ae"  # ["fedmax", "fedprox", "fedzip", "ae"]
    local_epochs = 1
    total_rounds = 500

    print(f"Running a {compression}-{aggregation} system on {dataset_name} with {percent_adv:.0%} {attack} adversaries.")
    print("Setting up the system...")
    num_clients = 10
    num_adversaries = int(num_clients * percent_adv)
    num_honest = num_clients - num_adversaries
    rng = np.random.default_rng(0)

    agg_kwargs = {}
    if aggregation == "contra":
        agg_kwargs['k'] = num_adversaries
    elif aggregation == "flame":
        agg_kwargs['eps'] = 3758

    # Setup the dataset
    dataset = ymir.mp.datasets.Dataset(*tenjin.load(dataset_name))
    batch_sizes = [8 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x)))
    if compression == "fedprox":
        client_opt = ymir.mp.optimizers.pgd(optax.sgd(0.1), 0.01, local_epochs=local_epochs)
    else:
        client_opt = optax.sgd(0.1)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    if compression == "fedmax":
        net_act = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x, True)))
        loss = ymir.mp.losses.fedmax_loss(net, net_act, dataset.classes)
    else:
        loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_honest):
        network.add_host("main", ymir.regiment.Scout(client_opt, client_opt_state, loss, data[i], local_epochs))
    for i in range(num_adversaries):
        c = ymir.regiment.Scout(client_opt, client_opt_state, loss, data[i + num_honest], local_epochs)
        ymir.fritz.labelflipper.convert(c, dataset, attack_from, attack_to)
        if attack == "onoff":
            ymir.fritz.onoff.convert(c)
        network.add_host("main", c)

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    if attack == "onoff":
        network.get_controller("main").add_update_transform(
            ymir.fritz.onoff.GradientTransform(
                params, server_opt, server_opt_state, network, getattr(ymir.garrison, aggregation),
                network.clients[-num_adversaries:], 1/num_clients if aggregation == "fedavg" else 1, False,
                **agg_kwargs
            )
        )
    if compression == "fedzip":
        network.get_controller("main").add_update_transform(lambda g: ymir.mp.compression.fedzip.encode(g, False))
        network.get_controller("main").add_update_transform(ymir.mp.compression.fedzip.Decode(params))
    elif compression == "ae":
        coder = ymir.mp.compression.ae.Coder(params, num_clients)
        network.get_controller("main").add_update_transform(ymir.mp.compression.ae.Encode(coder))
        network.get_controller("main").add_update_transform(ymir.mp.compression.ae.Decode(params, coder))
    model = getattr(ymir.garrison, aggregation).Captain(params, server_opt, server_opt_state, network, rng, **agg_kwargs)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(total_rounds)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'], {'from': attack_from, 'to': attack_to, 'datasets': ['test']})
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}", 'ASR': f"{results['test asr']:.3f}"})
        model.step()


if __name__ == "__main__":
    main()