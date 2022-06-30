"""
Evaluation of heterogeneous techniques applied to viceroy.
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pandas as pd
from tqdm import trange
import datasets
import einops
import ymir

import models
import network as network_lib
import compression as compression_lib


def ce_loss(model):

    @jax.jit
    def _loss(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    return _loss


def accuracy_fn(model):
    @jax.jit
    def _apply(params, X, y):
        return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == y)
    return _apply


def asr_fn(model, attack_from, attack_to):
    @jax.jit
    def _apply(params, X, y):
        preds = jnp.where(y == attack_from, jnp.argmax(model.apply(params, X), axis=-1), -1)
        return jnp.sum(preds == attack_to) / jnp.sum(y == attack_from)
    return _apply


def load_dataset(dataset_name):
    ds = datasets.load_dataset(dataset_name)
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


def main(args):
    dataset_name = args.dataset
    aggregation = args.agg
    attack = args.attack
    attack_from, attack_to = (0, 1)
    percent_adv = args.aper
    compression = args.comp
    local_epochs = 10
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
    dataset = ymir.utils.datasets.Dataset(load_dataset(dataset_name))
    batch_sizes = [32 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, ymir.utils.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = models.LeNet(dataset.classes)
    if compression == "fedprox":
        client_opt = ymir.client.fedprox.pgd(optax.sgd(0.1), 0.01, local_epochs=local_epochs)
    else:
        client_opt = optax.sgd(0.1)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    if compression == "fedmax":
        loss_fn = ymir.client.fedmax.loss
    else:
        loss_fn = ce_loss
    global network
    network = network_lib.Network()
    for i in range(num_honest):
        network.add_client(ymir.client.Client(params, client_opt, loss_fn(net.clone()), data[i], local_epochs))
    for i in range(num_adversaries):
        adv_data = dataset.get_iter("train", 32, rng=rng)
        c = ymir.client.Client(params, client_opt, loss_fn(net.clone()), adv_data, local_epochs)
        ymir.attacks.label_flipper.convert(c, attack_from, attack_to)
        # if attack == "onoff":
        #   onoff.convert(c)
        network.add_client(c)

    server_opt = optax.sgd(1)
    # if attack == "onoff":
    #   network.add_update_transform(
    #       onoff.GradientTransform(
    #           params, server_opt, server_opt_state, network, getattr(ymir.server, aggregation),
    #           network.clients[-num_adversaries:], 1/num_clients if aggregation == "fedavg" else 1, False,
    #           **agg_kwargs
    #       )
    #   )
    if compression == "fedzip":
        network.add_update_transform(lambda g: compression_lib.fedzip.encode(g, False))
        network.add_update_transform(compression_lib.fedzip.Decode(params))
    elif compression == "ae":
        coder = compression_lib.ae.Coder(params, num_clients)
        network.add_update_transform(compression_lib.ae.Encode(coder))
        network.add_update_transform(compression_lib.ae.Decode(params, coder))
    server = getattr(ymir.server, aggregation).Server(network, params, opt=server_opt, rng=rng, **agg_kwargs)

    print("Done, beginning training.")
    accuracy = accuracy_fn(net)
    asr = asr_fn(net, attack_from, attack_to)
    asrs, accs = [], []

    # Train/eval loop.
    for r in (pbar := trange(total_rounds)):
        acc_val = accuracy(server.params, *next(test_eval))
        asr_val = asr(server.params, *next(test_eval))
        pbar.set_postfix({'ACC': f"{acc_val:.3f}", 'ASR': f"{asr_val:.3f}"})
        accs.append(acc_val)
        asrs.append(asr_val)
        server.step()

    cur_results = {
        "Dataset": dataset_name,
        "Compression": compression,
        "Aggregation": aggregation,
        "Attack": attack,
        "Adv.": f"{percent_adv:.0%}",
        "Mean ASR": np.array(asrs).mean(),
        "STD ASR": np.array(asrs).std(),
        "Mean ACC": np.array(accs).mean(),
        "STD ACC": np.array(accs).std(),
    }

    result_file = "results.xlsx"
    file_exists = os.path.exists(result_file)
    print(f"Writing results to {result_file}.")
    if file_exists and (compression in pd.ExcelFile(result_file).sheet_names):
        full_results = pd.read_excel(result_file, sheet_name=compression, index_col=0).append(cur_results, ignore_index=True)
    else:
        full_results = pd.DataFrame(cur_results, index=[0])
    with pd.ExcelWriter(result_file, mode='a' if file_exists else 'w', if_sheet_exists='replace' if file_exists else None) as xls:
        full_results.to_excel(xls, sheet_name=compression)
    print("Done. bye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the reiny main experiment.')
    parser.add_argument('--agg', type=str, default="fedavg", help='Aggregation algorithm to use')
    parser.add_argument('--comp', type=str, default="fedmax", help='Compression algorithm to use')
    parser.add_argument('--attack', type=str, default="labelflip", help='Attack to use. Options: onoff, labelflip')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset to use')
    parser.add_argument('--aper', type=float, default=0.3, help='Percentage of adversaries in the network')
    args = parser.parse_args()

    main(args)
