#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch

# Reuse modules from sets2sets_new.py by importing its classes and helpers
from sets2sets_new import (
    EncoderRNN_new,
    AttnDecoderRNN_new,
    decoding_next_k_step,
    MAX_LENGTH,
    hidden_size,
    num_layers,
)


def load_data_paths(dataset: str, fold: int):
    base = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(base, "../../jsondata", f"{dataset}_history.json")
    future_path = os.path.join(base, "../../jsondata", f"{dataset}_future.json")
    keyset_path = os.path.join(base, "../../keyset", f"{dataset}_keyset_{fold}.json")
    return history_path, future_path, keyset_path


def load_data(history_path, future_path, keyset_path):
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    with open(future_path, 'r') as f:
        future_data = json.load(f)
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)
    return history_data, future_data, keyset


def build_models(input_size: int, device: torch.device):
    encoder = EncoderRNN_new(input_size, hidden_size, num_layers)
    decoder = AttnDecoderRNN_new(hidden_size, input_size, num_layers, dropout_p=0.1)
    encoder.to(device)
    decoder.to(device)
    return encoder, decoder


def load_best_checkpoints(dataset: str, fold: int, device: torch.device):
    base = os.path.dirname(os.path.abspath(__file__))
    model_version = f"{dataset}{fold}"
    enc_path = os.path.join(base, "models", f"encoder_{model_version}_model_best")
    dec_path = os.path.join(base, "models", f"decoder_{model_version}_model_best")
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        raise FileNotFoundError(f"Best checkpoints not found for {dataset} fold {fold}: {enc_path}, {dec_path}")
    map_loc = None if device.type == 'cuda' else torch.device('cpu')
    encoder = torch.load(enc_path, map_location=map_loc)
    decoder = torch.load(dec_path, map_location=map_loc)
    encoder.to(device)
    decoder.to(device)
    return encoder, decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--fold_id', required=True, type=int)
    parser.add_argument('--topk', type=int, default=400, help='Top-K to extract for ranking list')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional output directory for pred/truth JSONs')
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold_id

    history_path, future_path, keyset_path = load_data_paths(dataset, fold)
    history_data, future_data, keyset = load_data(history_path, future_path, keyset_path)

    input_size = keyset['item_num']
    test_keys = keyset['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best checkpoints trained by sets2sets_new.py
    encoder, decoder = load_best_checkpoints(dataset, fold, device)

    pred_dict = {}
    truth_dict = {}

    # Generate predictions for next basket (k=1) and take a long ranking list
    next_k_step = 1
    activate_codes_num = args.topk  # use top-k picks from the decoder

    for user_id in test_keys:
        inp = history_data[user_id]
        tgt = future_data[user_id]
        # Ensure we have enough steps
        if len(tgt) < 2 + next_k_step:
            continue
        decoded, prob_vecs = decoding_next_k_step(
            encoder, decoder, inp, tgt, input_size, next_k_step, activate_codes_num
        )
        # prob_vecs[0] contains a ranking list (length = topk); ensure plain ints
        rank_list = [int(x) if not isinstance(x, (int,)) else x for x in prob_vecs[0]]
        # Truth is the next basket indices list
        truth_list = tgt[1]
        pred_dict[user_id] = rank_list
        truth_dict[user_id] = truth_list

    # Save alongside other methods for evaluator discovery
    out_base = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir:
        out_base = args.output_dir
        os.makedirs(out_base, exist_ok=True)
    pred_out = os.path.join(out_base, f"{dataset}_pred{fold}.json")
    truth_out = os.path.join(out_base, f"{dataset}_truth{fold}.json")
    with open(pred_out, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_out, 'w') as f:
        json.dump(truth_dict, f)

    print(f"Saved predictions to {pred_out}")


if __name__ == '__main__':
    main()
