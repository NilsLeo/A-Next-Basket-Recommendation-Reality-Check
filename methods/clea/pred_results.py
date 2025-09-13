#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch


def configure_clea_args(dataset: str, foldk: int, num_users: int, num_product: int):
    """Patch module.config.args so Config() picks up our parameters."""
    import module.config as cfg_mod
    argv = [
        "pred_results",
        "--dataset", dataset,
        "--foldk", str(foldk),
        "--num_users", str(num_users),
        "--num_product", str(num_product),
        "--log_fire", "cleamodel",
        "--embedding_dim", "64",
        "--temp_learn", "0",
        "--temp", "10",
    ]
    parsed = cfg_mod.parser.parse_args(argv[1:])
    cfg_mod.args = parsed
    return cfg_mod.Config()


def load_csv_dims(dataset: str, repo_root: str):
    import pandas as pd
    hist = os.path.join(repo_root, "dataset", f"{dataset}_history.csv")
    fut = os.path.join(repo_root, "dataset", f"{dataset}_future.csv")
    df_h = pd.read_csv(hist)
    df_f = pd.read_csv(fut)
    num_users = int(pd.concat([df_h[['user_id']], df_f[['user_id']]]).nunique().iloc[0])
    num_product = int(pd.concat([df_h[['product_id']], df_f[['product_id']]]).max().iloc[0]) + 1
    return num_users, num_product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--foldk', required=True, type=int)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    jsondata_dir = os.path.join(repo_root, "jsondata")
    keyset_dir = os.path.join(repo_root, "keyset")

    history_path = os.path.join(jsondata_dir, f"{args.dataset}_history.json")
    future_path = os.path.join(jsondata_dir, f"{args.dataset}_future.json")
    keyset_path = os.path.join(keyset_dir, f"{args.dataset}_keyset_{args.foldk}.json")

    # Load data
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    with open(future_path, 'r') as f:
        future_data = json.load(f)
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    # Determine dims
    num_users, num_product = load_csv_dims(args.dataset, repo_root)

    # Configure CLEA and build model
    cfg = configure_clea_args(args.dataset, args.foldk, num_users, num_product)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Deferred imports until after args configured
    from module.model_1 import NBModel
    from module.utils import get_batch_TEST_DATASET

    model = NBModel(cfg, device).to(device)

    # Load best checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "models", args.dataset, f"model_1_{args.dataset}_{args.foldk}_cleamodel.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CLEA checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=(None if device.type == 'cuda' else torch.device('cpu')))
    model.load_state_dict(ckpt['model_state_dict'])
    g1_flag = ckpt.get('G1_flag', 1)
    temp0 = float(ckpt.get('temp0', 1.0))

    # Build a minimal weights vector (not used in inference path)
    weights = torch.ones(num_product, dtype=torch.float32, device=device)

    # Iterate test users using utils batching similar to training
    # Manually construct TEST_DATASET batches from JSON like utils.get_dataset does
    # Reconstruct padded S (all history baskets except last), H_pad, and target_basket
    max_basket_size = cfg.max_basket_size

    def pad_basket(b):
        if len(b) < max_basket_size:
            return b + [-1] * (max_basket_size - len(b))
        return b[:max_basket_size]

    test_keys = keyset['test']
    pred_dict = {}
    truth_dict = {}

    model.eval()
    with torch.no_grad():
        for user_id in test_keys:
            seq = history_data[user_id]
            # Padded baskets for all except last
            S = [pad_basket(b) for b in seq[:-1]]
            if len(S) == 0:
                continue
            target_basket = pad_basket(future_data[user_id][1])
            # Flatten S into S_pool (list of item ids length L*B)
            S_pool = []
            for b in S:
                S_pool.extend(b)
            # Build frequency history H (size num_product)
            H = np.zeros(num_product, dtype=np.float32)
            for b in S:
                for it in b:
                    if it >= 0:
                        H[it] += 1
            H = H / max(1, len(S))
            H_pad = np.zeros(num_product + 1, dtype=np.float32)
            H_pad[1:] = H

            # tensors
            uid_tensor = torch.tensor(int(user_id), dtype=torch.long, device=device)
            input_tensor = torch.tensor(S_pool, dtype=torch.long, device=device)
            history_tensor = torch.from_numpy(H_pad[0:2]).to(device)  # as per utils.get_dataset
            tar_b = list(set(target_basket) - {-1})
            if not tar_b:
                continue

            K = len(tar_b)
            target_items_tensor = torch.tensor(tar_b, dtype=torch.long, device=device)
            input_tensor_expand = input_tensor.view(1, -1).expand(K, -1)
            uid_tensor_expand = uid_tensor.expand(K)
            history_tensor_expand = history_tensor.view(1, -1).expand(K, -1)

            # Forward; returns (rest_ratio, repeat_ratio, prob)
            rest_ratio, repeat_ratio, prob = model(temp0, uid_tensor_expand, input_tensor_expand,
                                                   target_items_tensor, weights, history_tensor_expand,
                                                   neg_set_tensor=None, train=False, G1flag=g1_flag, pretrain=0, sd2=cfg.sd2)
            # prob: K x num_product scores; aggregate to a single ranking vector
            scores = prob.detach().float().cpu().numpy()
            agg = scores.mean(axis=0)  # length num_product
            # Produce top-N ranking list
            topk = min(args.topk, int(num_product))
            rank_list = np.argsort(-agg)[:topk].astype(int).tolist()

            pred_dict[user_id] = rank_list
            truth_dict[user_id] = tar_b

    # Output files
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    pred_out = os.path.join(out_dir, f"{args.dataset}_pred{args.foldk}.json")
    truth_out = os.path.join(out_dir, f"{args.dataset}_truth{args.foldk}.json")
    with open(pred_out, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_out, 'w') as f:
        json.dump(truth_dict, f)
    print(f"Saved predictions to {pred_out}")


if __name__ == '__main__':
    main()
