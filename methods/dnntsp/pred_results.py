import json
import torch
import sys
import os
from tqdm import tqdm
from utils.metric import evaluate
from utils.data_container import get_data_loader
from utils.load_config import get_attribute
from utils import load_config as cfg_mod
from utils.util import convert_to_gpu, convert_all_data_to_gpu
from train.train_main import create_model
from utils.util import load_model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold_id
    model_path = args.best_model_path
    history_path = f'../../jsondata/{dataset}_history.json'
    future_path = f'../../jsondata/{dataset}_future.json'
    keyset_path = f'../../keyset/{dataset}_keyset_{fold}.json'
    
    pred_path = f'{dataset}_pred{fold}.json'
    truth_path = f'{dataset}_truth{fold}.json'
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    model = create_model()
    model = load_model(model, model_path)
    # Ensure model and its embedding live on the correct device before building dataloader
    model = convert_to_gpu(model)

    def build_loader_for(model_obj):
        return get_data_loader(
            history_path=history_path,
            future_path=future_path,
            keyset_path=keyset_path,
            data_type='test',
            batch_size=1,
            item_embedding_matrix=model_obj.item_embedding
        )

    data_loader = build_loader_for(model)

    model.eval()

    pred_dict = dict()
    truth_dict = dict()
    test_key = keyset['test']
    user_ind = 0
    def infer_loop(model_obj, loader, use_cuda, start_idx=0):
        user_idx = start_idx
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(tqdm(loader)):
            if use_cuda:
                g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                    convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)
            with torch.no_grad():
                pred_data = model_obj(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
            pred_list = pred_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()
            truth_list = truth_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()
            pred_dict[test_key[user_idx]] = pred_list
            truth_dict[test_key[user_idx]] = truth_list
            user_idx += 1
        return user_idx

    try:
        user_ind = infer_loop(model, data_loader, use_cuda=(get_attribute('cuda') != -1 and torch.cuda.is_available()), start_idx=user_ind)
    except Exception as e:
        msg = str(e)
        if 'CUSPARSE' in msg or 'DGL' in msg or 'cusparse' in msg:
            print('GPU prediction failed due to DGL/CUSPARSE; falling back to CPU...')
            # Force CPU in config and rebuild model/loader on CPU
            cfg_mod.config['cuda'] = -1
            model_cpu = create_model()
            model_cpu = load_model(model_cpu, model_path)
            data_loader = build_loader_for(model_cpu)
            user_ind = infer_loop(model_cpu, data_loader, use_cuda=False, start_idx=user_ind)
        else:
            raise

    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_path, 'w') as f:
        json.dump(truth_dict, f)
