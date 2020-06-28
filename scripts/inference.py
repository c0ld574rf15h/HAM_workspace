import sys, pickle
sys.path.append('/workspace/modules')

from tqdm import tqdm
import colored_glog as log

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils import read_config, wrap_path
from model import HierarchicalAttentionModel


def fetch_dataset():
    config = read_config()
    X = np.load(wrap_path(config['TestDataPath']))
    y = np.load(wrap_path(config['TestLabelPath']))

    return X, y


if __name__ == '__main__':
    config = read_config('test')
    hit, all_preds = 0, []
    #flow_embeddings = np.zeros((10000, 256))

    model = HierarchicalAttentionModel().to('cuda')
    model.load_state_dict(
        torch.load(
            wrap_path(config['ModelSavePath']).format(epoch=config['Checkpoint'])
        )
    )

    criterion = nn.CrossEntropyLoss()

    X, y = fetch_dataset()
    test_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    log.info('Running inference...')
    with torch.no_grad():
        num_iteration = len(test_dataloader) // 128
        with tqdm(total=num_iteration) as pbar:
            for batch_id, (test_X, test_y) in enumerate(test_dataloader):
                preds, flow_embedding = model(test_X.type(torch.long).to('cuda'))
                loss = criterion(preds, test_y.type(torch.long).to('cuda'))

                hit += (preds.argmax(dim=1) == test_y.type(torch.long).to('cuda')).sum().item()
                all_preds += [i.item() for i in preds.argmax(dim=1)]
                #flow_embeddings[batch_id*128:batch_id*128+flow_embedding.shape[0],] = flow_embedding.data.cpu().numpy()

                pbar.update(1)

    accuracy = 100 * (hit / len(test_dataset))
    log.info('Accuracy : {:.2f}'.format(accuracy))

    #np.save(wrap_path('results/flow_embedding'), flow_embeddings)

    with open(
        wrap_path(
            config['PredictionSavePath'].format(
                epoch=config['Checkpoint']
            )
        ), 'wb') as f:
        pickle.dump(all_preds, f)
