import sys
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
    hit = 0

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
        for batch_id, (test_X, test_y) in enumerate(test_dataloader):
            preds = model(test_X.type(torch.long).to('cuda'))
            loss = criterion(preds, test_y.type(torch.long).to('cuda'))

            hit += (preds.argmax(dim=1) == test_y.type(torch.long).to('cuda')).sum().item()

    accuracy = 100 * (hit / len(test_dataset))
    log.info('Accuracy : {:.2f}'.format(accuracy))
