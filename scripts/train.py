import sys
sys.path.append('/workspace/modules')

import colored_glog as log
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from utils import read_config, wrap_path
from model import HierarchicalAttentionModel


def fetch_dataset():
    config = read_config()
    X = np.load(wrap_path(config['TrainDataPath']))
    y = np.load(wrap_path(config['TrainLabelPath']))

    return X, y


if __name__ == '__main__':
    config = read_config('train')
    history = []

    model = HierarchicalAttentionModel().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eval(config['LearningRate']))

    X, y = fetch_dataset()
    num_split = eval(config['KFoldSplit'])
    kfold = StratifiedKFold(n_splits=num_split)

    for epoch in range(eval(config['NumEpoch'])):
        log.info(f'Running Epoch #{epoch+1}')
        total_train_loss, total_valid_loss = 0, 0

        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        train_dataloader = DataLoader(train_dataset, batch_size=eval(config['BatchSize']))

        train_loss = 0
        num_train_iteration = len(train_dataset) // eval(config['BatchSize'])

        with tqdm(total=num_train_iteration) as pbar:
            for batch_id, (train_X, train_y) in enumerate(train_dataloader):
                pbar.update(1)
                optimizer.zero_grad()

                preds = model(train_X.type(torch.long).to('cuda'))
                loss = criterion(preds, train_y.type(torch.long).to('cuda'))
                train_loss += loss

                loss.backward()
                optimizer.step()

        total_train_loss += loss

        '''
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
            train_dataset = TensorDataset(torch.Tensor(X[train_idx]), torch.Tensor(y[train_idx]))
            valid_dataset = TensorDataset(torch.Tensor(X[valid_idx]), torch.Tensor(y[valid_idx]))

            train_dataloader = DataLoader(train_dataset, batch_size=eval(config['BatchSize']))
            valid_dataloader = DataLoader(valid_dataset, batch_size=eval(config['BatchSize']))

            train_loss, valid_loss = 0, 0
            num_train_iteration = len(train_dataset) // eval(config['BatchSize'])
            num_valid_iteration = len(valid_dataset) // eval(config['BatchSize'])

            # Train model
            with tqdm(total=num_train_iteration) as pbar:
                for batch_id, (train_X, train_y) in enumerate(train_dataloader):
                    pbar.update(1)
                    optimizer.zero_grad()

                    preds = model(train_X.type(torch.long).to('cuda'))
                    loss = criterion(preds, train_y.type(torch.long).to('cuda'))
                    train_loss += loss

                    loss.backward()
                    optimizer.step()

            total_train_loss += train_loss

            # Run validation
            with torch.no_grad():
                with tqdm(total=num_valid_iteration) as pbar:
                    for batch_id, (valid_X, valid_y) in enumerate(valid_dataloader):
                        pbar.update(1)

                        preds = model(valid_X.type(torch.long).to('cuda'))
                        loss = criterion(preds, valid_y.type(torch.long).to('cuda'))
                        valid_loss += loss

            total_valid_loss += valid_loss

        log_msg = 'Avg Loss ==> Train : {:.4f}\tValidation : {:.4f}'
        log.info(log_msg.format(total_train_loss/num_split, total_valid_loss/num_split))

        history.append((total_train_loss/num_split, total_valid_loss/num_split))
        torch.save(model.state_dict(), wrap_path(config['ModelSavePath'].format(epoch=epoch+1)))
        '''
        log.info('Train Loss : {:.4f}'.format(total_train_loss))
        torch.save(model.state_dict(), wrap_path(config['ModelSavePath'].format(epoch=epoch+1)))

    '''
    with open(wrap_path(config['HistorySavePath']), 'wb') as f:
        pickle.dump(history, f)
    '''
