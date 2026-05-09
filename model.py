import re
import os
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import timm
import numpy as np
from dataloader_test import GetData
from torch import nn, optim
from numpy import random
from os import listdir
from os import path as op
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import json as js
import scipy.interpolate as itp
import tqdm
import warnings


class Model_workflow:
    def __init__(self, data_path):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.data_path = data_path
        self.model = None
        self.save_path = None
        self.sample_set = None
        self.validating = []
        self.train_params = {'start': None, 'window': None, 'N_day': None, 'tech': None}
        self.model_name = None
        self.earlyStop = None

    def register_save_path(self, path: str):
        self.save_path = path

    def register_earlyStop(self, func):
        assert callable(func)
        self.earlyStop = func

    def check_point(self):
        pt_list = listdir(self.save_path)
        if not self.model_name:
            return 0
        if len(pt_list) > 0:
            compare_list = []
            pt_list_copy = copy.copy(pt_list)
            for i in pt_list_copy:
                if i[-11:-3] == 'complete':
                    pt_list.remove(i)
                    continue
                name = i[:-3].split('_')[0]
                suffix = i[-2:]
                if (name != self.model_name) or (suffix != 'pt'):
                    pt_list.remove(i)
                    continue
        if len(pt_list) > 0:
            for i in pt_list:
                epoch = int(i[:-3].split('_')[1])
                counter = int(i[:-3].split('_')[2])
                compare_list.append((epoch, counter))
            latest_file_idx = compare_list.index(max(compare_list))
            check_point = torch.load(op.join(self.save_path, pt_list[latest_file_idx]))
            return check_point
        else:
            return 0

    def clear_space(self):
        pt_list = listdir(self.save_path)
        if not self.model_name:
            return
        if len(pt_list) > 0:
            compare_list = []
            pt_list_copy = copy.copy(pt_list)
            for i in pt_list_copy:
                if i[-11:-3] == 'complete':
                    pt_list.remove(i)
                    continue
                name = i[:-3].split('_')[0]
                suffix = i[-2:]
                if (name != self.model_name) or (suffix != 'pt'):
                    pt_list.remove(i)
                    continue
        if len(pt_list) > 0:
            for i in pt_list:
                epoch = int(i[:-3].split('_')[1])
                counter = int(i[:-3].split('_')[2])
                compare_list.append((epoch, counter))
            oldest_file_idx = compare_list.index(min(compare_list))
        if len(pt_list) > 5:
            os.remove(op.join(self.save_path, pt_list[oldest_file_idx]))
        else:
            return

    def train(self, model, model_name: str, epoch: int, lr: float, batch_size: int, iteration: int, criteria, Optimizer,
              initializer, start, window, N_day_after, tech, minimum_epoch: int, seed: int):
        self.model = model
        self.model.to(self.device)
        self.model_name = model_name
        self.train_params['start'] = start
        self.train_params['window'] = window
        self.train_params['window'] = N_day_after
        self.train_params['tech'] = tech
        self.sample_set = GetData(self.data_path, start, window, N_day_after, flag='sample', market='us')
        self.sample_set.register_tech(tech)
        criteria.to(self.device)
        optimizer = Optimizer(self.model.parameters(), lr=lr, weight_decay=1e-8)
        cp = self.check_point()
        if cp == 0:
            for name, param in self.model.named_parameters():
                if name.endswith('weight'):
                    initializer(param.unsqueeze(0))
                else:
                    nn.init.constant_(param, 0)
        epoch_loss = []
        # writer = SummaryWriter('./log/')
        for e in tqdm.tqdm(range(epoch), desc='epoch'):
            torch.manual_seed(seed)
            train_loader = DataLoader(dataset=self.sample_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                      persistent_workers=True, pin_memory=True)
            cp = self.check_point()
            self.model.train()
            index = 1
            if cp != 0:
                if e == cp['epoch']:
                    self.model.load_state_dict(cp['model_state_dict'])
                    optimizer.load_state_dict(cp['optimizer_state_dict'])
                    train_loss = cp['train_loss']
                    epoch_loss = cp['epoch_loss']
                    index = cp['index']
                elif e < cp['epoch']:
                    epoch_loss = cp['epoch_loss']
                    continue
                else:
                    optimizer.load_state_dict(cp['optimizer_state_dict'])
                    train_loss = []
            else:
                train_loss = []
            train_iter_bar = tqdm.tqdm(total=iteration, desc='train_iteration', initial=index, leave=True)
            for item in train_loader:
                if index >= iteration:
                    break
                bad_batch = False
                for status in item[2]:
                    if not status:
                        bad_batch = True
                        print('bad batch detected!')
                        break
                if bad_batch:
                    print('bad batch jumped!')
                    continue
                img_tensor = item[0].to(self.device)
                label_tensor = item[1].to(self.device)
                # _ret = .01 * item[3].to(self.device)
                pred_label = self.model(img_tensor)
                optimizer.zero_grad()
                loss = criteria(input=pred_label, target=label_tensor)
                loss.backward()
                optimizer.step()
                # for b in range(pred_label.shape[0]):
                #     writer.add_scalar(f'epoch_{e}/pred', pred_label[b].detach().cpu().numpy().argmax(),
                #                       (index - 1) * pred_label.shape[0] + b)
                # writer.add_scalar(f'epoch_{e}/loss', loss, index)
                mean_loss = float(loss.detach().cpu().numpy().mean())

                train_loss.append(mean_loss)
                index += 1
                if index % 1000 == 0:
                    torch.save({'epoch': e,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch_loss': epoch_loss,
                                'train_loss': train_loss,
                                'index': index
                                },
                               op.join(self.save_path, model_name + f'_{str(e)}_{str(index)}.pt'))
                self.clear_space()
                train_iter_bar.update(1)
            epoch_loss.append(float(np.array(train_loss).mean()))
            torch.save({'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'epoch_loss': epoch_loss,
                        'index': 1,
                        },
                       op.join(self.save_path, model_name + f'_{str(e + 1)}_{str(0)}.pt'))
            self.sample_set.set_validating_flag(True)
            val_loader = DataLoader(dataset=self.sample_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                    persistent_workers=True, pin_memory=True)
            val_iteration = 20 * iteration
            val_iter_bar = tqdm.tqdm(total=val_iteration, desc='validating_iteration', initial=0, leave=True)
            val_accuracy = {'hit': 0, 'miss': 0}
            self.model.eval()
            v_index = 0
            for item in val_loader:
                if v_index >= val_iteration:
                    break
                bad_batch = False
                for status in item[2]:
                    if not status:
                        bad_batch = True
                        print('bad batch detected!')
                        break
                if bad_batch:
                    print('bad batch jumped!')
                    continue
                img_tensor = item[0].to(self.device)
                label_tensor = item[1].detach().numpy()
                pred_label = self.model(img_tensor)
                pred_label = pred_label.detach().cpu().numpy()
                for batch in range(label_tensor.shape[0]):
                    true = label_tensor[batch]
                    pred = pred_label[batch].argmax()
                    # print(f"true: {true}")
                    # print(f"prediction: {pred}")
                    if true == pred:
                        val_accuracy['hit'] += 1
                    else:
                        val_accuracy['miss'] += 1
                v_index += 1
                val_iter_bar.update(1)

            acu = val_accuracy['hit'] / (val_accuracy['miss'] + val_accuracy['hit'])
            with open('./log/validating accuracy.txt', 'a') as file:
                print(f'model name {self.model_name} epoch {e} validating accuracy: {acu}', file=file)
            self.sample_set.set_validating_flag(False)
            smooth_loss_curve = itp.make_smoothing_spline(np.array([i for i in range(1, len(train_loss) + 1)]),
                                                          train_loss)
            x = np.linspace(1, len(train_loss) + 1, 1000)
            fig, axs = plt.subplots(ncols=1, nrows=1, layout="constrained")
            axs.plot(x, smooth_loss_curve(x))
            axs.set_title('train loss curve')
            axs.set_xlabel('iteration')
            axs.set_ylabel('loss')
            axs.grid(True)
            plt.savefig(f'./fig/epoch{e}_loss_curve.svg')
            plt.close()
            seed += 1
            if self.earlyStop(epoch_loss, minimum_epoch):
                break
        if len(epoch_loss) > 5:
            smooth_loss_curve = itp.make_smoothing_spline(np.array([i for i in range(1, len(epoch_loss) + 1)]),
                                                          epoch_loss)
            x = np.linspace(1, len(epoch_loss) + 1, 1000)
            fig, axs = plt.subplots(ncols=1, nrows=1, layout="constrained")
            axs.plot(x, smooth_loss_curve(x))
            axs.set_title('total loss curve')
            axs.set_xlabel('epoch')
            axs.set_ylabel('loss')
            axs.grid(True)
            plt.savefig(f'./fig/total_loss_curve.svg')
            plt.close()
        torch.save(model.state_dict(), op.join(self.save_path, model_name + '_params_complete.pt'))


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, (12, 3)),
            nn.Conv2d(4, 8, (3, 3)),
            nn.Conv2d(8, 8, (3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(8, 8, (3, 3)),
            nn.Conv2d(8, 4, (3, 3)),
            nn.Conv2d(4, 2, (3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Flatten(),
            nn.Linear(20776, 2)
        )

    def forward(self, x):
        y = self.network(x)
        return y


def EarlyStop_func(loss_list: list, minimum):
    if len(loss_list) < minimum:
        return False
    else:
        if loss_list[-1] - loss_list[-2] <= 0:
            print('just in time to stop!!')
            return True


if __name__ == '__main__':
    from torchviz import make_dot
    cnn = CNN_model().to('cuda:0')
    x = torch.tensor(np.random.randn(1,3,224,224)).to('cuda:0').float()
    y = cnn(x)
    dot = make_dot(y)
    dot.view()
    # test_my_model = Train_model('./data/tech')
