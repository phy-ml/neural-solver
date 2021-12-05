import numpy as np
import torch
from torch.utils.data import Dataset
from boundary import Boundary

class Periodic_Data(Dataset):
    def __init__(self, data, device='cpu'):
        x_1 = np.stack(np.meshgrid(*data[0]), -1).reshape(-1,len(data[0]))
        x_2 = np.stack(np.meshgrid(*data[1]), -1).reshape(-1, len(data[1]))
        assert len(x_1) == len(x_2),f'The length of both input does not match.'
        self.x1 = torch.from_numpy(x_1).float().to(device)
        self.x2 = torch.from_numpy(x_2).float().to(device)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, i):
        return self.x1[i], self.x2[i]

class Periodic(Boundary):
    def __init__(self, x1, x2, device='cpu', name='Periodic'):
        super().__init__(name)
        assert isinstance(x1, dict),'Input must be in a dict format.'
        assert isinstance(x2, dict),'Input must be in a dict format.'
        assert x1.keys() == x2.keys(),'Both input must be same variable.'
        self.variable = [tuple(x1.keys()), tuple(x2.keys())]
        data = [x1.values(), x2.values()]
        self.data = Periodic_Data(data, device)

    def check_data(self,inputs, outputs):
        super().check_data()
        x1_, x2_ = self.variable
        assert x1_ == inputs,f'Boundary {self.name} has different inputs'
        # # temp code
        # print(f'x2: {x2}')
        # print(f'x2_: {x2_}')
        assert x2_ == inputs,f'Boundary {self.name} has different inputs'

    def boundary_loss(self, batch, model, criterion):
        x1, x2 = batch
        pred_1 = model(x1)
        pred_2 = model(x2)
        return {self.name: criterion(pred_1, pred_2)}
