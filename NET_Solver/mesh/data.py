import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, data, device='cpu', compt='on'):
        if compt == 'on':
            mesh_data = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))

        elif compt == 'off':
            mesh_data = np.stack(tuple(data), -1).reshape(-1, len(data))

        self.data = torch.from_numpy(mesh_data).float().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Mesh:
    def __init__(self, input_data, device='cpu', mesh=True):
        assert isinstance(input_data, dict), 'The input data should be in Dict format'
        self.variable, self.data = tuple(input_data.keys()), input_data.values()
        if mesh:
            self.gen_data = Data(self.data, device, compt='on')
        else:
            self.gen_data = Data(self.data, device, compt='off')

        self.device = device

    def load_data(self, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = len(self.gen_data)

        dataset = DataLoader(self.gen_data, batch_size=batch_size, shuffle=shuffle)

        return dataset
