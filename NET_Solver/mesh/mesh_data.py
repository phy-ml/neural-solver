import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Mesh_Dataset(Dataset):
    def __init__(self, data, device='cpu'):
        mesh_data = np.stack(tuple(data[0]), -1).reshape(-1, len(data[0]))
        comput_data = np.stack(tuple(data[1]), -1).reshape(-1, len(data[1]))

        assert len(mesh_data) == len(comput_data), f'The mesh has shape of {mesh_data.shape} and computation data has ' \
                                                   f'shape of {comput_data.shape}'
        self.mesh_data = torch.from_numpy(mesh_data).float().to(device)
        self.comput_data = torch.from_numpy(comput_data).float().to(device)

    def __len__(self):
        return len(self.mesh_data)

    def __getitem__(self, i):
        return self.mesh_data[i], self.comput_data[i]


class Mesh_Data:
    def __init__(self, mesh_data, comput_data, device='cpu'):
        assert isinstance(mesh_data, dict), f'Input mesh data should be in dict format'
        assert isinstance(comput_data, dict), f'Computation data should be in dict format'

        self.variable = tuple(mesh_data.keys())
        self.comp_var = tuple(comput_data.keys())
        self.data = [mesh_data.values(), comput_data.values()]
        self.gen_data = Mesh_Dataset(self.data, device)
        self.device = device

    def load_data(self, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = len(self.gen_data)

        dataset = DataLoader(self.gen_data, batch_size=batch_size, shuffle=shuffle)

        return dataset


if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xi = np.linspace(0, 1, 10)
    eta = np.linspace(0, 1, 10)
    mesh = Mesh_Data({'x': x, 'y': y}, {'xi': xi, 'eta': eta})
    x, y = mesh.gen_data[:5]
    print(x)
    print(y)
    print(mesh.variable, mesh.comp_var)
