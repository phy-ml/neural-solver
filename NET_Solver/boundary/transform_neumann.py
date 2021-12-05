import numpy as np
import torch
from torch.utils.data import Dataset
from boundary import Boundary


class Transform_Neumann_Data(Dataset):
    def __init__(self, data, device='cpu'):
        mesh_data = np.stack(tuple(data[0]), -1).reshape(-1, len(data[0]))
        comput_data = np.stack(tuple(data[1]), -1).reshape(-1, len(data[1]))

        assert len(mesh_data) == len(comput_data), f'The mesh data has shape of {mesh_data.shape} and computation data ' \
                                                   f'has shape {comput_data.shape}'

        self.mesh_data = torch.from_numpy(mesh_data).float().to(device)
        self.comput_data = torch.from_numpy(comput_data).float().to(device)

    def __len__(self):
        return len(self.mesh_data)

    def __getitem__(self, i):
        return self.mesh_data[i], self.comput_data[i]


class Transform_Neumann(Boundary):
    def __init__(self, mesh_data, comput_data, device='cpu', name='Neumann'):
        super().__init__(name)

        assert isinstance(mesh_data, dict), 'Input data must be in dict format'
        assert isinstance(comput_data, dict), 'Computation Data must be in dict format'

        self.variable = tuple(mesh_data.keys())
        self.comp_var = tuple(comput_data.keys())
        data = [mesh_data.values(), comput_data.values()]
        self.data = Transform_Neumann_Data(data, device)
        self.device = device

    def neumann_loss(self):
        raise NotImplementedError('Define the Neumann boundary function.')

    def check_data(self, input, output):
        super().check_data()
        assert input == self.variable, f'Boundary {self.name} has different inputs !'
        assert self.neumann_loss, 'Define the neumann loss function'

    def boundary_loss(self, batch, model, criterion):
        input_data, compt_data = batch
        input_data.requires_grad = True
        output = model(input_data)
        loss = self.neumann_loss(input_data, output, compt_data)
        return {f'{self.name}_{name}': criterion(i, torch.zeros(i.shape).to(self.device)) for name, i in loss.items()}

    def compute_grad(self, outputs, inputs):
        gradient, = torch.autograd.grad(outputs, inputs, grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        return gradient
