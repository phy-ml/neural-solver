import numpy as np
import torch
from torch.utils.data import Dataset
from boundary import Boundary


class Neumann_Data(Dataset):
    def __init__(self, data, device='cpu', compt='on'):
        if compt == 'on':
            x = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))

        elif compt == 'off':
            x = np.stack(tuple(data), -1).reshape(-1, len(data))

        self.data_x = torch.from_numpy(x).float().to(device)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, i):
        return self.data_x[i]


class Neumann(Boundary):
    def __init__(self, x, comput_data=None, device='cpu', mesh=True, name='Neumann'):
        super().__init__(name)
        assert isinstance(x, dict), 'Input data must be in dict format'

        if not (comput_data is None):
            assert isinstance(comput_data, dict), 'Computation Data must be in dict format'

        self.variable = tuple(x.keys())
        data = x.values()

        if mesh:
            self.data = Neumann_Data(data, device, compt='on')

        else:
            self.data = Neumann_Data(data, device, compt='off')

        self.device = device

    def neumann_loss(self):
        raise NotImplementedError('Define the Neumann boundary function.')

    def check_data(self, input, output):
        super().check_data()
        assert input == self.variable, f'Boundary {self.name} has different inputs !'
        assert self.neumann_loss, 'Define the neumann loss function'

    def boundary_loss(self, batch, model, criterion):
        input = batch
        input.requires_grad = True
        output = model(input)
        loss = self.neumann_loss(input, output)
        return {f'{self.name}_{name}': criterion(i, torch.zeros(i.shape).to(self.device)) for name, i in loss.items()}

    def compute_grad(self, outputs, inputs):
        gradient, = torch.autograd.grad(outputs, inputs, grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        return gradient
