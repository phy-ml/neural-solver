import numpy as np
import torch
from torch.utils.data import Dataset
from boundary import Boundary


class Dirichlet_Data(Dataset):
    def __init__(self, data, device='cpu', compt = 'on'):
        if compt == 'on':
            x = np.stack(np.meshgrid(*data[0]), -1).reshape(-1, len(data[0]))
            y = np.stack(data[1], axis=1)

        elif compt == 'off':
            x = np.stack(tuple(data[0]),-1).reshape(-1, len(data[0]))
            y = np.stack(data[1], axis=1)

        assert len(x) == len(y), f'The input has shape of {x.shape} and output has {y.shape}'

        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class Dirichlet(Boundary):
    def __init__(self, x, y, device='cpu', mesh=True, name='Dirichlet'):
        super(Dirichlet, self).__init__(name)

        assert isinstance(x, dict), 'Input data must be in dict format'
        assert isinstance(y, dict), 'Output data must be in dict format'

        self.variable = [tuple(x.keys()), tuple(y.keys())]
        data = [x.values(), list(y.values())]

        if mesh:
            self.data = Dirichlet_Data(data, device, compt='on')

        else:
            self.data = Dirichlet_Data(data, device, compt='off')

    def check_data(self, input, output):
        super().check_data()
        _input, _output = self.variable
        assert input == _input, f'Boundary {self.name} with different input!'
        if output != _output:
            print(f'Boundary {self.name} with different outputs ! {output} Vs {_output}')
        # filter the boundary data
        self.output_id = [output.index(i) for i in self.variable[1] if i in output]

    def boundary_loss(self, batch, model, criterion):
        x, y = batch
        pred = model(x)
        # temp code
        # print(pred[:,self.output_id].shape)
        # print(y.shape)
        # print(torch.nn.MSELoss()(pred[:, self.output_id], y).item())
        # #print(criterion(pred[:,self.output_id], (y.reshape(len(y),-1)))
        return {self.name: criterion(pred[:, self.output_id], y)}
