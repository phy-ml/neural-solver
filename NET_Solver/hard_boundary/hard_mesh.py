import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models import *
from torch.autograd import Variable


# define the boundary
class Hard_Mesh:
    def __init__(self, mesh_data, device='cpu'):
        assert isinstance(mesh_data, dict), f'The mesh data boundary should be in dict format'
        self.variables = tuple(mesh_data.keys())
        data = mesh_data.values()
        data = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))
        self.gen_data = torch.from_numpy(data).float().to(device)
        #self.gen_data.requires_grad = True

    def data(self):
        data = self.gen_data
        # data_1 = data[:,0].reshape(-1,1)
        # data_2 = data[:,1].reshape(-1,1)
        # data = torch.cat((data_2, data_1), dim=1)
        return data



class Trial_Solution:
    def __init__(self, model, nx, ny):
        self.model = model
        self.nx, self.ny = nx, ny

    def Dirichlet(self, model_data):
        # impose the dirichlet conditions
        # u = 0 @ eta = 0 and 1
        model_data[:, 0] = 0.
        model_data[:, -1] = 0.
#Variable(torch.Tensor([0.0]).float(), requires_grad=True)
        return model_data

    def Neumann(self, model_data):
        # impose neumann boundary
        # dudy =0 @ xi = 0 and 1
        model_data[0, :] = model_data[1, :]
        model_data[-1, :] = model_data[-2, :]

        return model_data

    def __call__(self, inputs):
        # predict the general output
        pred = self.model(inputs)

        # reshape into a cartesian format
        pred = pred.reshape(self.ny, self.nx)

        # apply the boundary conditions
        # dirichlet
        out = self.Dirichlet(pred)

        # neumann
        out = self.Neumann(out)

        # reshape back into original format
        out = out.reshape(-1, 1)

        return out


# class Load_Data(Dataset):
#     def __init__(self, inputs, prediction):
#         self.inputs = inputs
#         self.pred = prediction
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, item):
#         return self.inputs[item], self.pred[item]


if __name__ == '__main__':
    nx = 70
    ny = 70
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    mesh = Hard_Mesh({'x': x, 'y': y},device='cpu')
    print(mesh.device)
    # net = MLP(2, 1, 1, 10, act=torch.nn.Tanh())
    # trial = Trial_Solution(net, nx, ny)

    # test = mesh.load_data(trial)
    # for i, j in test:
    #     print(i.shape, j.shape)
