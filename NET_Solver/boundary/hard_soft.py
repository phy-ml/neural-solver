import numpy as np
import torch

class Hard_Soft:
    def __init__(self, Dirichlet, Neumann):
        assert isinstance(Dirichlet, dict), f'The Dirichlet data boundary should be in dict format'
        assert isinstance(Neumann, dict), f'The Neumann data boundary should be in dict format'

        self.diri = Dirichlet
        self.neumann = Neumann

    def Dirichlet(self):
        var = tuple(self.diri.keys())
        data = self.diri.values()
        in_data = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))
        in_data = torch.from_numpy(in_data).float()
        return in_data

    def Neumann(self):
        var = tuple(self.neumann.keys())
        data = self.neumann.values()
        in_data = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))
        in_data = torch.from_numpy(in_data).float()
        return in_data

    def slice_data(self, model_data, tol=0.000007):
        # combine the data from neumann and dirichlet
        combo_data = torch.cat((self.Dirichlet(), self.Neumann()), dim=0)

        # return the elements of model data without the boundary data for hard boundary
        # implementation
        # https://discuss.pytorch.org/t/any-way-of-filtering-given-rows-from-a-tensor-a/83828/2
        cdist = torch.cdist(model_data, combo_data, p=1)
        min_dist = torch.min(cdist, dim=1).values

        # return the tensor
        return model_data[min_dist > tol]