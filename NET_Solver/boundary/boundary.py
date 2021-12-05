import torch
from torch.utils.data import DataLoader


class Boundary:
    def __init__(self, name):
        self.name = name
        self.data = None

    def load_data(self, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = len(self.data)

        dataset = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)
        return dataset

    def check_data(self):
        assert self.boundary_loss, 'Define the boundary loss function'
        # raise ValueError'Define the function for loss calculation'



