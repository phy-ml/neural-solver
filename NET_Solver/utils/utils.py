import numpy as np
import matplotlib.pyplot as plt


class check_var:
    def __init__(self):
        print('pass')

    def __call__(self, input_data, output_data):
        self.is_list(input_data)
        self.is_list(output_data)
        self.is_unique(input_data)
        self.is_unique(output_data)
        self.repeat(input_data, output_data)

    def is_list(self, data):
        if isinstance(data, tuple):
            for i in data:
                if not isinstance(i, str):
                    raise ValueError(f'{str(i)} should must b string')

    def is_unique(self, data):
        for i, iter_1 in enumerate(data):
            for j, iter_2 in enumerate(data):
                if i != j and iter_1 == iter_2:
                    raise ValueError(f'{str(i)} : Repeated Value !!')

    def repeat(self, data_1, data_2):
        for i in data_1:
            if i in data_2:
                raise ValueError(f'{str(i)} is repeated in both input and output')


class Record:
    def __init__(self):
        self.record = {}
        self.current = {}
        self.round_off = 5

    def _add(self, param):
        for key, value in param.items():
            if key not in self.record:
                self.record[key] = []
            self.record[key].append(value)

    def update(self, param):
        for key, value in param.items():
            if key not in self.current:
                self.current[key] = []
            self.current[key].append(value)

    def average(self):
        return {key: round(np.mean(self.current[key]), self.round_off) for key in self.current}

    def _step(self):
        for key in self.current:
            self._add({key: np.mean(self.current[key])})
        self.current = {}

    def __str__(self):
        s = ''
        for key, value in self.record.items():
            s += f'| {key} {round(value[-1], self.round_off)}'

        return s


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# function to plot the grid data
def Plot_Grid(x,y, nx, ny):
    plt.figure(figsize=(10,8))
    for i in range(nx):
        plt.plot(x[:,i],y[:,i],'black')

    for j in range(ny):
        plt.plot(x[j,:], y[j,:],'black')

    plt.axis('equal')
    plt.show()

# if __name__ == '__main__':
#     input_ = ('X','Y', 'Z')
#     output = ('X','V')
#     check_var()(input_, output)
