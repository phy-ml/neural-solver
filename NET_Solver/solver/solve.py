from fastprogress import master_bar, progress_bar
# from IPython.display import display
from utils.utils import *
import torch
import matplotlib.pyplot as plt
from models.neural_net import _init_weights


class PDE():

    def __init__(self, inputs, outputs):
        # check if the input and output is in string format
        if isinstance(inputs, str):
            inputs = tuple(inputs)
        if isinstance(outputs, str):
            outputs = tuple(outputs)

        # check for any discrepancy in the input file
        check_var()(inputs, outputs)

        self.output = outputs
        self.input = inputs
        self.mesh = None
        self.boundary = []

    def set_mesh(self, mesh):
        assert mesh.variable == self.input, f'Mesh and Input variables does not match !!'
        self.mesh = mesh

    def add_boundary(self, boundary):
        assert boundary.name not in [boundary.name for boundary in
                                     self.boundary], f'Boundary {boundary.name} already exist, choose another name'
        boundary.check_data(self.input, self.output)
        self.boundary.append(boundary)

    def hard_soft(self, bound):
        self.trim_data = bound

    def compile(self, model, optimizer, scheduler=None, criterion=None):
        self.model = model
        # self.model.apply(_init_weights)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion else torch.nn.MSELoss()

    def PDELoss(self, pred, input):
        raise ValueError('Implement PDE loss function')

    def set_dataloader(self, batch_size, shuffle):
        data_loader = {'overall': self.mesh.load_data(batch_size, shuffle),
                       'boundary': {}}
        for boundary in self.boundary:
            data_loader['boundary'][boundary.name] = boundary.load_data(batch_size, shuffle)

        return data_loader

    def compute_grad(self, outputs, inputs):
        gradient, = torch.autograd.grad(outputs, inputs, grad_outputs=outputs.data.new(outputs.shape).fill_(1),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        return gradient

    def solve(self, epochs=50, batch_size=None, shuffle=True, graph=True):
        data_loader = self.set_dataloader(batch_size, shuffle)
        if graph:
            self.graph_fig, (self.graph_ax1, self.graph_ax2) = plt.subplots(1, 2, figsize=(15, 5))
            self.graph_out = display(self.graph_fig, display_id=True)

        # solve the PDE equation
        history = Record()
        mb = master_bar(range(1, epochs + 1))
        for epoch in mb:
            # self.model.apply(add_noise_to_weights)
            history._add({'lr': get_lr(self.optimizer)})
            # iterate over the entire dataset
            for batch in progress_bar(data_loader['overall'], parent=mb):
                x = batch
                self.optimizer.zero_grad()
                # optimize for boundary data points
                for boundary in self.boundary:
                    for batch in data_loader['boundary'][boundary.name]:
                        loss = boundary.boundary_loss(batch, self.model, self.criterion)
                        for name, b_loss in loss.items():
                            b_loss.backward()
                            history.update({name: b_loss.item()})

                # implement the hard boundary
                # x = self.trim_data.slice_data(x_)
                # optimize for the entire domain
                x.requires_grad = True
                # print(x.shape)
                pred = self.model(x)
                pde_loss = self.PDELoss(x, pred)
                assert isinstance(pde_loss, dict), 'Loss function should return output in dict format'
                for name, loss in pde_loss.items():
                    domain_loss = self.criterion(loss, torch.zeros(loss.shape).to(self.mesh.device))
                    #domain_loss = self.pde_criterion(loss)
                    domain_loss.backward(retain_graph=True)
                    history.update({name: domain_loss.item()})

                self.optimizer.step()
                mb.child.comment = str(history.average())
            history._step()
            mb.main_bar.comment = str(history)
            if graph:
                self.plot_history(history)

            if self.scheduler:
                self.scheduler.step()

        if graph:
            plt.close()
        return history.record

    def plot_history(self, history):
        self.graph_ax1.clear()
        self.graph_ax2.clear()
        for key, value in history.record.items():
            if key != 'lr':
                self.graph_ax1.plot(value, label=key)
            else:
                self.graph_ax2.plot(value, label=key)
        self.graph_ax1.legend(loc='upper right')
        self.graph_ax2.legend(loc='upper right')
        self.graph_ax1.grid(True)
        self.graph_ax2.grid(True)
        self.graph_ax1.axis('equal')
        self.graph_ax1.axis('equal')
        self.graph_ax1.set_yscale('log')
        self.graph_out.update(self.graph_fig)

    def eval(self, mesh_data, batch_size=None):
        load_data = mesh_data.load_data(batch_size, shuffle=False)
        output = torch.tensor([]).to(mesh_data.device)
        self.model.eval()
        with torch.no_grad():
            for batch in load_data:
                output = torch.cat([output, self.model(batch)])
        return output

    def axial_vel(self, mesh_data, trial_sol, batch_size=None):
        data = mesh_data.gen_data[:]
        # output = torch.tensor([]).to(mesh_data.device)
        output = trial_sol(data)
        output = self.compute_grad(output, data)
        vx, vy = output[:, 0], output[:, 1]
        return vx, vy
