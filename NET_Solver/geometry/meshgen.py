import numpy as np
from geometry import Annulus_Boundary
from utils import Plot_Grid
import torch
from geometry import *
from utils import *


class EllipticGrid(Annulus_Boundary):
    def __init__(self, r_outer, r_inner, eccentricity, cg, rg, h):
        super().__init__(r_outer, r_inner, eccentricity)
        self.nx = cg
        self.ny = rg
        self.h = h
        self.boundary = Annulus_Boundary(r_outer, r_inner, eccentricity).__call__(cg, rg)



    def __call__(self):
        # generate the mesh for physical space
        x,y = self.physical_grid(tol=1e-10)

        # generate the computational space
        xi, eta = self.computation_grid()

        # generate derivative of the physical plane
        dx_dxi, dy_dxi, dx_deta, dy_deta, jac, jac_inv = self.grid_derivative(x,y)

        return {'x': x,
                'y': y,
                'xi': xi,
                'eta': eta,
                'dx_dxi':dx_dxi,
                'dy_dxi':dy_dxi,
                'dx_deta': dx_deta,
                'dy_deta': dy_deta,
                'jac': jac,
                'jac_inv': jac_inv}

    def pre_processing(self):
        xl, yl = self.boundary['left'][0, :], self.boundary['left'][1, :]
        xr, yr = self.boundary['right'][0, :], self.boundary['right'][1, :]
        xlow, ylow = self.boundary['low'][0, :], self.boundary['low'][1, :]
        xtop, ytop = self.boundary['top'][0, :], self.boundary['top'][1, :]

        ######################################################################
        # combine the x and y into a 2d matrix
        x = np.zeros([self.ny, self.nx])
        y = np.zeros([self.ny, self.nx])
        x[:, 0] = xl
        y[:, 0] = yl
        x[:, -1] = xr
        y[:, -1] = yr
        x[0, :] = xlow
        y[0, :] = ylow
        x[-1, :] = xtop
        y[-1, :] = ytop

        return x, y

    def physical_grid(self, tol=1e-10):
        x, y = self.pre_processing()
        err = 2.2e-16
        assert x.shape == y.shape, f'Shape of X and Y does not match'
        count = 1
        A = np.ones([self.ny - 2, self.nx - 2])
        B = np.ones([self.ny - 2, self.nx - 2])
        C = np.ones([self.ny - 2, self.nx - 2])
        err_total = []
        while True:
            X = (A * (x[2:, 1:-1] + x[0:-2, 1:-1]) + C * (x[1:-1, 2:] + x[1:-1, 0:-2]) -
                 B / 2 * (x[2:, 2:] + x[0:-2, 0:-2] - x[2:, 0:-2] - x[0:-2, 2:])) / 2 / (A + C)
            Y = (A * (y[2:, 1:-1] + y[0:-2, 1:-1]) + C * (y[1:-1, 2:] + y[1:-1, 0:-2]) -
                 B / 2 * (y[2:, 2:] + y[0:-2, 0:-2] - y[2:, 0:-2] - y[0:-2, 2:])) / 2 / (A + C)

            error = np.max(np.max(np.abs(x[1:-1, 1:-1] - X)) + np.max(np.abs(y[1:-1, 1:-1] - Y)))
            err_total.append(error)
            x[1:-1, 1:-1] = X
            y[1:-1, 1:-1] = Y
            A = ((x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / self.h) ** 2 + (
                    (y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / self.h) ** 2 + err
            B = (x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / self.h * (x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / self.h + (
                    y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / self.h * (y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / self.h + err
            C = ((x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / self.h) ** 2 + (
                    (y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / self.h) ** 2 + err

            if error < tol:
                #print('Mesh Converged')
                break
                pass
            if count > 50000:
                print('Mesh did not reach convergence')
                break
                pass
            count += 1
        return x, y

    def computation_grid(self):
        xi_ = np.linspace(0, self.nx - 1, self.nx)
        eta_ = np.linspace(0, self.ny - 1, self.ny)
        xi, eta = np.meshgrid(xi_, eta_)
        xi = xi * self.h
        eta = eta * self.h
        return xi, eta

    def grid_derivative(self, x, y):
        # create array to store the derivatives
        dx_dxi = np.zeros(x.shape)
        dx_deta = np.zeros(x.shape)
        dy_dxi = np.zeros(y.shape)
        dy_deta = np.zeros(y.shape)

        # compute the derivatives
        dx_dxi_central = (-x[:, 4:] + 8 * x[:, 3:-1] - 8 * x[:, 1:-3] + x[:, 0:-4]) / 12 / self.h
        dx_dxi_left = (-11 * x[:, 0:-3] + 18 * x[:, 1:-2] - 9 * x[:, 2:-1] + 2 * x[:, 3:]) / 6 / self.h
        dx_dxi_right = (11 * x[:, 3:] - 18 * x[:, 2:-1] + 9 * x[:, 1:-2] - 2 * x[:, 0:-3]) / 6 / self.h

        dy_dxi_central = (-y[:, 4:] + 8 * y[:, 3:-1] - 8 * y[:, 1:-3] + y[:, 0:-4]) / 12 / self.h
        dy_dxi_left = (-11 * y[:, 0:-3] + 18 * y[:, 1:-2] - 9 * y[:, 2:-1] + 2 * y[:, 3:]) / 6 / self.h
        dy_dxi_right = (11 * y[:, 3:] - 18 * y[:, 2:-1] + 9 * y[:, 1:-2] - 2 * y[:, 0:-3]) / 6 / self.h

        dx_deta_central = (-x[4:, :] + 8 * x[3:-1, :] - 8 * x[1:-3, :] + x[0:-4, :]) / 12 / self.h
        dx_deta_low = (-11 * x[0:-3, :] + 18 * x[1:-2, :] - 9 * x[2:-1, :] + 2 * x[3:, :]) / 6 / self.h
        dx_deta_up = (11 * x[3:, :] - 18 * x[2:-1, :] + 9 * x[1:-2, :] - 2 * x[0:-3, :]) / 6 / self.h

        dy_deta_central = (-y[4:, :] + 8 * y[3:-1, :] - 8 * y[1:-3, :] + y[0:-4, :]) / 12 / self.h
        dy_deta_low = (-11 * y[0:-3, :] + 18 * y[1:-2, :] - 9 * y[2:-1, :] + 2 * y[3:, :]) / 6 / self.h
        dy_deta_up = (11 * y[3:, :] - 18 * y[2:-1, :] + 9 * y[1:-2, :] - 2 * y[0:-3, :]) / 6 / self.h

        # store the central, forward and backward derivatives in a single array
        dx_dxi[:, 2:-2] = dx_dxi_central
        dx_dxi[:, 0:2] = dx_dxi_left[:, 0:2]
        dx_dxi[:, -2:] = dx_dxi_right[:, -2:]

        dy_dxi[:, 2:-2] = dy_dxi_central
        dy_dxi[:, 0:2] = dy_dxi_left[:, 0:2]
        dy_dxi[:, -2:] = dy_dxi_right[:, -2:]

        dx_deta[2:-2, :] = dx_deta_central
        dx_deta[0:2, :] = dx_deta_low[0:2, :]
        dx_deta[-2:, :] = dx_deta_up[-2:, :]

        dy_deta[2:-2, :] = dy_deta_central
        dy_deta[0:2, :] = dy_deta_low[0:2, :]
        dy_deta[-2:, :] = dy_deta_up[-2:, :]

        # compute jacobian
        jac = dx_dxi * dy_deta - dx_deta * dy_dxi

        # inverse of jacobian
        jac_inv = 1 / jac

        return dx_dxi, dy_dxi, dx_deta, dy_deta, jac, jac_inv


if __name__ == '__main__':
    cg, rg = 70, 40
    h = 0.01
    anulus = EllipticGrid(1, 0.4, -0.99, cg, rg, h)()
    x, y = anulus['x'], anulus['y']
    print(x.shape, y.shape)

    Plot_Grid(x, y, cg, rg)

#
# class TFI:
#     def __init__(self, xi, eta, annulus, Boundary=False):
#         """
#         Transfinite Interpolation for generating grid using analytical function
#         This Function maps the complex physical space into cartesian rectangular domain
#
#         # NOTE the input Xi and Eta should be meshed before using in this function
#
#         xi_, eta_ = np.linspace(0,1,nx), np.linspace(0,1,ny)
#         xi, eta = np.meshgrid(xi_, eta_)
#         annulus = {some function with implemented boundary conditions}
#         x = TFI(xi, eta, annulus).X()
#         y = TFI(xi, eta, annulus).Y()
#
#         :param xi: Xi is the computational space in the x axis
#         :param eta: Eta is the computational space in the y axis
#         :param annulus: Annulus is a function which computes the grid in physical plane wrt boundary conditions
#         :param Boundary: Boundary a boolean parameter to specify if TFI is used for generating boundary or satisfying
#                             boundary conditions for computation
#         """
#         self.xi = xi
#         self.eta = eta
#         self.annulus = annulus
#         self.dx = (xi.max() - xi.min()) / len(xi)
#         self.dy = (eta.max() - eta.min()) / len(eta)
#         self.bound = Boundary
#
#     def __call__(self):
#         # get all the x related values
#         x = self.X()['x']
#         dxdxi = self.X()['dxdxi']
#         dxdeta = self.X()['dxdeta']
#
#         # get all the y related values
#         y = self.Y()['y']
#         dydxi = self.Y()['dydxi']
#         dydeta = self.Y()['dydeta']
#
#         # calculate the jacobian
#         jac = dxdxi*dydeta - dxdeta*dydxi
#
#
#
#         return {'x':x, 'dxdxi': dxdxi, 'dxdeta':dxdeta, 'y':y, 'dydxi':dydxi, 'dydeta':dydeta}
#
#     def X(self):
#         """
#         X returns the interpolated x-axis values from xi and eta
#         :return: x axis values for physical plane in a dict format
#         """
#         out = ((1 - self.eta) * self.annulus.Xr(self.xi) + self.eta * self.annulus.Xl(self.xi) + (1 - self.xi)
#                * self.annulus.Xt(self.eta) + self.xi * self.annulus.Xb(self.eta) -
#                (self.xi * self.eta * self.annulus.Xl(np.array([1])) + self.xi * (1 - self.eta) *
#                 self.annulus.Xr(np.array([1])) + self.eta * (1 - self.xi) * self.annulus.Xl(np.array([0])) +
#                 (1 - self.xi) * (1 - self.eta) * self.annulus.Xr(np.array([0]))))
#
#
#         dxdxi = np.gradient(out, self.dx)[0]
#         dxdeta = np.gradient(out, self.dx)[1]
#
#         # testing np.gradient
#         #test_dx_dxi = np.gradient(out, self.xi)
#         #print(dxdxi)
#
#         return {'x': out, 'dxdxi': dxdxi, 'dxdeta': dxdeta}
#         #return out
#
#     def Y(self):
#         """
#         Y returns the interpolated y-axis values from xi and eta
#         :return: Y axis values for physical plane in a dict format
#         """
#         out = ((1 - self.eta) * self.annulus.Yr(self.xi) + self.eta * self.annulus.Yl(self.xi) + (1 - self.xi)
#                * self.annulus.Yt(self.eta) + self.xi * self.annulus.Yb(self.eta) -
#                (self.xi * self.eta * self.annulus.Yl(np.array([1])) + self.xi * (1 - self.eta) *
#                 self.annulus.Yr(np.array([1])) + self.eta * (1 - self.xi) * self.annulus.Yl(np.array([0])) +
#                 (1 - self.xi) * (1 - self.eta) * self.annulus.Yr(np.array([0]))))
#
#         dydxi = np.gradient(out, self.dy)[0]
#         dydeta = np.gradient(out, self.dy)[1]
#
#         return {'y': out, 'dydxi': dydxi, 'dydeta': dydeta}
#         #return out
#
# if __name__ == '__main__':
#     xi_ = np.linspace(0,1,40)
#     eta_ = np.linspace(0,1,40)
#     xi, eta = np.meshgrid(xi_, eta_)
#     anulus = Analytical_Annulus(1., 0.6, 0.)
#     grid = TFI(xi, eta, anulus)
#     print(grid())