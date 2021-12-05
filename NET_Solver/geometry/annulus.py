import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt


class Annulus_Boundary:
    def __init__(self, r_outer, r_inner, eccentricity):
        """
        Analytical Annulus maps the boundary values of a eccentric annulus from a computational domain into a physical
         domain for performing computational analysis using analytical scheme.
        ==>> The outer circle is positioned at the center of origin, while the inner circle position is calculated
        from the eccentricity
        :param r_outer: The radius of the outer circle
        :param r_inner: The radius of the inner circle
        :param eccentricity: The eccentricity describing the position of inner circle within outer circle
        """
        self.r_outer, self.r_inner = r_outer, r_inner
        # condition to check if radius of outer circle is larger than inner
        if r_outer <= r_inner:
            raise ValueError('Outer circle should be greater than inner circle')

        # condition to check if eccentricity is not greater than 1from
        # the eccentricity should range between -1 to 1.
        if eccentricity > 1.0:
            raise ValueError('Eccentricity should not be greater than 1')
        else:
            self.ecc = eccentricity

        # calculate the position of the inner circle wrt the eccentricity
        self.x_inner = (self.ecc * (2 * self.r_outer - 2 * self.r_inner)) / 2
        # The y axis for center of inner circle is assumed to be
        self.y_inner = 0

    def Xt(self, x):
        """
        Xt calculates the x-axis values in physical domain to implement top-boundary conditions
        :param x: xi -- (X axis values in computational domain)
        :return: x -- (X- axis values in physical domain)
        """
        out = (1 - x) * np.array([np.pi])
        return np.cos(out)* self.r_outer

    def Yt(self, x):
        """
        Yt calculates the y-axis values in physical domain to implement top-boundary conditions
        :param x: eta -- (Y-axis values in computational domain)
        :return: y -- (Y-axis values in physical domain)
        """
        out = (1 - x) * np.array([np.pi])
        return np.sin(out)* self.r_outer

    def Xb(self, x):
        """
        Xb calculates the x-axis values in physical domain to implement bottom-boundary conditions
        :param x: xi -- (X axis values in computational domain)
        :return: x -- (X- axis values in physical domain)
        """
        out = (1 - x) * np.array([np.pi])
        return np.cos(out) * self.r_inner + self.x_inner

    def Yb(self, x):
        """
        Yb calculates the y-axis values in physical domain to implement bottom-boundary conditions
        :param x: eta -- (Y-axis values in computational domain)
        :return: y -- (Y-axis values in physical domain)
        """
        out = (1 - x) * np.array([np.pi])
        return np.sin(out) * self.r_inner

    def Xl(self, x):
        """
        Xl calculates the x-axis values in physical domain to implement left-boundary conditions
        :param x: xi -- (X axis values in computational domain)
        :return: x -- (X- axis values in physical domain)
        """
        # interpolation ==>> (0,1)* diff + min
        diff = -self.r_outer - (self.x_inner - self.r_inner)
        out = diff * x + (self.x_inner - self.r_inner)
        return out

    def Yl(self, x):
        """
        Yl calculates the y-axis values in physical domain to implement left-boundary conditions
        :param x: eta -- (Y-axis values in computational domain)
        :return: y -- (Y-axis values in physical domain)
        """
        out = x * 0
        return out

    def Xr(self, x):
        """
        Xr calculates the x-axis values in physical domain to implement right-boundary conditions
        :param x: xi -- (X axis values in computational domain)
        :return: x -- (X- axis values in physical domain)
        """
        # below trick is implemented to cast any range of values
        # diff = (max - min) ==>> (0,1)* diff + min
        diff = self.r_outer - (self.x_inner + self.r_inner)
        out = diff * x + (self.x_inner + self.r_inner)
        return out

    def Yr(self, x):
        """
        Yr calculates the y-axis values in physical domain to implement right-boundary conditions
        :param x: eta -- (Y-axis values in computational domain)
        :return: y -- (Y-axis values in physical domain)
        """
        out = x * 0
        return out

    def check(self):
        """
        Test function to check if all tye boundary conditions are met.
        """
        # the following four conditions should be satisfied for a logical boundary data implementation
        assert self.Xb(0) == self.Xl(0), f'The bottom and left boundary should be in order'
        assert self.Xb(1) == self.Xr(0), f'The bottom and right boundary should be in order'
        assert self.Xr(1) == self.Xt(1), f'The right and top boundary should be in order'
        assert self.Xl(1) == self.Xt(0), f'The left and top boundary should be in order'
        print('PASS : Boundary Conditions in Order !!')

    def __call__(self, cg, rg):
        # circumferential grid (cg)
        # radial grid (rg)
        cg_x, cg_y = np.linspace(0,1,cg), np.linspace(0,1,cg)
        rg_x, rg_y = np.linspace(0,1,rg), np.linspace(0,1,rg)
        xl = self.Xl(rg_x)
        yl = self.Yl(rg_y)
        xr = self.Xr(rg_x)
        yr = self.Yr(rg_y)
        xlow = self.Xb(cg_x)
        ylow = self.Yb(cg_y)
        xtop = self.Xt(cg_x)
        ytop = self.Yt(cg_y)
        return {'left':np.stack((xl,yl)),
                'right':np.stack((xr,yr)),
                'top':np.stack((xtop, ytop)),
                'low':np.stack((xlow,ylow))}

    def Plot_Boundary(self, cg, rg):
        boundary = self.__call__(cg,rg)
        xl, yl = boundary['left'][0,:], boundary['left'][1,:]
        xr, yr = boundary['right'][0,:], boundary['right'][1,:]
        xlow, ylow = boundary['low'][0,:], boundary['low'][1,:]
        xtop, ytop = boundary['top'][0,:], boundary['top'][1,:]


        plt.figure(figsize=(10,8))
        plt.plot(xl,yl, label='Left')
        plt.plot(xr, yr, label='Right')
        plt.plot(xlow, ylow, label='lower')
        plt.plot(xtop, ytop, label='Top')
        plt.legend()
        plt.title('Physical Domain')
        plt.show()
        plt.axis('equal')
        plt.close()

# circumferential grid
# radial grid

if __name__ == '__main__':
    cg, rg = 40, 40
    anulus = Annulus_Boundary(1, 0.7, -0.5)
    anulus.Plot_Boundary(cg,rg)
#
#
# class Annulus:
#     def __init__(self, r_outer, r_inner, eccentricity, nx, ny):
#
#         self.r_outer, self.r_inner = r_outer, r_inner
#         self.nx, self.ny = nx, ny
#         if r_outer <= r_inner:
#             raise ValueError('Outer circle should be greater than inner circle')
#
#         if eccentricity > 1.0:
#             raise ValueError('Eccentricity should not be greater than 1')
#         else:
#             self.ecc = eccentricity
#
#         self.x_inner = (self.ecc * (2 * self.r_outer - 2 * self.r_inner)) / 2
#         self.y_inner = 0
#
#     def outer_casing(self):
#         theta = self.r_outer * torch.linspace(0, np.pi, self.ny)
#         radius = self.r_outer * torch.linspace(0, np.pi, self.nx)
#         x = torch.cos(radius)
#         y = torch.sin(theta)
#         return x, y
#
#     def inner_pipe(self):
#         theta = torch.linspace(0, np.pi, self.ny)
#         radius = torch.linspace(0, np.pi, self.nx)
#         x = torch.cos(radius) * self.r_inner + self.x_inner
#         y = torch.sin(theta) * self.r_inner
#         return x, y
#
#     def branch_out_1(self):
#         outer_x, outer_y = self.outer_casing()
#         inner_x, inner_y = self.inner_pipe()
#         x = torch.linspace(outer_x[-1], inner_x[-1], self.nx)[::-1]
#         y = torch.zeros(self.ny)[::-1]
#         return x, y
#
#     def branch_out_2(self):
#         outer_x, outer_y = self.outer_casing()
#         inner_x, inner_y = self.inner_pipe()
#         x = np.linspace(outer_x[1], inner_x[1], self.nx)[::-1]
#         y = np.zeros(self.ny)[::-1]
#         return x, y
#
#     def plot(self):
#         outer_x, outer_y = self.outer_casing()
#         inner_x, inner_y = self.inner_pipe()
#         branch_1_x, branch_1_y = self.branch_out_1()
#         branch_2_x, branch_2_y = self.branch_out_2()
#
#         plt.figure(figsize=(10, 6))
#         plt.plot(outer_x, outer_y)
#         plt.plot(inner_x, inner_y)
#         plt.plot(branch_1_x, branch_1_y)
#         plt.plot(branch_2_x, branch_2_y)
#         plt.title('Boundary Conditions')
#         plt.axis('equal')
#         plt.show()
#
#
# class Analytical_Annulus:
#     def __init__(self, r_outer, r_inner, eccentricity):
#         """
#         Analytical Annulus maps the boundary values of a eccentric annulus from a computational domain into a physical
#          domain for performing computational analysis using analytical scheme.
#         ==>> The outer circle is positioned at the center of origin, while the inner circle position is calculated
#         from the eccentricity
#         :param r_outer: The radius of the outer circle
#         :param r_inner: The radius of the inner circle
#         :param eccentricity: The eccentricity describing the position of inner circle within outer circle
#         """
#         self.r_outer, self.r_inner = r_outer, r_inner
#         # condition to check if radius of outer circle is larger than inner
#         if r_outer <= r_inner:
#             raise ValueError('Outer circle should be greater than inner circle')
#
#         # condition to check if eccentricity is not greater than 1from
#         # the eccentricity should range between -1 to 1.
#         if eccentricity > 1.0:
#             raise ValueError('Eccentricity should not be greater than 1')
#         else:
#             self.ecc = eccentricity
#
#         # calculate the position of the inner circle wrt the eccentricity
#         self.x_inner = (self.ecc * (2 * self.r_outer - 2 * self.r_inner)) / 2
#         # The y axis for center of inner circle is assumed to be
#         self.y_inner = 0
#
#     def Xt(self, x):
#         """
#         Xt calculates the x-axis values in physical domain to implement top-boundary conditions
#         :param x: xi -- (X axis values in computational domain)
#         :return: x -- (X- axis values in physical domain)
#         """
#         out = (1 - x) * np.array([np.pi])
#         return np.cos(out)* self.r_outer
#
#     def Yt(self, x):
#         """
#         Yt calculates the y-axis values in physical domain to implement top-boundary conditions
#         :param x: eta -- (Y-axis values in computational domain)
#         :return: y -- (Y-axis values in physical domain)
#         """
#         out = (1 - x) * np.array([np.pi])
#         return np.sin(out)* self.r_outer
#
#     def Xb(self, x):
#         """
#         Xb calculates the x-axis values in physical domain to implement bottom-boundary conditions
#         :param x: xi -- (X axis values in computational domain)
#         :return: x -- (X- axis values in physical domain)
#         """
#         out = (1 - x) * np.array([np.pi])
#         return np.cos(out) * self.r_inner + self.x_inner
#
#     def Yb(self, x):
#         """
#         Yb calculates the y-axis values in physical domain to implement bottom-boundary conditions
#         :param x: eta -- (Y-axis values in computational domain)
#         :return: y -- (Y-axis values in physical domain)
#         """
#         out = (1 - x) * np.array([np.pi])
#         return np.sin(out) * self.r_inner
#
#     def Xl(self, x):
#         """
#         Xl calculates the x-axis values in physical domain to implement left-boundary conditions
#         :param x: xi -- (X axis values in computational domain)
#         :return: x -- (X- axis values in physical domain)
#         """
#         # interpolation ==>> (0,1)* diff + min
#         diff = -self.r_outer - (self.x_inner - self.r_inner)
#         out = diff * x + (self.x_inner - self.r_inner)
#         return out
#
#     def Yl(self, x):
#         """
#         Yl calculates the y-axis values in physical domain to implement left-boundary conditions
#         :param x: eta -- (Y-axis values in computational domain)
#         :return: y -- (Y-axis values in physical domain)
#         """
#         out = x * 0
#         return out
#
#     def Xr(self, x):
#         """
#         Xr calculates the x-axis values in physical domain to implement right-boundary conditions
#         :param x: xi -- (X axis values in computational domain)
#         :return: x -- (X- axis values in physical domain)
#         """
#         # below trick is implemented to cast any range of values
#         # diff = (max - min) ==>> (0,1)* diff + min
#         diff = self.r_outer - (self.x_inner + self.r_inner)
#         out = diff * x + (self.x_inner + self.r_inner)
#         return out
#
#     def Yr(self, x):
#         """
#         Yr calculates the y-axis values in physical domain to implement right-boundary conditions
#         :param x: eta -- (Y-axis values in computational domain)
#         :return: y -- (Y-axis values in physical domain)
#         """
#         out = x * 0
#         return out
#
#     def check(self):
#         """
#         Test function to check if all tye boundary conditions are met.
#         """
#         # the following four conditions should be satisfied for a logical boundary data implementation
#         assert self.Xb(0) == self.Xl(0), f'The bottom and left boundary should be in order'
#         assert self.Xb(1) == self.Xr(0), f'The bottom and right boundary should be in order'
#         assert self.Xr(1) == self.Xt(1), f'The right and top boundary should be in order'
#         assert self.Xl(1) == self.Xt(0), f'The left and top boundary should be in order'
#         print('PASS : Boundary Conditions in Order !!')
#
#     def __call__(self, cg, rg):
#         # circumferential grid (cg)
#         # radial grid (rg)
#         cg_x, cg_y = np.linspace(0,1,cg), np.linspace(0,1,cg)
#         rg_x, rg_y = np.linspace(0,1,rg), np.linspace(0,1,rg)
#         xl = self.Xl(rg_x)
#         yl = self.Yl(rg_y)
#         xr = self.Xr(rg_x)
#         yr = self.Yr(rg_y)
#         xlow = self.Xb(cg_x)
#         ylow = self.Yb(cg_y)
#         xtop = self.Xt(cg_x)
#         ytop = self.Yt(cg_y)
#         return {'left':np.stack((xl,yl)),
#                 'right':np.stack((xr,yr)),
#                 'top':np.stack((xtop, ytop)),
#                 'low':np.stack((xlow,ylow))}
#
#     def Plot_Boundary(self, cg, rg):
#         boundary = self.__call__(cg,rg)
#         xl, yl = boundary['left'][0,:], boundary['left'][1,:]
#         xr, yr = boundary['right'][0,:], boundary['right'][1,:]
#         xlow, ylow = boundary['low'][0,:], boundary['low'][1,:]
#         xtop, ytop = boundary['top'][0,:], boundary['top'][1,:]
#
#
#         plt.figure(figsize=(10,8))
#         plt.plot(xl,yl, label='Left')
#         plt.plot(xr, yr, label='Right')
#         plt.plot(xlow, ylow, label='lower')
#         plt.plot(xtop, ytop, label='Top')
#         plt.legend()
#         plt.title('Physical Domain')
#         plt.show()
#         plt.axis('equal')
#         plt.close()
#
#
# if __name__ == '__main__':
#     xi_ = np.linspace(0,1,40)
#     eta_ = np.linspace(0,1,40)
#     xi, eta = np.meshgrid(xi_, eta_)
#     anulus = Analytical_Annulus(1., 0.6, 0.)
#     (anulus.check())