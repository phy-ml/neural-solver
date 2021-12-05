import numpy as np

class Dimensional:
    def __init__(self, radius, K, n, density):
        self.radius = radius
        self.k = K
        self.n = n
        self.rho = density
        #self.rho = 1000 #if units are in kg/m3 or 8.33 in ppg
        self.pow = 1/(2-self.n)

    def non_dim_radius(self, radius):
        return radius/self.radius

    def non_dim_velocity(self, velocity):
        const = (self.k/ (self.rho * (self.radius**self.n) ) )**self.pow
        return velocity/const

    def non_dim_press(self, press):
        k = self.k**2
        rho = self.rho**(self.n)
        rad = self.radius**(self.n+2)
        const = (k/(rho*rad))**self.pow
        return press/const

    def non_dim_yield(self, yield_stress):
        k = self.k ** 2
        rho = self.rho ** (self.n)
        rad = self.radius ** (self.n * 2)
        const = (k / (rho * rad)) ** self.pow
        return yield_stress/const

    def non_dim_shear(self, shear):
        const = (self.k/(self.rho * self.radius**2))**self.pow
        return shear/const

    def non_dim_vis(self, vis):
        rho = self.rho**(self.n -1)
        rad = self.radius**(2*self.n - 2)
        const = (self.k/(rho*rad))**self.pow
        return vis/ const

    def non_dim_constrain(self, m):
        const = (self.k/( self.rho * self.radius**2 ))**self.pow
        return m*const

    def non_dim_volume(self, vol):
        const = ((self.k / (self.rho * (self.radius**self.n))) ** self.pow)*self.radius**2
        return vol/const

    # check it again!!
    def dim_rad(self, dim_radius):
        return self.radius*dim_radius

    def dim_velocity(self, non_dim_vel):
        const = (self.k / (self.rho * (self.radius ** self.n))) ** self.pow
        return const*non_dim_vel

    def dim_press(self, non_dim_press):
        k = self.k ** 2
        rho = self.rho ** (self.n)
        rad = self.radius ** (self.n * 2)
        const = (k / (rho * rad)) ** self.pow
        return const*non_dim_press

    def dim_yield(self, non_dim_yield):
        k = self.k ** 2
        rho = self.rho ** (self.n)
        rad = self.radius ** (self.n * 2)
        const = (k / (rho * rad)) ** self.pow
        return const * non_dim_yield

    def dim_shear(self, non_dim_shear):
        const = (self.k / (self.rho * self.radius ** 2)) ** self.pow
        return const * non_dim_shear

    def dim_vis(self, non_dim_vis):
        rho = self.rho ** (self.n - 1)
        rad = self.radius ** (2 * self.n - 2)
        const = (self.k / (rho * rad)) ** self.pow
        return const*non_dim_vis

    def dim_constrain(self, non_dim_cons):
        const = (self.k / (self.rho * self.radius ** 2)) ** self.pow
        return non_dim_cons/const

    def dim_vol(self, non_dim_vol):
        const = ((self.k / (self.rho * (self.radius ** self.n))) ** self.pow) * self.radius ** 2
        return non_dim_vol*const
