"""Define the vertex mass resolution as a function of mass"""

# dataclasses handle a lot of the boiler plate for you
from dataclasses import dataclass
from typing import List

@dataclass
class polynomial:
    """General class to store a polynomial by its coefficients in order"""
    coefficients: List[float]

    def __call__(self, x):
        """Calculate the polynomial"""
        return sum(c*x**i for i, c in enumerate(self.coefficients))

    @property
    def order(self):
        return len(self.coefficients)-1


#mass_resolution_2016_bump_hunt = polynomial([0.00032, 0.019, -0.11, 1.39, -4.33])
mass_resolution_2016_bump_hunt = [-0.00243101086956522, 0.144611940993789, -0.43936801242236, -2.5, 10]
"""the polynomial function for the mass resolution given in 2016 Bump Hunt paper

the input and output units are GeV"""
