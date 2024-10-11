from __future__ import division
import numpy as np
from PolyAbs import PolyAbs


class SimpleCargo(PolyAbs):

    def __init__(self):

        super(SimpleCargo, self).__init__(1, index=-1)
        self.rigid_count = 0
        self.position[0] = [0, 0, 0]
        self.mass[0] = 1
        self.type[0] = 'C'

    def is_ion(self, ind1):

        return False
