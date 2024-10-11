from __future__ import division
import numpy as np
from PduABody import PduABody


class PduAOneSided(PduABody):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduAOneSided, self).__init__(index=0, with_blocker=with_blocker)
        self.types[17] = "AX"
        self.types[18] = "AX"
        self.types[23] = "AX"
        self.types[24] = "AX"
        self.types[29] = "AX"
        self.types[30] = "AX"
        self.types[26] = "AX"
        self.types[27] = "AX"
        self.types[20] = "AX"
        self.types[21] = "AX"

        self.types[3] = "AX"
        self.types[4] = "AX"
        self.types[5] = "AX"
        self.types[6] = "AX"
        #self.types[2] = "AX"

