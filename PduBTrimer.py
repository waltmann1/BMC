from __future__ import division
import numpy as np
from PduATrimer import PduATrimer


class PduBTrimer(PduATrimer):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduBTrimer, self).__init__(index=0, with_blocker=with_blocker)
        for ind, tipe in enumerate(self.types):
            if len(tipe) > 1:
                self.types[ind] = "B" + tipe[1:]