from __future__ import division
import numpy as np
from PduABody import PduABody


class PduNHEXBody(PduABody):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduNHEXBody, self).__init__(index=0, with_blocker=with_blocker)
        for ind, tipe in enumerate(self.types):
            if len(tipe) > 1:
                self.types[ind] = "N" + tipe[1:]