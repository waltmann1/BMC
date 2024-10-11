from __future__ import division
import numpy as np
from PduABody import PduABody


class PduATrimer(PduABody):
    """ PduA protein
    points: a list containing positions of points from the body
    charges: a list of charges
    """

    def __init__(self, index=0, with_blocker=False):
        super(PduATrimer, self).__init__(index=0, with_blocker=with_blocker)
        self.types[17] = "A6"
        self.types[18] = "A7"
        self.types[23] = "A6"
        self.types[24] = "A7"
        self.types[29] = "A6"
        self.types[30] = "A7"
        #for ind, tipe in enumerate(self.types):
            #print(ind, tipe)
        #quit()
            #if len(tipe) > 1:
                #self.types[ind] = "B" + tipe[1:]