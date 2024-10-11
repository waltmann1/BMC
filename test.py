from  __future__ import division
import sys
sys.path.append('/home/cwj8781/Filtration')
sys.path.append('/home/waltmann/PycharmProjects/Filtration')

from PduABody import PduABody
from Mer import Mer

a = Mer(PduABody())
a.dump_xyz()
