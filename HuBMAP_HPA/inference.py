# The following code is created on Kaggle. So you probably have to set differently the sys environment for your system.
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../input/einops/einops-master')
sys.path.append('../input/timmaster')
sys.path.append('../input/myPackage')

!pip install ../input/staintools-offline/spams-2.6.5.4-cp37-cp37m-linux_x86_64.whl &> /dev/null
!pip install ../input/staintools-offline/staintools-2.1.2-py3-none-any.whl &> /dev/null

import staintools
import tifffile
import cv2
import gc

from utils import *
from coat import *

class Net(nn.Module):
  def __init__(self,
               encoder=coat_small_plus,
               decoder=daformer_conv3x3,
               encoder_cfg={},
               decoder_cfg={}):
      super(Net, self).__init__()
      decoder_dim = decoder_cfg.get('decoder_dim', 320)
      self.ouput_type= ['inference']
      self.rgb = RGB()
      self.encoder = encoder
      encoder_dim = self.encoder.embed_dims
      
      self.decoder = decoder(encoder_dim=encoder_dim,
                             decoder_dim=decoder_dim)
      self.logit = nn.Sequential(nn.Conv2d(decoder_dim, 1, kernel_size=1))
      self.aux = nn.ModuleList([nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(5)])
