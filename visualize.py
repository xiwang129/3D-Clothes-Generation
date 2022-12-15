import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from model.model_test import Model
from configs import opts

if __name__ == '__main__':

    opts.pretrain_model_G = "50_tshirt_G.pth"
    opts.log_dir = "log/20221110-1006/"

    model = Model(opts)
    model.draw_correspondense() # draw the correspondense between sphere and shape
 