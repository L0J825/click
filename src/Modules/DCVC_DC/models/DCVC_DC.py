import torch.nn as nn

from Modules.DCVC_DC.models.video_model import DMC
from Modules.DCVC_DC.models.image_model import IntraNoAR


class DCVC_DC(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.intra_compressor = IntraNoAR(ec_thread=args.ec_thread,
                                          stream_part=args.stream_part_i,
                                          inplace=True)
        self.inter_compressor = DMC(ec_thread=args.ec_thread,
                                    stream_part=args.stream_part_p,
                                    inplace=True)

        self.args = args
        self.intra_compressor.update()
        self.inter_compressor.update()
