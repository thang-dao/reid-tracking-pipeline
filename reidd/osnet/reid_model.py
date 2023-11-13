import os
import sys
current_file = __file__
parent_directory = os.path.dirname(os.path.abspath(current_file))
sys.path.pop(0)
sys.path.insert(0, parent_directory)
from pathlib import Path

import torch
import torch.nn.functional as F
from reidd.model_base import BaseReidModel
from boxmot.deep.reid_multibackend import ReIDDetectMultiBackend


class OsnetReid(BaseReidModel):
    def __init__(self, model_cfg, device):
        super().__init__(model_cfg, device)
        self.model = ReIDDetectMultiBackend(Path(model_cfg.get('WEIGHTS')), device)

    def load_model(self):
        pass

    def run(self, img_batch):
        with torch.no_grad():
            feat = self.model(img_batch)
            return feat.cpu().detach().numpy()
            # normalized_feat = F.normalize(feat, p=2, dim=1)
            # return normalized_feat.cpu().detach().numpy()