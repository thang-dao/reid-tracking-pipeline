import os
import sys
current_file = __file__
parent_directory = os.path.dirname(os.path.abspath(current_file))
sys.path.pop(0)
sys.path.insert(0, parent_directory)
import torch
from reidd.centroids_reid.config import cfg
import torchvision.transforms as T
from reidd.centroids_reid.train_ctl_model import CTLModel
from reidd.model_base import BaseReidModel


class CentroidsReid(BaseReidModel):
    def __init__(self, model_cfg, device):
        super().__init__(model_cfg, device)
        cfg.TEST.ONLY_TEST = True
        cfg.MODEL.PRETRAIN_PATH = model_cfg['MODEL']['PRETRAIN_PATH']
        cfg.NORMALIZE_WITH_BN = model_cfg['NORMALIZE_WITH_BN']

    def create_transform(self, ):
        normalize_transform = T.Normalize(mean=self.cfg['INPUT']['PIXEL_MEAN'], std=self.cfg['INPUT']['PIXEL_STD'])
        self.transform = T.Compose([
                T.Resize(self.cfg['INPUT']['SIZE_TEST']),
                T.ToTensor(),
                normalize_transform
            ])
        
    def load_model(self):
        self.model = CTLModel.load_from_checkpoint(
            cfg.MODEL.PRETRAIN_PATH,
            cfg=cfg,
            num_query=3368,
            num_classes=751,
        )
        self.use_cuda = True if torch.cuda.is_available() and self.device else False
        self.model = self.model.to(self.device)
        self.model.eval()

    def run(self, batch_data):
        with torch.no_grad():
            trs_data = self.apply_transform(batch_data)
            _, global_feat = self.model.backbone(
                trs_data.cuda() if self.use_cuda else trs_data
            )
            if cfg.NORMALIZE_WITH_BN:
                global_feat = self.model.bn(global_feat)
            return global_feat.cpu().numpy()