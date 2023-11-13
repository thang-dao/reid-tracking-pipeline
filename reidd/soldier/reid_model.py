import os
import sys
import sys
sys.path.append('../')
from dotenv import load_dotenv
from pathlib import Path
import torch
import torch.nn.functional as F
from reidd.model_base import BaseReidModel
from boxmot.deep.reid_multibackend import ReIDDetectMultiBackend
from embedding_model import make_model
from config import cfg
import cv2 
from torchvision import transforms
load_dotenv(dotenv_path='env/.env')


class SoldierReid(BaseReidModel):
    def __init__(
        self, 
        model_cfg,
        device,
        ):
        num_class = 100, 
        camera_num = 3, 
        num_view = 3, 
        cfg.merge_from_file(os.getenv("SWIN_TINY_CFG"))
        self.model = make_model.make_model(cfg, 
                                           num_class= num_class, 
                                           camera_num= camera_num, 
                                           view_num = num_view, 
                                           semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
        self.model.load_param(model_cfg['weights'])
        self.model.eval()
        self.device = device
        self.model.cuda()
    
    def _apply_transform(self, batch):
        images = []
        for image in batch:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.ToPILImage(),  # Convert numpy array to PIL image
                transforms.Resize((384, 128)),  # Resize the image to a desired size
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the image
            ])

            transformed_image = transform(image)
            images.append(transformed_image)
        
        images = torch.stack(images, dim = 0)    
        
        return images.cuda()
        
    def load_model(self):
        pass 
    
    def run(self, img_batch):
        img_batch = self._apply_transform(img_batch)
        with torch.no_grad():
            feat, masks = self.model(img_batch)
            normalized_feat = F.normalize(feat, p=2, dim=1)
            
        return normalized_feat.cpu().detach().numpy()