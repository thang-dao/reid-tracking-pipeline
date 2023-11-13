from abc import ABC, abstractmethod
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch


class BaseReidModel(ABC):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.create_transform()
        
    def create_transform(self, ):
        pass
    
    def apply_transform(self, batch):
        data = [Image.fromarray(np_array) for np_array in batch]
        transformed_data = [self.transform(pil_image) for pil_image in data]
        batch_data = torch.stack(transformed_data)
        return batch_data
        
    @abstractmethod
    def load_model(self,):
        pass

    @abstractmethod
    def run(self, x):
        pass
