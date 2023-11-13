import logging
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from reidd.model_base import BaseReidModel


class NvidiaReidModel(BaseReidModel):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.client = InferenceServerClient(url=cfg['TRITON_SERVER_URL'])
    
    def load_model(self):
        if not self.client.is_server_live():
            logging.info("Triton Server isn't live")
            return False
        elif not self.client.is_model_ready(self.cfg['MODEL_NAME']):
            logging.info("Model {} isn't ready".format(self.cfg['MODEL_NAME']))
            return False
        return True

    def apply_transform(self, batch):
        transformed_images = []
        for image in batch:
            image = cv2.resize(image, (128, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            pixel_mean = np.array([0.485, 0.456, 0.406])
            pixel_std = np.array([0.226, 0.226, 0.226])
            image = (image - pixel_mean) / pixel_std
            image = image.transpose(2, 0, 1).astype(np.float32)
            transformed_images.append(image)

        transformed_images = np.stack(transformed_images, axis = 0)    
        
        return transformed_images

    def run(self, batch_data):
        input_image = self.apply_transform(batch_data)
        inputs = [InferInput('input', input_image.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [InferRequestedOutput('fc_pred')]

        response = self.client.infer(self.cfg['MODEL_NAME'], inputs, outputs=outputs)

        embs = response.as_numpy('fc_pred')
        return normalize(embs)