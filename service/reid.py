import logging
from reidd.reid_model_zoo import create_reid_model


class ReIDModel:
    def __init__(self, model_name, cfg, device):
        print(model_name, cfg)
        logging.info('Loading reid model {}, {}'.format(model_name, cfg))
        self.model = create_reid_model(reid_model=model_name, reid_cfg=cfg, device=device)
    
    def run(self, data):
        output = self.model.run(data)
        return output


if __name__ == "__main__":
    from reidd.reid_model_zoo import create_reid_model
    import cv2
    import time
    import numpy as np
    import settings

    batch_size = 32
    model_name = 'convnext_novelty'
    reid_model = ReIDModel(model_name, settings.REID_MODEL_DICT[model_name], "1")
    img_path = 'data/16:49:38.611540.jpg'
    img = cv2.imread(img_path)
    for _ in range(50):
        feat1 = reid_model.run([img for _ in range(batch_size)])
        print(feat1.shape)
    
    # batch_size = 30
    # inference_time = []
    # for _ in range(50):
    #     data = [img for _ in range(batch_size)]
    #     s = time.time()
    #     feat = reid_model.run(data)
    #     e = time.time()
    #     inference_time.append(e-s)
    #     print(feat.shape, e-s)

    # print('average time inference: ', sum(inference_time) / len(inference_time))