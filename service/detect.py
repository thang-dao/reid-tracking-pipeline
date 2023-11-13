from service.utils import read_config
from detection.models import create_detection_model


class DetectionModelWrapper:
    def __init__(self, model_name, cfg, device):
        self.choosen_cls = cfg.get('CLASS_IDS')
        self.model = create_detection_model(model_name, cfg, device)

    def parse_input2batch(self, inputs):
        """
        """
        meta_data_batch, frames = [], []
        for frame_data in inputs:
            meta_data = frame_data['meta_data']
            frame = frame_data['frame']
            meta_data_batch.append(meta_data)
            frames.append(frame)
        return frames, meta_data_batch 
    
    def _post_process(self, frame, dets):
        bboxes = [(det[0], det[1], det[2], det[3], det[4], int(det[5]))
                  for det in dets if int(det[5]) in self.choosen_cls]
        return bboxes

    def run(self, batch_input):
        batch_frames, batch_meta_data = self.parse_input2batch(batch_input)
        batch_dets = self.model.predict(batch_frames)
        posted_dets = [{'meta_data': meta_data, 'dets': self._post_process(frame, dets)} for frame, meta_data, dets in zip(batch_frames, batch_meta_data, batch_dets)]
        return posted_dets


# if __name__ == '__main__':
#     import cv2

#     cfg_path = 'detection/yolov8s.yml'
#     cfg = read_config(cfg_path)
#     model = DetectionModelWrapper(model_name='YoloV8', cfg=cfg, device='cuda:1')
#     img_path = 'retail.jpg'
#     batch_size = 1
#     img = cv2.imread(img_path)
#     o = model.run([img for _ in range(batch_size)])
#     print(o)