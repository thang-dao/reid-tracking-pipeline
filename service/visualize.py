import os
import sys
import time
import json
import ast
from datetime import datetime
import cv2
import numpy as np
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors
from boxmot import create_tracker
from service.reid import ReIDModel
from service.detect import DetectionModelWrapper
from search.milvus import MilvusDBWapper
from search.faiss_search import FaissSearching
import settings
from service.utils import read_config, crop_person, get_current_time


def draw_result(annotator, outputs):
    for j, (output) in enumerate(outputs):
        id = output[-1]
        if id == -1:
            id = output[1]
        bbox = output[0]
        cls = output[2]
        conf = output[3]
        if isinstance(id, str):
            color = colors(int(id.split('-')[0]), True)
        else:
            color = colors(int(id), True)
        annotator.box_label(bbox, str(id), color=color)


def main_f(video_path, tracking_method, tracking_config):
    print(video_path)
    source = video_path.split("/")[-1]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(f'visualize-{video_path.split("/")[-1]}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # tracker_camera = create_tracker(tracking_method, tracking_config)
    tracker_camera = tracker_camera_dict[source]
    similarity_thresh = settings.FAISS_CFG['DISTANCE_THRESH']
    db = []
    c = 0
    while True:
        c += 1
        ret, frame = cap.read()
        if not ret:
            break
        annotator = Annotator(frame, line_width=2, example=str(1))
        detected_result = detector.run([{'meta_data': [], 'frame': frame}])
        dets = detected_result[0]['dets']
        if not dets:
            video_writer.write(frame)
            continue

        person_dets, persons_represents = crop_person(frame, dets, source)
        curr_tracks, undefine_trackers, dead_tracks = tracker_camera.update(
            person_dets, persons_represents)
        absol_entities = []
        absol_feats = []
        appeared_entities = []
        if undefine_trackers:
            print('\nUndefine tracker: ', len(undefine_trackers))
            represents = [trk_inf[2] for trk_inf in undefine_trackers]
            feat_trks = reid_model.run(represents)
            similalities, indexes = searcher.search(feat_trks)
            print(similalities, indexes)
            for i, (trk, feat, similarity, ids) in enumerate(zip(undefine_trackers, feat_trks, similalities, indexes)):
                if ids[0] == -1:
                    absol_entities.append(trk[:2])
                    absol_feats.append(feat)
                    print('Create new tracker')
                    # cv2.imwrite(f'data/gallery/{cam_id}_{c}_{i}.jpg', represents[i])
                else:
                    if similarity[0] < similarity_thresh:
                        print('top 1 search: ', similarity[0])
                        absol_entities.append(trk[:2])
                        absol_feats.append(feat)
                        print('Create new tracker')
                        # cv2.imwrite(f'data/sample/{source}_{c}_{similarity[0]}.jpg', represents[i])
                    else:
                        print('top 1 search: ', ids[0])
                        print('Recognize this id: ', ids[0])
                        trk = trk[:2] + (ids[0],)
                        appeared_entities.append(trk)
                        cv2.imwrite(f'data/samples/{source}_{c}_{ids[0]}_{similarity[0]}.jpg', represents[i])
        if absol_entities:
            absol_ids = []
            for feat in absol_feats:
                id, _ = searcher.insert(np.stack(absol_feats))
                absol_ids.append(id)
            # absol_ids = [int(str(int(time.time()))) for _ in range(len(absol_feats))]
            # print(absol_ids)
        else:
            absol_ids = []
        tracker_camera.append_new_tracker(absol_entities, absol_ids, appeared_entities)

        updated_trackers = tracker_camera.get_current_trackers()
        draw_result(annotator, updated_trackers)
        im0 = annotator.result()
        video_writer.write(im0)
        cv2.imwrite('vis.jpg', im0)
    video_writer.release()


def get_results(video_path, timestamp, bbox):
    print(video_path, timestamp, bbox)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = int(timestamp * fps)
    print(frame_index)
    c = -1
    while c < frame_index:
        ret, img = cap.read()
        if not ret: break
        c += 1
    if c == frame_index:
        ret, img = cap.read()
        print('reach frame index: ', frame_index)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, img = cap.read()

    xyxy = ast.literal_eval(bbox)
    # xyxy = bbox
    pedestrian = img[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    cv2.imwrite('loop.jpg', img)

    cap.release()
    return pedestrian


def get_results_seeking(video_path, timestamp, bbox):
    print(video_path, timestamp, bbox)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = int(timestamp * fps)
    # Set the desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame at the desired index
    ret, frame = cap.read()
    cv2.imwrite('seek.jpg', frame)
    xyxy = ast.literal_eval(bbox)
    # xyxy = bbox
    pedestrian = frame[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    cap.release()
    return pedestrian


def visualize_searching_results(data_path):
    num_vectors = 1
    num_dimensions = 768
    inserted_vectors = np.random.rand(num_vectors, num_dimensions)
    milvus_db = MilvusDBWapper(settings.MILVUS_DB_CFG)
    search_results = milvus_db.search(inserted_vectors)
    print(search_results)
    predictions = []
    for i, query_result in enumerate(search_results):
        for i,result in enumerate(query_result[:1]):
            print(result)
            source = result.entity.get('source')
            trk_id = result.entity.get('trackid')
            bbox = result.entity.get('bbox')
            timestamp = result.entity.get('timestamp')
            url = result.entity.get('path')
            print(url)
            # pedestrian = get_results(source, timestamp, bbox)
            # pedestrian = cv2.resize(pedestrian, (150, 300))
            # predictions.append(pedestrian)
    merged_frame = np.concatenate(predictions, axis=1) 
    cv2.imwrite('pred.jpg', merged_frame)


if __name__ == '__main__':
    cfg_path = 'detection/yolov8s.yml'
    cfg = read_config(cfg_path)
    detector = DetectionModelWrapper(model_name='YoloV8', cfg=cfg, device='cuda:0')
    # model_name = 'osnet_ain_x1_0'
    # cfg = 'boxmot/deep/configs/osnet.yaml'
    # device = 'cuda:0'
    # reid_model = ReIDModel(model_name, cfg, device)
    # model_name = 'soldier'
    # cfg = "reidd/soldier/swin_tiny.yml"
    device = 'cuda:0'
    # reid_model = ReIDModel(model_name, cfg, device)
    model_name = 'efficientnet'
    reid_model = ReIDModel(model_name, settings.REID_MODEL_DICT[model_name], device)
    searcher = MilvusDBWapper(settings.MILVUS_DB_CFG)
    # searcher = FaissSearching(settings.FAISS_CFG)
    video_paths = ['/storage/thangdv/dataset/videos/campus-5-team-1m25-2m05.mp4',
                '/storage/thangdv/dataset/videos/campus-5-team.mp4',
                '/storage/thangdv/dataset/videos/campus-4-team.mp4',
                '/storage/thangdv/dataset/videos/campus-6-team.mp4',
                '/storage/thangdv/dataset/videos/campus-7-team.mp4']
    tracker_camera_dict = {source.split('/')[-1]: create_tracker(settings.TRACKING_METHOD, settings.TRACKING_CFG, tracker_name=source.split('/')[-1]) for source in video_paths}
    results = {source.split('/')[-1]: set() for source in video_paths}
    # for video_path in video_paths:
        # main(video_path, settings.TRACKING_METHOD, settings.TRACKING_CFG)
    # insert_data(video_paths[0])
    # for video_path in video_paths[1:]:
        # search_data(video_path)
    # print(searcher.get_collection_inf())
    data_path = '/storage/thangdv/dataset/videos'
    visualize_searching_results(data_path)