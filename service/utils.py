import yaml 
import cv2
from datetime import datetime
import numpy as np
from PIL import Image
from io import BytesIO


def read_config(filename):
    with open(filename, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg


def write_json(file_name, data):
    import json

    with open(file_name, "w") as f:
        json.dump(data,f)


def read_json(filename):
    import json 
    
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def crop_person(img, dets, min_width=100, min_height=300, person_ratio=0.75):
    person_imgs, person_dets = [], []
    if len(dets) == 0:
        person_dets = np.empty((0,6))
    else:
        for det in dets:
            left, top, right, bottom, _, _ = det
            width = right - left
            height = bottom - top
            if width > min_width and height > min_height and width / height < person_ratio:
                person_img = img[int(top):int(bottom), int(left):int(right)]
                person_imgs.append(person_img)
                person_dets.append(det)
    if len(person_dets) == 0:
        return np.empty((0,6)), []
    return np.array(person_dets), person_imgs


def get_current_time(format_datetime):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime(format_datetime)
    return formatted_datetime



def padding(data):
    target_height = max(image['frame'].shape[0] for image in data)
    target_width = max(image['frame'].shape[1] for image in data)

    for frame_data in data:
        image = frame_data['frame']
        pad_height = target_height - image.shape[0]
        pad_width = target_width - image.shape[1]
        padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame_data['frame'] = padded_image
    return data


def convert_to_byteio(image_array):
    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to an image object
    image = Image.fromarray(image_array_rgb)

    # Create a BytesIO object to temporarily hold the image data
    image_bytesio = BytesIO()
    image.save(image_bytesio, format='JPEG')  # Specify the desired image format (e.g., JPEG)

    # Reset the file pointer of the BytesIO object to the beginning
    image_bytesio.seek(0)
    return image_bytesio


def convert_datetime_2_timestamp(date_string):
    # Convert the string to a datetime object
    date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")

    # Get the timestamp (in seconds since the epoch)
    timestamp = date_object.timestamp()
    return timestamp


# paths = ['/storage/thangdv/dataset/reid-benchmark-1/campus-7-team.mp4-1689852844-51.jpg', 
#          '/storage/thangdv/dataset/reid-benchmark-1/campus-7-team.mp4-1689852844-41.jpg',
#          '/storage/thangdv/dataset/reid-benchmark-1/campus-6-team.mp4-1689852636-11.jpg',
#          ]
# data = [{'frame': cv2.imread(path), 'path': path} for path in paths]
# padded_data = padding(data)
# import os
# for frame_data in padded_data:
#     image = frame_data['frame']
#     cv2.imwrite(os.path.basename(frame_data['path']), image)
#     print(frame_data['path'], image.shape)