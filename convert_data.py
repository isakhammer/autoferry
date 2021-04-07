import base64
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import urllib
import PIL.Image as Image
import cv2
import torch
import os
import torchvision
from IPython.display import display
from sklearn.model_selection import train_test_split
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

def b64_to_img(b64):
    im_b64 = sample["imageData"]
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

def show_img(sample):
    sample_shapes = sample["shapes"][0]
    im_b64 = sample["imageData"]
    b64_to_img(im_b64)
    # decoding the image based on base64

    # Annotation
    w = sample['imageWidth']
    h = sample['imageHeight']
    points = sample_shapes['points']

    p1, p2 = points

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    label="0"

    cv2.rectangle(
      img,
      (int(x1), int(y1)),
      (int(x2), int(y2)),
      color=(0, 255, 0),
      thickness=2
    )

    ((label_width, label_height), _) = cv2.getTextSize(
        label,
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1.75,
        thickness=2
    )
    cv2.rectangle(
      img,
      (int(x1), int(y1)),
      (int(x1 + label_width + label_width * 0.05), int(y1 + label_height + label_height * 0.25)),
      color=(0, 255, 0),
      thickness=cv2.FILLED
    )
    cv2.putText(
      img,
      label,
      org=(int(x1), int(y1 + label_height + label_height * 0.25)), # bottom left
      fontFace=cv2.FONT_HERSHEY_PLAIN,
      fontScale=1.75,
      color=(255, 255, 255),
      thickness=2
    )

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def get_json_file_paths():
    # Extract all json file paths with corresponding image path.
    cwd = os.getcwd()
    data_path = cwd  + "/label_data"
    json_file_paths = []
    image_file_paths = []
    for subdir, dirs, files in os.walk(cwd):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".json"):
                json_file_paths.append(filepath)
            if filepath.endswith(".png"):
                image_file_paths.append(filepath)
    return json_file_paths

def load_data(json_file_paths):
    # Gather samples and find all categories
    categories = []
    data = []
    for path in json_file_paths:
        with open(path) as data_file:
            sample = json.load(data_file)
            data.append(sample)
            if "shapes" in sample and len( sample["shapes"] ) > 0:
                shape = sample["shapes"][0]
                categories.extend(shape["label"])
    categories = list(set(categories)) # removing duplicates from list

    # Separate test and train data
    train_data, val_data = train_test_split(data, test_size=0.1)
    return train_data, val_data, categories

def create_dataset(data, categories, dataset_type):
    images_path = Path(f"data/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path = Path(f"data/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exit_ok=True)
    for img_id, row in enumerate(tqdm(data)):
        print(img_id, row)

    # image_name = f"{img_id}.jpeg"
    # img = urllib.request.urlopen(row["content"])
    # img = Image.open(img)
    # img = img.convert("RGB")
    # img.save(str(images_path / image_name), "JPEG")
    # label_name = f"{img_id}.txt"
    # with (labels_path / label_name).open(mode="w") as label_file:
    #   for a in row['annotation']:
    #     for label in a['label']:
    #       category_idx = categories.index(label)
    #       points = a['points']
    #       p1, p2 = points
    #       x1, y1 = p1['x'], p1['y']
    #       x2, y2 = p2['x'], p2['y']
    #       bbox_width = x2 - x1
    #       bbox_height = y2 - y1
          # label_file.write(
          #   f"{category_idx} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"

if __name__=="__main__":
    train_data, val_data, categories = load_data(get_json_file_paths())

    # Finding a sample which is labelled
    index = None
    for i in range(len(train_data)):
        row = train_data[i]
        if len(row["shapes"]) > 0:
            index = i
    sample = train_data[index]

    show_img(sample)



