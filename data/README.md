# Data Pipeline

Data generation for CNN learning

### Before generation

1. load pretrained CRAFT model
  - download https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
  - make directory named `weights` and load inside the file.

2. set fonts file

Some `.ttf` files should be loaded for text generation.

```
$ mkdir fonts
$ cd fonts
$ wget http://cdn.naver.com/naver/NanumFont/fontfiles/NanumFont_TTF_ALL.zip
$ unzip NanumFont_TTF_ALL.zip
```

## Data generation

### How to run the code?

```
$ python data_text.py
$ python data_transform.py

# Install torch and load pretrained weight before inference craft model.
$ python craft_inference.py
```

### 1. random text generation
  - The format of text position should be prepared explicitly before generate text.
  - The text format is dependent for format of document.
  - All result file is saved with numpy array format in `imgs` folder. For the usage ...

```python
import cv2
import pickle
import numpy as np

img = cv2.imread("imgs/origin/0.jpg") # shape = (height, width, 3)
y = np.load("imgs/origin_label/0.npy") # shape = (width / 2, height / 2)
with open("imgs/origin_bbox/0.pickle", "rb") as fin:
    bbox = pickle.load(fin) # list of bbox tuples
```

  - After run the `data_text.py`, the images would be saved in `imgs/origin` with each image files, and the label of the images would be saved in `imgs/origin_label` with numpy array.
  - Also, bounding box data would be saved in `imgs/origin_bbox`.
    - `imgs/origin`: All images are saved in .jpg format in `960*1280` size.
    - `imgs/origin_label`: Label of all pixels in each images. Label of numpy array has just one channnel, so the shape of the label is (480, 640). Lable has half size of the origin image for the craft model.
    - `imgs/origin_bbox`: All bounding box information of each images is saved with `pickle` object. Bounding boxes are shape with `[(label, x1, y1, x2, y2, x3, y3, x4, y4), ...]` that x1, y1 is upper left point of the box, x2, y2 is lower left point of the box, x3, y3 is lower right point of the box, and x4, y4 is upper right point of the box.

### 2. noise generation
  - All noises effect will be generated in random.
  - After run the `data_transform.py`, noised images would be saved in `imgs/origin_noise/` with image file, and the label of the images would be saved in `imgs/origin_noise_label/`.
  - Also, transformed bounding box data would be saved in `imgs/origin_noise_bbox`.


### 3. craft inference
  - Craft network inferences the text score and link score at once. We use the sum of the inferences.
  - Sum of the scores is [0, 1] float32 at (width/2, height/2) scale.
  - After run the `craft_inference.py`, score numpy array would be saved in `imgs/origin_craft/` folder.

