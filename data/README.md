## Data Pipeline

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

### Data generation

#### How to run the code?

```
$ python data_text.py
$ python data_transform.py

# Install torch and load pretrained weight before inference craft model.
$ python craft_inference.py
```

1. random text generation
- The format of text position should be prepared explicitly before generate text.
- The text format is dependent for format of document.
- All result file is saved with numpy array format in `imgs` folder. For the usage ...

```python
import numpy as np

imgs = np.fromstring(open("imgs/origin/0"), "rb").read(), dtype=np.uint8)
imgs = imgs.reshape(-1, height, weight, 3)
```

- After run the `data_text.py`, the images would be saved in `imgs/origin` with numpy array str, and the label of the images(also shape with full image) would be saved in `imgs/origin_label` with numpy array str.

2. noise generation
- All noises effect will be generated in random.
- After run the `data_transform.py`, noised images would be saved in `imgs/origin_noise/` with numpy array str, and the label of the images(also shape with full image) would be saved in `imgs/origin_noise_label/` with numpy array str.


3. craft inference
- Craft network inferences the text score and link score at once. We use the sum of the inferences.
- Sum of the scores is [0, 1] float32 at (width/2, height/2) scale.
- After run the 'craft_inference.py', score numpy array would be saved in 'imgs/origin_craft/' folder within batch scale.

