from PIL import Image
import torch
import numpy as np

def load_image(image_path) :
    image = Image.open(image_path)
    img_width = 416
    img_height = 128
    img = np.array(image.resize((img_width, img_height)))
    imgt = np.transpose(img, (2, 0, 1))
    tensor_img = torch.from_numpy(imgt.astype(np.float32)).unsqueeze(0)
    tensor_img = ((tensor_img/255 - 0.5)/0.5)
    return tensor_img, img
