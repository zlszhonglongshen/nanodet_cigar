import time
import os
import cv2
import torch

from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline

config_path = 'nanodet_card.yml'
model_path = 'model_last.pth'
image_path = '000-0.jpg'


load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,raw_img=img,img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))
        
        
predictor = Predictor(cfg, model_path, logger, device='cpu')


from nanodet.util import overlay_bbox_cv

from IPython.display import display
from PIL import Image

def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))

frame = cv2.imread("000-0.jpg")
meta, res = predictor.inference(frame)
result = overlay_bbox_cv(meta['raw_img'], res, cfg.class_names, score_thresh=0.35)

imshow_scale = 1.0
cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))
