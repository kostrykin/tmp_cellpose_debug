import numpy as np
import skimage.io
from cellpose import models
import torch


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

gpu = False
model_type = 'cyto2'
channels = [2, 3]

options = {
    "diameter": None,
    "resample": True,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "niter": None,
    "do_3D": False,
    "rescale": None,
    "invert": False,
}


img = skimage.io.imread('img02.png')

model1 = models.CellposeModel(gpu=gpu, model_type=model_type)
masks1, flows1, styles1 = model1.eval(img, channels=channels, **options)

model2 = models.Cellpose(gpu=gpu, model_type=model_type)
masks2, flows2, styles2, _ = model2.eval(img, channels=channels, **options)

assert np.allclose(masks1, masks2), f'{masks1.sum()}, {masks2.sum()}'
assert np.allclose(flows1, flows2)
