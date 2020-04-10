from darknet import parse_cfg, create_modules
from pprint import pprint
import torch
from darknet import get_test_input, Darknet

model = Darknet("cfg/yolov3.cfg")
model.load_weights('cfg/yolov3.weights')
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print(pred.shape)
