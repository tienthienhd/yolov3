
from darknet import parse_cfg, create_modules
from pprint import pprint

blocks = parse_cfg("cfg/yolov3.cfg")
# pprint(blocks)
# print(create_modules(blocks))

