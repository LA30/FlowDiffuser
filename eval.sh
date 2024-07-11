#!/bin/bash
python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=sintel
python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=kitti