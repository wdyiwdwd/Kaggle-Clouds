#!/bin/sh

./run.py --test --image dataset/0a14f2b.jpg --features-init 90 --model-type shl --save-sized-masks
./run.py --test --image dataset/00a0954.jpg --features-init 90 --model-type shl --save-sized-masks
./run.py --test --image dataset/0a1b596.jpg --features-init 90 --model-type shl --save-sized-masks

