#!/bin/sh
docker run --gpus all -it --rm -p 5901:5901 -p 6080:6080 --shm-size=512M -v ~/Projects/matlab:/home/matlab/matlab mymatlab:r2021b -vnc
