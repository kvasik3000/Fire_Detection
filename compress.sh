#!/bin/bash

ffmpeg -i fire001_inf_v5.mp4 -vcodec h264 -hide_banner -loglevel error -crf 28 fire001_v5200.mp4