# sightsprite
Real-time machine vision with voice alerts. 

Fun with machine vision. Contains
- OpenCV utilities for capturing training data. 
- Tools for training models (will use deepglue)
- Realtime inference with verbal nudges (OpenCV and Pytorch model)

This is in pre-alpha stage, all of it, including the readme. I'm currently finishing up the basic capture utilities in `capture.py` so I can get training data. 

## To do
- Get `capture.py` in slightly better shape
  - add logging instead of print
  - fix up the weird Qt warnings
  - capture more data 
  - Make some simple tests 
- Train on sleep data (use deepglue), build out `training.py` module
- Once you have model trained, build out out `inference.py` module
