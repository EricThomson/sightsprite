# sightsprite 
Real-time machine vision with voice alerts. :sparkles: 

Fun with machine vision. Will contain:
- OpenCV utilities for capturing training data. 
- Tools for training pytorch models (will use deepglue)
- Utilities for realtime inference with verbal nudges

This is in pre-alpha stage, all of it (including the readme). I'm currently finishing up the basic capture utilities in `capture.py` so I can get training data. 

## To do
1. Get `capture.py` in slightly better shape
   - add logging instead of print
   - fix up the weird Qt warnings
   - capture more data 
   - Make some simple tests 
2. Train on sleep data (use deepglue), build out `training.py` module
3. Once you have model trained, build out out `inference.py` module
