# sightsprite 
<img src="https://raw.githubusercontent.com/EricThomson/sightsprite/main/src/sightsprite/assets/sightsprite_logo.jpg" alt="sightsprite logo" align="right" width="200">
Real-time machine vision with voice alerts.<br/><br/> 

Doing fun things with machine vision. Includes utilities for all stages of pipeline development, from data collection to realtime inference.

Under rapid and active development. Will include:
- OpenCV utilities for capturing training data. 
- Tools for training pytorch models
- Utilities for realtime inference with verbal nudges

This is in pre-alpha stage, all of it (including the readme).

## To do
1. Get `capture.py` in slightly better shape
   - Add ability to change width/height instead of default width x height 
   - fix up the weird Qt warnings in linux
   - capture more data 
   - Make some simple tests 
2. Create pyproject.toml, etc and pypi it (opencv-python is currently only dependency)
3. Train on sleep data (use deepglue), build out `training.py` module. 
   - Lift and shift ImageLabeler and image copier to training.py from sandbox.
   - Train network using deepglue tools within sandbox.  
4. Once you have model trained, build out out `inference.py` module
