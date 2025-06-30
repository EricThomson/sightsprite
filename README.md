# sightsprite 
<img src="https://raw.githubusercontent.com/EricThomson/sightsprite/main/src/sightsprite/data/sightsprite_logo.jpg" alt="sightsprite logo" align="right" width="250">
Real-time machine vision with voice alerts.<br/><br/> 

Fun with machine vision. Utilities for all stages of pipeline development for machine vision with voice alerts: from data collection to real-time inference.

Under rapid and active development, breaking changes are guaranteed through summer of 2025. Am currently building:
- OpenCV utilities for capturing training data. 
- Simple tools (in matplotlib) for labeling and organizing captured data.
- Tools for training pytorch models.
- Utilities for realtime inference with voice alerts (e.g., "get off the couch").

Everything is changing very fast. Will update when things are less unstable. 

## To do
Once following steps working I'll consider sightsprite to be in alpha stage:
1. Get `capture.py` in slightly better shape
   - Add ability to change width/height instead of default width x height 
   - fix up the weird Qt warnings in linux
   - Make some simple tests 
2. Train on sleep data (use deepglue), build out `training.py` module. 
   - Label sleep data using ImageLabeler
   - Train network using deepglue tools within sandbox.  
4. Once you have model trained, build out out `inference.py` 
   -  Real-time inference
   -  Voice alerts to nudge you.
5. Build some docs to explain how it all works. 


### Notes etc
Any images skipped during labeling will appear first when you restart the `ImageLabeler`
   
