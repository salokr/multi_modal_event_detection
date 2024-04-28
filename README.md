# Multimodal-Disaster-Event-Identification-from-Social-Media-Posts

## Setting up: 
1) Download the dataset from here https://drive.google.com/file/d/1LIybWnY9JwYYhHDsXU2yMcHmdC52Q1el/view?usp=sharing
2) Install requirements `pip install -r requirements.txt`
3) Make sure the directory structure is like this:
heatmaps: initially empty will be used to store the attention heatmaps
Images: ignore
Code: contains notebook which you can ignore
Dataset: once you unzip this should contain the Dataset directory and the rest of the important files.

## Getting started:
We have implemented 3 models as of now:

### MultimodalClassifier

Run the code:
`python vanilla.py`

### Basic Attention 
The code is implemented using Bhadnau's attention and a similar attention module for images
Run the code:
`python basic_attention.py`

### Cross Modal Self-Attention 
Run the code:
`python cross_attention.py`
