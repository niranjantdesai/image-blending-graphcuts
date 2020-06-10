# Image blending with graphcuts

The implementation follows the [paper](https://www.cc.gatech.edu/~turk/my_papers/graph_cuts.pdf) "Graphcut Textures: Image and Video Synthesis Using Graph Cuts" by Kwatra et al.

## Installation

The implementation is in Python 3.

Install the required Python packages using the following command:
```
pip install -r python_requirements.txt
```

The `networkx` library is used for graph plotting.

## Running
To run the UI for creating masks, use the following command:
```
python gui.py -i <path-to-image-directory>
```

Run graph cuts with the following command:
```
python graph_cuts.py -i <path-to-image-directory>
```

The image directory should have two images, `src.jpg` and `target.jpg`. It is recommended that you resize your images to a small resolution like 640 x 480 so that the blending doesn't take too long.

A few examples are included in the `images` directory.