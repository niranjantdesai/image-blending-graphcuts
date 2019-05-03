# Image blending with graphcuts

## Installation

The implementation is in Python 3.

Install the required Python packages using the following command:
```
pip install -r python_requirements.txt
```

`maxflow` requires Microsoft Visual C++ Compiler for Python 2.7 (https://www.microsoft.com/en-us/download/confirmation.aspx?id=44266)
Install PyMaxFlow in the end

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