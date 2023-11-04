------------------------------ UTIL ---------------------------------

constant.py
    This file stores the global constants, including output paths and scene directory.

mesh_to_sdf.py
    This file inputs a mesh and outputs a SDF.

redistancing.py
    This file inputs the zero level set and outputs a SDF.

math_util.py
    This file implements some basic utility functions.

util.py 
    This file implements some camera and image utility functions.

----------------------------- OPTIMIZE -------------------------------

config.py
    This file specifies the differentiable renderer configuration.

opt_configs.py
    This file specifies the optimization configuration, including the number of cameras, the loss function, and the time to upscale.

losses.py
    This file implements the loss functions.

regularizations.py
    This file implements the laplacian regularization functions.

optimize.py
    This file optimizes the geometry.
    This is the MAIN file.

----------------------------- POST PROCESS --------------------------------

create_video.py
    This file creates a video from the rendered images.

render_turnable.py
    This file renders a turntable video.
    This is not used in other files.

