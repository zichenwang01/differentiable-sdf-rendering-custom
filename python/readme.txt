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

----------------------------- SCENE XML --------------------------------

common.xml
    This file specifies the common parameters for all scenes.
    This includes intergrator_sdf.xml and sensors.xml
    This is included in all scenes

intergrator_sdf.xml
    This file is an integrator interface

sensors.xml
    This file specifies the cameras

----------------------------- OPTIMIZE -------------------------------

config.py
    This file specifies the differentiable renderer configuration.

opt_configs.py
    This file specifies the optimization configuration, including the number of cameras, the loss function, and the time to upscale.

losses.py
    This file implements the loss functions.

regularizations.py
    This file implements the laplacian regularization functions.

shapes.py 

variables.py
    This file implements the Variable class.
    A Variable is a set of parameters of a certain property to optimize.

optimize.py
    This is the MAIN entrance point.
    This renders the reference images.

shape_opt.py 


----------------------------- POST PROCESS --------------------------------

create_video.py
    This file creates a video from the rendered images.

render_turnable.py
    This file renders a turntable video.
    This is not used in other files.

