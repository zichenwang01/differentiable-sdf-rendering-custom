import os
import numpy as np 
import drjit as dr
import mitsuba as mi

from os.path import join
from constants import SCENE_DIR

mi.set_variant('cuda_ad_rgb')

from configs import apply_cmdline_args, get_config
config = get_config('warp')

from opt_configs import is_valid_opt_config, get_opt_config
opt_config = get_opt_config('no-tex-12')

# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/bunny/bunny.xml'

# sdf_scene = mi.load_file(
#     scene_name, 
#     shape_file='dummysdf.xml', # load BSDF placeholder
#     sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_64.vol'),
#     integrator=config.integrator, # load custom integrator
#     resx=512, resy=512
# )

# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/empty/empty.xml'
scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/lego/lego.xml'

sdf_scene = mi.load_file(
    scene_name,
    shape_file='dummysdf.xml', # load BSDF placeholder
    sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_128.vol'),
    integrator=config.integrator, # load custom integrator
    # integrator='path', # load path integrator
    resx=512, resy=512
)

params = mi.traverse(sdf_scene)

from util import get_regular_cameras
sensor = get_regular_cameras(n_sensors=1, resx=512, resy=512)[0]

img = mi.render(sdf_scene, params=params, sensor=sensor,
                seed=0, spp=config.spp * config.primal_spp_mult,
                spp_grad=config.spp)
img = mi.util.convert_to_bitmap(img)
mi.util.write_bitmap('test.png', img)