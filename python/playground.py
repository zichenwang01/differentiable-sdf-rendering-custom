import os
from os.path import join

import numpy as np 
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

# get config
from configs import apply_cmdline_args, get_config
config = get_config('warp')

print("Loaded config")
print("config: ", config)
print("integrator: ", config.integrator)
print("spp: ", config.spp)
print()

# get opt config
from opt_configs import is_valid_opt_config, get_opt_config
opt_config = get_opt_config('no-tex-6-hqq')

print("Loaded opt_config")
print("opt_config: ", opt_config)
print("opt_config.resx: ", opt_config.resx)
print()

from constants import SCENE_DIR

# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/dummysdf.xml'
scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/dragon/dragon.xml'

# sdf_scene = mi.load_file(
#     scene_name,
#     shape_file='dummysdf.xml', # load BSDF placeholder
#     sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_128.vol'),
#     integrator=config.integrator, # load custom integrator
#     # integrator='path', # load path integrator
#     resx=512, resy=512
# )

sdf_scene = mi.load_file(
    scene_name,
    shape_file = 'dummysdf.xml',
    # sdf_filename = join(SCENE_DIR, 'sdfs', 'bunny_opt_768.vol'),
    # sdf_filename = '/home/zw336/IR/differentiable-sdf-rendering/scenes/sdfs/bunny_128.vol',
    integrator = 'sdf_direct_reparam'
)

params = mi.traverse(sdf_scene)

integrator = sdf_scene.integrator()

print("Loaded scene")
# print("sdf_scene: ", sdf_scene)
print("integrator: ", sdf_scene.integrator())
print("integrator name: ", sdf_scene.integrator().name)
# print("sdf filename: ", sdf_scene.sdf_filename)

# load camera
from util import get_regular_cameras
sensor = get_regular_cameras(n_sensors=1, resx=opt_config.resx, resy=opt_config.resy)[0]

# render mesh
image = mi.render(
    sdf_scene, params=params, sensor=sensor, seed=0, 
    spp=config.spp * config.primal_spp_mult, spp_grad=config.spp
)
mi.util.write_bitmap('primal.exr', image)

# render forward
image = integrator.render(scene=sdf_scene, sensor=sensor, spp=config.spp * config.primal_spp_mult, mode=dr.ADMode.Forward)
mi.util.write_bitmap('forward.exr', image)