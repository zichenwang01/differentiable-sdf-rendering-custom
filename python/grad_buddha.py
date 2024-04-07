import os
import sys
from os.path import join

import numpy as np 
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

# ------------------------------ CONFIG ------------------------------
from configs import apply_cmdline_args, get_config
config = get_config('warp')

print("Loaded config")
print("config: ", config)
print("integrator: ", config.integrator)
print("spp: ", config.spp)
print()

# ------------------------------ OPT CONFIG ------------------------------
from opt_configs import is_valid_opt_config, get_opt_config
opt_config = get_opt_config('no-tex-6-hqq')

print("Loaded opt_config")
print("opt_config: ", opt_config)
print("opt_config.resx: ", opt_config.resx)
print()

# ------------------------------ SCENE ------------------------------
# scene name
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/dragon/dragon.xml'
scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/custom_bottom.xml'
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/custom.xml'
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/custom_display.xml'

# shape name
mesh_name = 'dummysdf.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/lucy/lucy-shape.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/buddha/buddha-shape.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/logo_bottom/logo-shape.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/vbunny/vbunny-shape.xml'

# sdf name
from constants import SCENE_DIR
# sdf_filename = join(SCENE_DIR, 'sdfs', 'bunny_res=256.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'bunny_background_res=512.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'bunny_bottom_res=256.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'dragon_background_res=512.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'dragon_bottom_res=512.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'buddha_background_res=512.sdf')
# sdf_filename = join(SCENE_DIR, 'sdfs', 'buddha_bottom_res=512.sdf')
sdf_filename = join(SCENE_DIR, 'sdfs', 'buddha_bottom_rotated.sdf')
# sdf_filename = '/home/zw336/IR/bp/data/bottom_plane/vbunny_z_res=512.sdf'

# sensor name
# sensor_filename = join(SCENE_DIR, 'custom_background_sensors.xml')
sensor_filename = join(SCENE_DIR, 'custom_bottom_sensors.xml')

# emitter name
# emitter_filename = join(SCENE_DIR, 'emitters', 'cathedral.xml')
emitter_filename = join(SCENE_DIR, 'emitters', 'custom_top_area.xml')
# emitter_filename = join(SCENE_DIR, 'emitters', 'custom_tilt_front_area.xml')
# emitter_filename = join(SCENE_DIR, 'emitters', 'custom_point.xml')

# load scene
sdf_scene = mi.load_file(
    scene_name,
    shape_file = mesh_name,
    sdf_filename = sdf_filename,
    sensors_filename = sensor_filename,
    emitter_scene = emitter_filename,
    integrator = 'sdf_direct_reparam'
)

params = mi.traverse(sdf_scene)

integrator = sdf_scene.integrator()

# load camera
from util import get_regular_cameras, get_sensors
# sensor = get_regular_cameras(n_sensors=1, resx=opt_config.resx, resy=opt_config.resy)[0]
sensor = sdf_scene.sensors()[0]
# sensors = get_sensors(24, resx=256, resy=256)

# sensor = mi.load_dict({
#     'type': 'perspective', 'fov': 45,
#     'sampler': {'type': 'independent'},
#     'film': {
#         'type': 'hdrfilm',
#         'width': 1024, 'height': 1024,
#         'filter': {'type': 'gaussian'}
#     },
#     'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.5, 0.5], origin=[0.5, 0.55, 0.9], up=[0, 1, 0]),
# })

# print scene
print("Loaded scene")
# print("sdf_scene: ", sdf_scene)
# print("sdf filename: ", sdf_scene.sdf_filename)
print("integrator: ", sdf_scene.integrator())
print("integrator name: ", sdf_scene.integrator().name)
# print("sensors: ", sdf_scene.sensors())
# print(sensors[18])
# exit(0)

# ------------------------------ RENDER ------------------------------
image = mi.render(
    sdf_scene, params=params, sensor=sensor, seed=0, 
    spp=1024, spp_grad=config.spp
)
mi.util.write_bitmap('primal.exr', image)

bitmap = mi.Bitmap(image)
mi.util.write_bitmap('primal.png', bitmap)

# render forward
# image = integrator.render(scene=sdf_scene, sensor=sensor, spp=config.spp * config.primal_spp_mult, mode=dr.ADMode.Forward)
# mi.util.write_bitmap('forward.exr', image)

print("Rendered scene")
# exit(0)
# ------------------------------ FORWARD GRADIENTS -----------------------------
sys.path.append('/home/zw336/IR/differentiable-sdf-rendering/figures')

import configs
# method = configs.Warp()
# method = configs.ConvolutionWarp32()
# method = configs.ConvolutionWarp16()
method = configs.ConvolutionWarp8()

import time
from common import *
start = time.time()
dr.set_flag(dr.JitFlag.KernelHistory, True)
method.use_finite_differences = True
image, gradient, stats = eval_forward_gradient(scene=sdf_scene, config=method, axis='x', spp=1024)
end = time.time()
print("Forward gradient time: ", end - start)
print("Forward gradient stats: ", stats)

# mi.util.write_bitmap('vicini.exr', gradient)
# mi.util.write_bitmap('conv32.exr', gradient)
# mi.util.write_bitmap('conv16.exr', gradient)
# mi.util.write_bitmap('conv8.exr', gradient)
mi.util.write_bitmap('fd.exr', gradient)

bitmap = mi.Bitmap(gradient)
# mi.util.write_bitmap('vicini.png', bitmap)
# mi.util.write_bitmap('conv32.png', bitmap)
# mi.util.write_bitmap('conv16.png', bitmap)
# mi.util.write_bitmap('conv8.png', bitmap)
mi.util.write_bitmap('fd.png', gradient)