import os
import sys
from os.path import join

import numpy as np 
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

# ------------------------------ UTIL ------------------------------
from constants import SCENE_DIR

def turntable_env_fixed(
    scene:mi.Scene, folder:str, 
    num_frame=128, resx=1024, resy=1024, spp=256
):    
    scene.integrator().hide_emitters = False
    
    # create folder
    video_folder = join("exp", folder, "final")
    frame_folder = join("exp", folder, "final", "frames")
    os.makedirs(frame_folder, exist_ok=True)
    
    for frame in range(num_frame):
        print(f"Rendering frame {frame} / {num_frame}")
        # if already render then continue 
        if os.path.exists(join(frame_folder, f'frame{frame}.png')):
            continue
        
        # camera parameters
        radius = 0.5
        angle = frame / num_frame * 2 * dr.pi
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.5, dr.sin(angle) * radius + 0.5)
        # o = mi.ScalarPoint3f(0.5, dr.cos(angle) * radius + 0.5, dr.sin(angle) * radius + 0.5) # logo
        
        scene = mi.load_file(
            "/home/zw336/IR/differentiable-sdf-rendering/scenes/custom_turntable.xml",
            shape_file = mesh_name,
            # sdf_filename = sdf_filename,
            # sensors_filename = sensor_filename,
            # emitter_scene = emitter_filename,
            integrator = 'sdf_direct_reparam',
            angle = - angle / dr.pi * 180
        )
        scene.integrator().sdf = sdf
        # scene.angle = angle / dr.pi * 180
        
        # load camera
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 45.0,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm', 
                'width': resx, 'height': resy, 
                'pixel_filter': {'type': 'gaussian'}
            }, 
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0])
        })   
        
        # render
        with dr.suspend_grad():
            image = mi.render(scene=scene, sensor=sensor, spp=spp)
            bitmap = mi.util.convert_to_bitmap(image)
            
        # save image
        # mi.util.write_bitmap(join(frame_folder, f'frame{frame}.exr'), image)
        mi.util.write_bitmap(join(frame_folder, f'frame{frame}.png'), bitmap)
    
    # convert frames to video
    frame_name = join(frame_folder, 'frame%d.png')
    video_name = join(video_folder, 'turntable.mp4')
    run_ffmpeg(frame_name, video_name)

def turntable_env_rotated(
    scene:mi.Scene, folder:str, 
    num_frame=64, resx=1024, resy=1024, spp=128
):    
    scene.integrator().hide_emitter = False
    
    # create folder
    video_folder = join("exp", folder, "final")
    frame_folder = join("exp", folder, "final", "frames")
    os.makedirs(frame_folder, exist_ok=True)
    
    for frame in range(num_frame):
        print(f"Rendering frame {frame} / {num_frame}")
        # if already render then continue 
        if os.path.exists(join(frame_folder, f'frame{frame}.png')):
            continue
        
        # camera parameters
        radius = 0.5
        angle = frame / num_frame * 2 * dr.pi
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.5, dr.sin(angle) * radius + 0.49)
        
        # load camera
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 45.0,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm', 
                'width': resx, 'height': resy, 
                'pixel_filter': {'type': 'gaussian'}
            }, 
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.49], [0, 1, 0])
        })   
        
        # render
        with dr.suspend_grad():
            image = mi.render(scene=scene, sensor=sensor, spp=spp)
            bitmap = mi.util.convert_to_bitmap(image)
            
        # save image
        # mi.util.write_bitmap(join(frame_folder, f'frame{frame}.exr'), image)
        mi.util.write_bitmap(join(frame_folder, f'frame{frame}.png'), bitmap)
    
    # convert frames to video
    frame_name = join(frame_folder, 'frame%d.png')
    video_name = join(video_folder, 'turntable.mp4')
    run_ffmpeg(frame_name, video_name)

def turntable_env_black(
    scene:mi.Scene, folder:str, 
    num_frame=64, resx=1024, resy=1024, spp=128
):        
    # create folder
    video_folder = join("exp", folder, "final")
    frame_folder = join("exp", folder, "final", "frames")
    os.makedirs(frame_folder, exist_ok=True)
    
    for frame in range(num_frame):
        print(f"Rendering frame {frame} / {num_frame}")
        # if already render then continue 
        if os.path.exists(join(frame_folder, f'frame{frame}.png')):
            continue
        
        # camera parameters
        radius = 0.5
        angle = frame / num_frame * 2 * dr.pi
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.5, dr.sin(angle) * radius + 0.5)
        
        # load camera
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 45.0,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm', 
                'width': resx, 'height': resy, 
                'pixel_filter': {'type': 'gaussian'}
            }, 
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0])
        })   
        
        # render
        with dr.suspend_grad():
            image = mi.render(scene=scene, sensor=sensor, spp=spp)
            bitmap = mi.util.convert_to_bitmap(image)
            
        # save image
        # mi.util.write_bitmap(join(frame_folder, f'frame{frame}.exr'), image)
        mi.util.write_bitmap(join(frame_folder, f'frame{frame}.png'), bitmap)
    
    # convert frames to video
    frame_name = join(frame_folder, 'frame%d.png')
    video_name = join(video_folder, 'turntable.mp4')
    run_ffmpeg(frame_name, video_name)

def run_ffmpeg(frame_name, video_path):
    """ Converts a sequence of frames to a video. Requires ffmpeg to be installed on the system. Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """
    import shutil
    if shutil.which('ffmpeg') is None:
        print("Cannot find ffmpeg, skipping video generation")
        return

    # Escape spaces in paths that are passed to ffmpeg. For now, only tested on Linux
    frame_name = frame_name.replace(' ', '\\ ')
    video_path = video_path.replace(' ', '\\ ')
    ffmpeg_cmd = f'ffmpeg -y -hide_banner -loglevel error -i {frame_name} -c:v libx264 -movflags +faststart -vf format=yuv420p -crf 15 -nostdin {video_path}'
    
    import subprocess
    subprocess.call(ffmpeg_cmd, shell=True)

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
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/custom_turntable.xml'
scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/vbunny/vbunny.xml'
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/buddha/buddha.xml'
# scene_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/lucy/lucy.xml'

# shape name
mesh_name = 'dummysdf.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/vbunny/vbunny-shape.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/lucy/lucy-shape.xml'
# mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/buddha/buddha-shape.xml'
mesh_name = '/home/zw336/IR/differentiable-sdf-rendering/scenes/kangaroo/kangaroo-shape.xml'


# sdf name
# sdf_filename = '/home/zw336/IR/bp/exp/10.sdf_eps=1e-4/opt_dragon_res=64_lr=0.001_reg=1e-05_sdf_eps=0.0001/sdf/final.sdf'
# sdf_filename = '/home/zw336/IR/bp/exp/11.bottom/opt_logo_bottom_res=64_lr=0.001_s=0.001_L=100000_num_sensor=4_2/sdf/final.sdf'
# sdf_filename = '/home/zw336/IR/bp/exp/opt_buddha_high_res=64_lr=0.001_reg_weight=1e-05_view=24/sdf/final.sdf'
# sdf_filename = '/home/zw336/IR/bp/exp/10.sdf_eps=1e-4/lucy_refine_5/sdf/final.sdf'
# sdf_filename = '/home/zw336/IR/differentiable-sdf-rendering/outputs/lucy/exp_lucy_batch (freezed)/warp/params/512_499.sdf'
# sdf_filename = '/home/zw336/IR/bp/exp/opt_buddha_high_res=64_lr=0.001_reg_weight=1e-05_view=24_studio/sdf/final.sdf'
# sdf_filename = "/home/zw336/IR/differentiable-sdf-rendering/outputs/vbunny/exp_vbunny_new (freezed)/warp/params/512_499.sdf"
# sdf_filename = "/home/zw336/IR/differentiable-sdf-rendering/outputs/buddha/exp_buddha_batch24 (freezed)/warp/params/512_499.sdf"
sdf_filename = "/home/zw336/IR/bp/exp/15. eps/opt_kangaroo_res=64_lr=0.001_view=24_epoch=1000_resx=1024_eps=0.0001/sdf/final.sdf"

# sensor name
# sensor_filename = join(SCENE_DIR, 'custom_background_sensors.xml')
# sensor_filename = join(SCENE_DIR, 'custom_bottom_sensors.xml')

# emitter name
emitter_filename = join(SCENE_DIR, 'emitters', 'cathedral.xml')
# emitter_filename = join(SCENE_DIR, 'emitters', 'custom_top_area.xml')
# emitter_filename = join(SCENE_DIR, 'emitters', 'custom_tilt_front_area.xml')
# emitter_filename = join(SCENE_DIR, 'emitters', 'custom_point.xml')

# load scene
sdf_scene = mi.load_file(
    scene_name,
    shape_file = mesh_name,
    # sdf_filename = sdf_filename,
    # sensors_filename = sensor_filename,
    # emitter_scene = emitter_filename,
    integrator = 'sdf_direct_reparam'
)
sdf = sdf_scene.integrator().sdf

# turntable_env_fixed(sdf_scene, "buddha_mesh")
# turntable_env_black(sdf_scene, "vbunny_env_black")
turntable_env_fixed(sdf_scene, "kangaroo_mesh")
# turntable_env_rotated(sdf_scene, "lucy_env_rotated")
# turntable_env_fixed(sdf_scene, "vbunny_mesh_slow")