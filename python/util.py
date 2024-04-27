import os
import tqdm
import subprocess
from os.path import join

import drjit as dr
import mitsuba as mi

# Initialize the image filters used to resample images
GAUSSIAN_RFILTER = mi.scalar_rgb.load_dict({'type': 'gaussian'})
BOX_RFILTER = mi.scalar_rgb.load_dict({'type': 'box'})

def resize_img(img, target_res, smooth=False):
    """Resizes a Mitsuba Bitmap using either a box filter (smooth=False)
       or a gaussian filter (smooth=True)"""
    assert isinstance(img, mi.Bitmap)
    source_res = img.size()
    if target_res[0] == source_res[0] and target_res[1] == source_res[1]:
        return img
    rfilter = GAUSSIAN_RFILTER if smooth else BOX_RFILTER
    return img.resample([target_res[1], target_res[0]], rfilter)


def render_turntable(
    scene, output_dir, resx=128, resy=128, 
    spp=64, n_frames=64, write_exr_files=False
):
    """Renders a given scene as a turntable video"""

    # output directory
    frame_output_dir = os.path.join(output_dir, 'turntable')
    os.makedirs(frame_output_dir, exist_ok=True)
    
    for frame in tqdm.tqdm(range(n_frames)):
        # camera angle
        angle = frame / n_frames * 2 * dr.pi
        
        # camera position
        radius = 1.5
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.8, dr.sin(angle) * radius + 0.5)
        
        # load sensor
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 39.0,
            'sampler': {'type': 'independent'},
            'film': {'type': 'hdrfilm', 'width': resx, 'height': resy, 'pixel_filter': {'type': 'gaussian'}}, 'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0])
        })

        # render
        with dr.suspend_grad():
            result = mi.render(scene, seed=frame, spp=spp, sensor=sensor)
            
        # save render images
        bmp = mi.Bitmap(result)
        fn = join(frame_output_dir, f'frame-{frame:04d}.exr')
        if write_exr_files:
            mi.util.write_bitmap(fn, bmp)
        mi.util.write_bitmap(fn.replace('.exr', '.png'), bmp)

    mi.Thread.wait_for_tasks()

    frame_name = join(frame_output_dir, 'frame-%04d.png')
    video_dir = os.path.join(output_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    video_path = join(video_dir, 'turntable.mp4')
    run_ffmpeg(frame_name, video_path)


def run_ffmpeg(frame_name, video_path):
    """Converts a sequence of frames to a video. Requires ffmpeg to be installed on the system."""
    import shutil
    if shutil.which('ffmpeg') is None:
        print("Cannot find ffmpeg, skipping video generation")
        return

    # Escape spaces in paths that are passed to ffmpeg. For now, only tested on Linux
    frame_name = frame_name.replace(' ', '\\ ')
    video_path = video_path.replace(' ', '\\ ')
    ffmpeg_cmd = f'ffmpeg -y -hide_banner -loglevel error -i {frame_name} -c:v libx264 -movflags +faststart -vf format=yuv420p -crf 15 -nostdin {video_path}'
    subprocess.call(ffmpeg_cmd, shell=True)


def get_file_sensors(fn, resx, resy, indices):
    """Retrieves a list of sensors from a Mitsuba scene file"""
    
    from constants import SCENE_DIR
    sensors = mi.load_file(
        os.path.join(SCENE_DIR, fn),
        integrator='sdf_direct_reparam', resx=resx, resy=resy
    ).sensors()
    
    if indices is not None:
        return [sensors[i] for i in indices]
    else:
        return sensors


def get_regular_camera_positions(angle_steps, height_steps, hemisphere=True,
                                 vary_height=True, radius=2.0, angle_shift=0.0,
                                 height_scale=1.0):
    """Returns a list of camera positions that are regularly spaced on a sphere."""
    min_elevation = 0.1
    max_elevation = 0.9

    if height_steps > 1:
        n_sensors = height_steps * angle_steps
        n_angles = n_sensors // height_steps
        angles, elevation = dr.meshgrid(
            dr.linspace(mi.Float, 0, 1 - 1 / n_angles, n_angles) * 2 * dr.pi,
            dr.linspace(mi.Float, 1 - max_elevation + 0.5 / height_steps, max_elevation - min_elevation, height_steps) * dr.pi
        )
        if hemisphere:
            elevation = elevation / 2
    else:
        n_sensors = angle_steps
        angles = (dr.linspace(mi.Float, 0, 1, n_sensors, endpoint=False) + angle_shift / n_sensors) * 2 * dr.pi
        if vary_height:
            # Fixed elevation and potentially scale it to shift it upwards
            elevation = mi.Float(1.15) / height_scale
            elevation += dr.sin(angles * angle_steps / 4) * 0.5
            if hemisphere:
                elevation = dr.clamp(elevation, 0.0, dr.pi / 2 + 0.05)
        else:
            elevation = mi.Float(1.5)

    origins = mi.Point3f(
        dr.cos(angles) * dr.sin(elevation) * radius, 
        dr.cos(elevation) * radius, 
        dr.sin(angles) * dr.sin(elevation) * radius
    )
    origins = origins + mi.Vector3f(0.5, 0.0, 0.5)
    return origins


def get_regular_cameras(n_sensors, angle_shift=0.0, resx=256, resy=256, radius=2.0, height_scale=1.0):
    """Generates regularly spaced sensors for optimization. Returns a list of Mitsuba sensors"""

    import numpy as np

    height_steps = int(n_sensors > 1)
    origins = get_regular_camera_positions(
        n_sensors, height_steps, hemisphere=True, vary_height=True,
        radius=radius, angle_shift=angle_shift, height_scale=height_scale
    )
    origins = np.array(origins)
    sensors = []
    sampler = mi.load_dict({'type': 'independent'})
    film = mi.load_dict({
        'type': 'hdrfilm', 'width': resx, 'height': resy,
        'pixel_format': 'rgb', 'pixel_filter': {'type': 'gaussian'}, 'sample_border': True})

    for o in origins:
        s = mi.load_dict({
            'type': 'perspective',
            'fov': 39.0,
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0]),
            'sampler': sampler,
            'film': film})
        sensors.append(s)
    return sensors


def get_regular_cameras_top(n_sensors, angle_shift=0.0, resx=128, resy=128, radius=2.0):
    """Generates regularly spaced sensors that primarily look down from a top view"""
    return get_regular_cameras(n_sensors, angle_shift, resx, resy, radius, height_scale=1.3)


def get_sensors(num_sensor=16, resx=256, resy=256):
    """Return a list of sensors arranged in a circle around the origin"""
    sensors = []
    for i in range(num_sensor):
        angle = 360.0 / num_sensor * i
        # angle = 0
        sensors.append(mi.load_dict({
            'type': 'perspective', 'fov': 45,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm',
                'width': resx, 'height': resy,
                'filter': {'type': 'gaussian'}
            },
            'to_world': mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]).rotate([0, 1, 0], angle).look_at(target=[0, 0, 0], origin=[0, 0, 0.5], up=[0, 1, 0]),
        }))
    return sensors

def get_sensors_big(num_sensor=24, resx=512, resy=512):
    """Return a list of sensors arranged in a circle around the origin"""
    sensors = []
    for i in range(num_sensor):
        angle = 360.0 / num_sensor * i
        # angle = 0
        sensors.append(mi.load_dict({
            'type': 'perspective', 'fov': 45,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm',
                'width': resx, 'height': resy,
                'filter': {'type': 'gaussian'}
            },
            'to_world': mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]).rotate([0, 1, 0], angle).look_at(target=[0, 0, 0], origin=[0, 0, 1.3], up=[0, 1, 0]),
        }))
    return sensors

def get_bottom_sensor(num_sensor=1, resx=1024, resy=1024):
    sensors = []
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.1, 0.5], origin=[0.5, 0.5, 1], up=[0, 1, 0]),
    }))
    return sensors

def get_bottom_sensors(num_sensor=4, resx=1024, resy=1024):
    sensors = []
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.1, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.9, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.1, 0.5, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.9, 0.5, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    return sensors

def get_suzanne_sensors(num_sensor=6, resx=1024, resy=1024):
    sensors = []
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.1, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.9, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.1, 0.5, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.9, 0.5, 0.5], origin=[0.5, 0.5, 1.0], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.5, 0.1], origin=[0.5, 0.1, 0.5], up=[0, 1, 0]),
    }))
    sensors.append(mi.load_dict({
        'type': 'perspective', 'fov': 45,
        'sampler': {'type': 'independent'},
        'film': {
            'type': 'hdrfilm',
            'width': resx, 'height': resy,
            'filter': {'type': 'gaussian'}
        },
        'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.5, 0.9], origin=[0.5, 0.1, 0.5], up=[0, 1, 0]),
    }))
    return sensors

def get_h2_sensors(
    num_sensor=25, num_sensor_theta=12, num_sensor_phi=4, 
    resx=512, resy=512, radius=1.3
):
    """Return a list of sensors arranged evenly on a hemisphere"""
    if num_sensor == 50:
        num_sensor_theta = 24
    
    import numpy as np
    sensors = []
    for j in range(num_sensor_phi):
        for i in range(num_sensor_theta // (j+1)):
            # Convert the index to spherical coordinates
            theta = np.pi * 2 * i / (num_sensor_theta // (j+1)) # azimuthal angle, from 0 to 2pi
            phi = np.pi / 3 * j / num_sensor_phi  # zenith angle, from 0 to pi/2
            # phi = np.arccos(1 - 2 * j / (num_sensor_phi - 1))  # zenith angle, from 0 to pi/2

            # Convert spherical coordinates to Cartesian coordinates
            z = np.cos(phi) * np.cos(theta) * radius + 0.5
            x = np.cos(phi) * np.sin(theta) * radius + 0.5
            y = np.sin(phi) * radius + 0.35

            sensors.append(mi.load_dict({
                'type': 'perspective', 'fov': 45,
                'sampler': {'type': 'independent'},
                'film': {
                    'type': 'hdrfilm',
                    'width': resx, 'height': resy,
                    'filter': {'type': 'gaussian'}
                },
                'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.5, 0.5], origin=[x, y, z], up=[0, 1, 0]),
            }))
    return sensors

def set_sensor_res(sensor, res):
    """Sets the resolution of an existing Mitsuba sensor"""
    params = mi.traverse(sensor)
    params['film.size'] = res
    params.update()

def dump_metadata(config, opt_config, extra=None, fn='test.json'):
    """Dumps out config and optimization config to a JSON file"""
    import inspect
    import json
    import sys

    import numpy as np

    def convert_to_string(obj):
        if hasattr(obj, 'name'):
            return obj.name
        if callable(obj) or inspect.isclass(obj):
            return obj.__name__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, mi.ScalarPoint2i):
            return [obj[0], obj[1]]
        return obj
    d = {}
    d['config'] = dict(config.__dict__)
    d['opt_config'] = dict(opt_config.__dict__)
    # For now, remove information which is difficult to serialize
    remove_keys = ['sensors', 'sensors_reordered', 'variables']
    for k in remove_keys:
        if k in d['opt_config']:
            d['opt_config'].pop(k)

    for k, v in d['opt_config'].items():
        d['opt_config'][k] = convert_to_string(v)
    d['cmd'] = ' '.join(sys.argv)

    if extra is not None:
        d.update(extra)
    with open(fn, 'wt') as f:
        json.dump(d, f, indent=4)


def optimization_result_exists(output_dir, config, opt_config, scene_name):
    """Checks if an optimization results already exists in the output directory"""

    if isinstance(opt_config, str):
        opt_name = opt_config
    else:
        opt_name = opt_config.name
    output_dir = os.path.join(output_dir, scene_name, opt_name, config.name)

    # return os.path.isfile(os.path.join(output_dir, 'metadata.json'))
    return os.path.isfile(os.path.join(output_dir, 'loss.png'))


def get_checkpoint_path_and_suffix(output_dir, scene_name, opt_name, config_name):
    """Retrieves the output path for a specific optimization and returns the
    suffix of the latest optimization state."""

    import glob
    p = join(output_dir, scene_name, opt_name, config_name)
    p = os.path.realpath(p)
    params_dir = join(p, 'params')
    sdf_filename = sorted(glob.glob(join(params_dir, '*sdf*.vol')))[-1]
    suffix = os.path.splitext(os.path.split(sdf_filename)[-1])[0].split('-')[-1]
    try:
        suffix = int(suffix)
    except ValueError:
        pass
    return p, suffix


def atleast_4d(tensor):
    """Ensures a sensor has at least 4 dimensions"""
    if tensor.ndim == 3:
        return tensor[..., None]
    return tensor
