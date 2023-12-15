"""Figure 7: Comparison of different ways of nesting reparameterizations"""
import os
import sys
import tqdm

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from common import *
from constants import SCENE_DIR
import configs

def render_nested_reparam_figure(force=0):
    # output directory
    output_dir = join(FIGURE_DIR, 'nested_reparam')
    os.makedirs(output_dir, exist_ok=True)
    
    # render parameters
    resx, resy = 128, 128
    spp, fd_spp = 1024, 1024

    # scene filename
    ref_scene_name = join(SCENE_DIR, 'figures', 'nested_reparam', 'nested_reparam.xml')
 
    # sdf filename
    sdf_fn = join(SCENE_DIR, 'sdfs', 'shadowing_spheres_256.vol')

    # sensor indices
    sensors = [0, 1]

    # configs
    used_configs = [
        configs.Warp(), 
        configs.Warp(),
        configs.Warp(), 
        configs.FiniteDifferences()
    ]
    
    technique_names = [
        "correct", 
        "decouple_reparam", 
        "detach_indirect_si", 
        "fd"
    ]
    
    # flag to set for each config
    scene_attrs = [
        {}, 
        {'decouple_reparam': True}, 
        {'detach_indirect_si': True}, 
        {}
    ]
    
    pbar = tqdm.tqdm(range(len(sensors) * len(used_configs)))
    for sensor in sensors:
        for config, name, scene_attr in zip(used_configs, technique_names, scene_attrs):
            # image/gradient filenames
            img_fn = join(output_dir, f'{sensor}_{name}.exr')
            grad_fn = join(output_dir, f'grad_{sensor}_{name}.exr')
            
            # skip if file exists
            if os.path.isfile(grad_fn) and os.path.isfile(img_fn) and not force > 0:
                print("Skipping, file exists: ", f'grad_{sensor}_{name}.exr')
                pbar.update()
                continue
            
            # load scene
            scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, sdf_filename=sdf_fn, **scene_attr)
            
            # forward gradients
            img, grad, _ = eval_forward_gradient(scene, config, 'z', sensor=sensor, spp=spp, fd_spp=fd_spp)
            
            # save images
            mi.util.write_bitmap(grad_fn, grad)
            mi.util.write_bitmap(img_fn, img)
            pbar.update()

    mi.Thread.wait_for_tasks()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    render_nested_reparam_figure(force=args.force)
