import os
import time
import tqdm
from os.path import join

import numpy as np
import drjit as dr
import mitsuba as mi

from constants import IS_DEBUG
from constants import SCENE_DIR
from create_video import create_video
from util import dump_metadata, render_turntable, resize_img, set_sensor_res

def load_ref_images(paths, multiscale=False):
    """Load the reference images and compute scale pyramid for multiscale loss"""
    if not multiscale:
        return [mi.TensorXf(mi.Bitmap(fn)) for fn in paths]
    
    result = []
    for fn in paths:
        bmp = mi.Bitmap(fn)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = mi.TensorXf(resize_img(bmp, new_res, smooth=True))
        result.append(d)
    return result


def optimize_shape_timed(scene_config, mts_args, ref_image_paths,
                   output_dir, config, write_ldr_images=True):
    """Main function that optmizes the geometry"""

    if IS_DEBUG:
        print("Enter geometry optimization")
 
    # Print out the command line arguments passed to Mitsuba
    if len(mts_args) > 0:
        print(f"Cmdline arguments passed to Mitsuba: {mts_args}")

    # Create output directories
    opt_image_dir = join(output_dir, 'opt')
    os.makedirs(opt_image_dir, exist_ok=True)
    
    grad_dir = join(output_dir, 'grad')
    os.makedirs(grad_dir, exist_ok=True)

    # Load the reference images
    scene_name = scene_config.scene
    ref_scene_name = join(SCENE_DIR, scene_name, f'{scene_name}.xml')
    ref_images = load_ref_images(ref_image_paths, True)
    
    if IS_DEBUG:
        print("Loaded reference images")
        print("ref_scene_name: ", ref_scene_name)
        print("scene_name: ", scene_name)
    
    # Load scene, currently handle SDF shape separately from Mitsuba scene
    sdf_scene = mi.load_file(
        ref_scene_name, 
        shape_file='dummysdf.xml', # load initial sphere
        # shape_file='dummysdf_bottom.xml', # load initial sphere
        # sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_opt_255.vol'),
        # sdf_filename=join(SCENE_DIR, 'sdfs', 'sphere_res=256.sdf'),
        sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_res=256.sdf'),
        integrator=config.integrator, # load custom integrator
        resx=scene_config.resx, resy=scene_config.resy, 
        **mts_args
    )
    
    if IS_DEBUG:
        print("Loaded scene")
        print("integrator", sdf_scene.integrator())
        print("integrator name", sdf_scene.integrator().name)
        print("sdf: ", sdf_scene.integrator().sdf)
        print("sdf shape: ", sdf_scene.integrator().sdf_shape)
    
    sdf_object = sdf_scene.integrator().sdf
    sdf_scene.integrator().warp_field = config.get_warpfield(sdf_object)
    
    # print("sdf object", sdf_object)

    # Check SDF placeholder
    params = mi.traverse(sdf_scene)
    assert any('_sdf_' in shape.id() for shape in sdf_scene.shapes()), \
           "Could not find a placeholder shape for the SDF"
    params.keep(scene_config.param_keys)

    # print("start optimization")

    # Render shape initialization
    with dr.suspend_grad():
        for idx, sensor in enumerate(scene_config.sensors):
            img = mi.render(sdf_scene, sensor=sensor, seed=idx,
                            spp=config.spp * config.primal_spp_mult)
            bitmap = mi.Bitmap(img)
            mi.util.write_bitmap(join(output_dir, f'init-{idx:02d}.exr'), img[..., :3])
            mi.util.write_bitmap(join(output_dir, f'init-{idx:02d}.png'), bitmap)

    # Initialize sensor resolutions
    for sensor in scene_config.sensors:
        set_sensor_res(sensor, scene_config.init_res)
    
    # Initialize optimizer
    opt = mi.ad.Adam(lr=config.learning_rate, params=params, mask_updates=config.mask_optimizer)
    n_iter = config.n_iter
    scene_config.initialize(opt, sdf_scene)
    params.update(opt)
    
    # Run optimization
    seed = 0
    loss_values = []
    try:
        pbar = tqdm.tqdm(range(n_iter))
        for i in pbar:
            loss = mi.Float(0.0)
            for idx, sensor in scene_config.get_sensor_iterator(i):
                # Render image
                start = time.time()
                img = mi.render(
                    sdf_scene, params=params, sensor=sensor, seed=seed, 
                    spp=config.spp * config.primal_spp_mult,
                    seed_grad=seed + 1 + len(scene_config.sensors), 
                    spp_grad=config.spp
                )
                seed += 1 + len(scene_config.sensors)
                end = time.time()
                print("render time: ", end - start)
                
                # Compute image loss
                # start = time.time()
                loss_func = scene_config.loss
                if loss_func.__name__ == 'l2_xixi':
                    img2 = mi.render(
                        sdf_scene, params=params, sensor=sensor, seed=seed, 
                        spp=config.spp * config.primal_spp_mult,
                        seed_grad=seed + 1 + len(scene_config.sensors), 
                        spp_grad=config.spp
                    )
                    # view_loss = loss_func(img, img2, ref_images[idx][sensor.film().crop_size()[0]]) / scene_config.batch_size
                    view_loss = loss_func(img, img2, ref_images[idx][sensor.film().crop_size()[0]])
                else:
                    # print(sensor.film().crop_size()[0])
                    # view_loss = scene_config.loss(img, ref_images[idx][sensor.film().crop_size()[0]]) / scene_config.batch_size
                    view_loss = scene_config.loss(img, ref_images[idx][sensor.film().crop_size()[0]])
                
                # Backpropagate loss
                start = time.time()
                dr.backward(view_loss)
                end = time.time()
                print("loss time: ", end - start)
                
                # Save rendered image
                # start = time.time()
                # bmp = resize_img(mi.Bitmap(img), scene_config.target_res)
                # mi.util.write_bitmap(join(opt_image_dir, f'opt-{i:04d}-{idx:02d}' + ('.png' if write_ldr_images else '.exr')), bmp)
                # end = time.time()
                # print("write image time: ", end - start)
                
                # accumulate loss
                loss += view_loss

            # Compute regularization loss
            start = time.time()
            reg_loss = scene_config.eval_regularizer(opt, sdf_object, i)
            if dr.grad_enabled(reg_loss):
                dr.backward(reg_loss)
            loss += dr.detach(reg_loss)
            end = time.time()
            print("regularization time: ", end - start)

            # Save checkpoint SDF
            scene_config.save_params(opt, output_dir, i, force=i == n_iter - 1)
            
            # Check gradients
            start = time.time()
            scene_config.validate_gradients(opt, i)
            end = time.time()
            print("gradient check time: ", end - start)
            
            # Print loss to progress bar
            img_loss_mean = (loss - reg_loss) / scene_config.batch_size
            pbar.set_description(f"img loss {1e3*img_loss_mean[0]:.2f}, reg loss {1e3*reg_loss[0]:.2f}")
            
            # loss_str = f'Loss: {1000*loss[0]:.4f}'
            # if dr.grad_enabled(reg_loss):
            #     loss_str += f' (reg (avg. x 1e4): {1e3*reg_loss[0] / dr.prod(sdf_object.shape):.4f})'
            # pbar.set_description(loss_str)
            
            # Save loss
            loss_values.append(loss[0])
            
            # Update parameters
            start = time.time()
            opt.step()
            end = time.time()
            print("update time: ", end - start)
            scene_config.validate_params(opt, i)
            scene_config.update_scene(sdf_scene, i)
            params.update(opt)
            
            if i in scene_config.upsample_iter:
                # upsample spp
                config.spp *= 2
                print(f"Upsampling to {config.spp} spp")
            
    finally:
        # Plot loss figure
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(loss_values)), loss_values)
        plt.xlabel('Iterations')
        plt.ylabel('Objective function value')
        avg_loss = np.mean(np.array(loss_values)[-5:])
        plt.title(f"Final loss: {1000*loss_values[-1]:.3f} (avg. over 5 its: {1000*avg_loss:.3f})")
        plt.savefig(join(output_dir, 'loss.pdf'))
        plt.savefig(join(output_dir, 'loss.png'))
        np.save(join(output_dir, "loss.npy"), loss_values)

        # Write out total time and basic config info to json
        d = {'total_time': time.time() - pbar.start_t, 'loss_values': loss_values}
        dump_metadata(config, scene_config, d, join(output_dir, 'metadata.json'))

    # Save optimization video and turntable video
    print("[+] Writing convergence video")
    create_video(output_dir)
    
    # Load the exponential moving average of the parameters and save them
    if scene_config.param_averaging_beta is not None:
        scene_config.load_mean_parameters(opt)
        scene_config.save_params(opt, output_dir, 'final')
        params.update(opt)
        
    # Save turntable images
    print("[+] Rendering turntable")
    sdf_scene.integrator().warp_field = None
    render_turntable(sdf_scene, output_dir, resx=512, resy=512, spp=256, n_frames=64)
