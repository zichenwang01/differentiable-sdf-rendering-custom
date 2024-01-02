import gc

import drjit as dr
import mitsuba as mi
import numpy as np
from shapes import Grid3d
from warp import DummyWarpField


class ReparamIntegrator(mi.SamplingIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.name = props.get('name', 'sdf-reparam')
        self.max_depth = props.get('max_depth', 4)

        self.use_mis = props.get('use_mis', False)
        self.force_optix = props.get('force_optix', False)
        self.weight_by_spp = props.get('weight_by_spp', False)
        self.antithetic_sampling = props.get('antithetic_sampling', False)
        assert not self.weight_by_spp, "Not supported"

        self.warp_field = None
        self.sdf_shape = None
        self.is_prepared = False
        self.use_optix = True

        # sdf transform
        sdf_transform = None
        if props.has_property('sdf_to_world'):
            sdf_transform = props.get('sdf_to_world', mi.ScalarTransform4f())
            sdf_transform = mi.ScalarTransform4f(sdf_transform.matrix)

        # load sdf
        sdf_filename = props.get('sdf_filename', '')
        if sdf_filename != '':
            self.sdf = Grid3d(sdf_filename, transform=sdf_transform)
        else:
            self.sdf = None

    def prepare(self, sensor, seed, spp, aovs=[]):
        """prepare film and sampler"""
        film = sensor.film()
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()        
        film.prepare(aovs)
        
        wavefront_size = dr.prod(film_size) * spp
        if dr.is_llvm_v(mi.Float):
            wavefront_size_limit = 0xffffffff
        else:
            wavefront_size_limit = 0x40000000      
        if wavefront_size > wavefront_size_limit:
            raise Exception(f"Wavefront {wavefront_size} exceeds {wavefront_size_limit}")
        
        sampler = sensor.sampler().clone()
        if spp != 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)
        
        return sampler, spp


    def prepare_sdf(self, scene):
        """prepare SDF"""
        
        if self.is_prepared:
            # print("already prepared")
            return

        if self.sdf is None:
            self.is_prepared = True
            return

        # Enable optix if we detect that there is more than just one shape
        self.use_optix = self.force_optix or len(scene.shapes()) > 1
        # extract the dummy shape that holds the SDFs BSDF
        for idx, s in enumerate(scene.shapes()):
            # print(s.id())
            # print(s)
            if '_sdf_' in s.id():
                self.sdf_shape = s
                shape_idx = idx
                break
            
        # check dummy sdf
        if self.sdf_shape is None:
            raise ValueError("Scene is missing a dummy SDF shape that holds the BSDF")

        self.sdf_shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx)
        # print("sdf shape", self.sdf_shape)
        dr.eval(self.sdf_shape)
        self.is_prepared = True

    def eval_sample(
        self, mode, scene, sensor, sampler, block, _aovs, 
        position_scaled, spp, active=mi.Bool(True)
    ):
        # Aperture sample for depth of field
        aperture_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d(active)
            
        # Shutter open time
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d(active) * sensor.shutter_open_time()

        # Wavelength sample of continuous spectrum
        wavelength_sample = sampler.next_1d(active)
        
        # sample a batch of rays
        ray, ray_weight = sensor.sample_ray_differential(time, wavelength_sample, position_scaled, aperture_sample)
        
        # scale the ray differentials
        diff_scale_factor = dr.rsqrt(mi.ScalarFloat(spp))
        ray.scale_differential(diff_scale_factor)
        
        # evaluate ray contribution
        rgb, valid_ray, det, aovs_ = self.sample(mode, scene, sampler, ray,
                                                 None, None, None, active)

        # Re-evaluate sample's sensor position and sensor importance
        it = dr.zeros(mi.Interaction3f)
        it.p = ray.o + ray.d
        ds, ray_weight = sensor.sample_direction(it, aperture_sample)
        ray_weight = dr.select(ray_weight > 0.0, ray_weight / dr.detach(ray_weight), 1.0)
        ray_weight = dr.replace_grad(type(ray_weight)(1.0), ray_weight)
        position_sample = ds.uv
        rgb = ray_weight * rgb

        # put RGB values in the block
        has_alpha_channel = block.channel_count() == 5 + len(aovs_)
        aovs = [None] * 3
        aovs[0] = rgb.x
        aovs[1] = rgb.y
        aovs[2] = rgb.z
        if has_alpha_channel:
            aovs.append(dr.select(valid_ray, mi.Float(1.0), mi.Float(0.0)))
        aovs.append(dr.replace_grad(mi.Float(1.0), det * ray_weight[0]))

        aovs = aovs + aovs_
        block.put(position_sample, aovs, active)

    def render(
        self, scene, sensor=0, seed=0,
        spp=0, develop=True, evaluate=True,
        mode=dr.ADMode.Primal
    ):
        if not develop:
            raise Exception("Must use develop=True for this AD integrator")

        # load sensor
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # prepare film and sampler
        sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names())

        # prepare SDF
        self.prepare_sdf(scene)

        # print("finish preparing")

        # get film parameters
        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()
        if film.sample_border():
            film_size += 2 * border_size        
        spp = sampler.sample_count()
        
        # Compute the pixel index
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)
        # Avoid division if spp is a power of 2
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)
            
        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)
        if film.sample_border():
            pos -= border_size
        pos += mi.Vector2i(film.crop_offset())

        # Compute the random offset within the pixel
        offset = sampler.next_2d()

        # Compute the sample position on the image plane
        position_sample = pos + offset
        if self.antithetic_sampling:
            position_sample2 = pos - offset + 1.0
            sampler2 = sampler.clone()

        # Rescale the sample position in [0, 1]^2
        crop_offset = sensor.film().crop_offset()
        crop_size = mi.Vector2f(sensor.film().crop_size())
        position_scaled = (position_sample - crop_offset) / crop_size
        if self.antithetic_sampling:
            position_scaled2 = (position_sample2 - crop_offset) / crop_size

        # initialize film block
        block = sensor.film().create_block()
        _aovs = [None] * len(self.aov_names())
        
        # If we ask for AOVs, reparameterize also in forward pass to get values
        if self.warp_field is not None and self.warp_field.return_aovs:
            mode = dr.ADMode.Forward

        # print("eval_sample")

        # sample and evaluate light path 
        # this calls the main sample function
        active = mi.Bool(True)
        self.eval_sample(mode, scene, sensor, sampler, block, _aovs, 
                         position_scaled, spp, active)
        
        # Antithetic sample: mirrored
        if self.antithetic_sampling:
            self.eval_sample(mode, scene, sensor, sampler2, block, _aovs, 
                             position_scaled2, spp, active)

        # collect garbage
        gc.collect()

        # Develop the film given the block
        sensor.film().put_block(block)
        primal_image = sensor.film().develop()
        return primal_image

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):
        image = self.render(scene=scene, sensor=sensor, seed=seed,
                            spp=spp, develop=True, evaluate=False, mode=dr.ADMode.Backward)
        dr.backward_from(image * grad_in)

    def render_forward(self, scene, params, sensor=0, seed=0, spp=0):
        image = self.render(scene=scene, sensor=sensor, seed=seed, spp=spp,
                            develop=True, evaluate=False, mode=dr.ADMode.Forward)
        dr.forward_to(image)
        return dr.grad(image)

    def ray_test(self, scene, sampler, ray, depth=0, reparam=True, active=True):
        return self.ray_intersect(scene, sampler, ray, depth=depth, reparam=reparam, ray_test=True, active=active)

    def ray_intersect(self, scene, sampler, ray, depth=0, ray_test=False, reparam=True, active=True):
        """Intersects both SDFs and other scene objects if necessary"""

        si = dr.zeros(mi.SurfaceInteraction3f)
        si_d = dr.zeros(mi.SurfaceInteraction3f)
        div = mi.Float(1.0)
        extra_output = {}
        its_found = mi.Mask(False)
        if self.use_optix:
            if ray_test:
                its_found |= scene.ray_test(ray, active)
            else:
                si = scene.ray_intersect(ray, active)
                si_d = dr.detach(si)

        if self.sdf is not None:
            wf = self.warp_field if self.warp_field is not None else DummyWarpField(self.sdf)
            if ray_test:
                its_found2, div, extra_output = wf.ray_intersect(
                    self.sdf_shape, sampler, ray, depth=depth, ray_test=ray_test, reparam=reparam, active=active)
                its_found |= its_found2
            else:
                si2, si_d2, div, extra_output = wf.ray_intersect(
                    self.sdf_shape, sampler, ray, depth=depth, ray_test=ray_test, reparam=reparam, active=active)
                valid = (not self.use_optix) | (si2.t < si.t)
                si[valid] = si2
                si_d[valid] = si_d2

        if ray_test:
            return its_found, div, extra_output
        else:
            return si, si_d, div, extra_output

    def ray_intersect_preliminary(self, scene, ray, active=True):
        """Computes a preliminary shape intersection relatively efficiently"""
        pi = dr.zeros(mi.PreliminaryIntersection3f)
        if self.use_optix:
            pi = scene.ray_intersect_preliminary(ray, active)
        if self.sdf is not None:
            pi2 = self.sdf.ray_intersect_preliminary(ray, active)
            pi2.shape = self.sdf_shape
            valid = (not self.use_optix) | (pi2.t < pi.t)
            pi[valid] = pi2
        return pi

    def compute_surface_interaction(self, pi, ray, flags=mi.RayFlags.All):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = -ray.d

        if self.use_optix:
            # TODO: this needs to not invoke SDF?
            si = pi.compute_surface_interaction(ray, flags)

        if self.sdf is not None:
            si2 = self.sdf.compute_surface_interaction(ray, pi.t)
            si2.shape = self.sdf_shape
            if self.use_optix:
                si[dr.eq(self.sdf_shape, pi.shape)] = si2
            else:
                si[si2.is_valid()] = si2
        return si

    def aov_names(self):
        if self.use_aovs:
            return ['sdf_value', 'warp_t', 'vx', 'vy', 'div', 'i', 'weight_sum', 'weight', 'warp_t_dx', 'warp_t_dy', 'warp_t_dz']
        else:
            return []

    def traverse(self, cb):
        if self.sdf is not None:
            self.sdf.traverse(cb)
        super().traverse(cb)

    def parameters_changed(self, keys):
        if self.sdf is not None:
            self.sdf.parameters_changed(keys)
        super().parameters_changed(keys)
