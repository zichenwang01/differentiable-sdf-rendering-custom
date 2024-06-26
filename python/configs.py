import sys
import inspect

import integrators.sdf_direct_reparam
import integrators.sdf_silhouette_reparam
import integrators.sdf_simple_shading_reparam
import integrators.sdf_prb_reparam

from warp import WarpField2D, WarpFieldConvolution, DummyWarpField

class BaseConfig:
    def __init__(self):
        # general parameters
        self.name = 'default'
        self.pretty_name = 'baseconfig'
        self.mask_optimizer = False
        self.use_autodiff = True
        self.use_finite_differences = False
        
        # optimization parameters
        self.learning_rate = 1e-2
        # self.learning_rate = 4e-2
        # self.n_iter = 512
        self.n_iter = 5000
        self.spp = 64
        # self.spp = 16
        self.integrator = 'sdf_direct_reparam'
        
        # self.primal_spp_mult = 4
        self.primal_spp_mult = 1
        self.edge_epsilon = 0.01
        # self.edge_epsilon = 0.001
        self.refined_intersection = False
        
        # clamp parameters
        self.geom_clamp_threshold = 0.05
        self.warp_weight_strategy = 6

    def get_warpfield(self, sdf_object):
        warp = WarpField2D(
            sdf_object, 
            weight_strategy=self.warp_weight_strategy,
            edge_eps=self.edge_epsilon
        )
        warp.clamping_thresh = self.geom_clamp_threshold
        return warp


class Warp(BaseConfig):
    """ Our main method.
        This handles both primary and secondary rays (check this)"""
    
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Ours'
        self.pretty_name_short = 'Ours'
        self.name = 'warp'


class WarpPRB(BaseConfig):
    """ Our method implemented with path replay.
        This handles both primary and secondary rays """
        
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Ours'
        self.pretty_name_short = 'Ours'
        self.name = 'warpprb'
        self.integrator = 'sdf_prb_reparam'


class WarpPrimary(BaseConfig):
    """ Our method.
        This handles only primary rays """
    
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Ours (primary only)'
        self.pretty_name_short = 'Ours (primary only)'
        self.name = 'warpprimary'

    def get_warpfield(self, sdf_object):
        warp = WarpField2D(
            sdf_object, 
            weight_strategy=self.warp_weight_strategy,
            edge_eps=self.edge_epsilon
        )
        warp.clamping_thresh = self.geom_clamp_threshold
        warp.max_reparam_depth = 0 # main difference
        return warp


class WarpPRBPrimary(BaseConfig):
    """ Our method implemented with path replay
        This handles only primary rays """

    def __init__(self):
        super().__init__()
        self.pretty_name = 'Ours'
        self.pretty_name_short = 'Ours'
        self.name = 'warpprbprimary'
        self.integrator = 'sdf_prb_reparam'

    def get_warpfield(self, sdf_object):
        warp = WarpField2D(sdf_object, weight_strategy=self.warp_weight_strategy,
                           edge_eps=self.edge_epsilon)
        warp.clamping_thresh = self.geom_clamp_threshold
        warp.max_reparam_depth = 0
        return warp


class WarpNotNormalized(Warp):
    """ Our method without normalization of the warp field.
        This is used for a figure in the paper."""
        
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Ours (not normalized)'
        self.pretty_name_short = 'Ours (not normalized)'

        self.name = 'warpnotnormalized'

    def get_warpfield(self, sdf_object):
        warp = WarpField2D(sdf_object, weight_strategy=self.warp_weight_strategy,
                           edge_eps=self.edge_epsilon)
        warp.clamping_thresh = self.geom_clamp_threshold
        warp.normalize_warp_field = False
        return warp


class ConvolutionWarp(BaseConfig):
    """ Warp Field method [Bangaru 2020]
        Deafult 16 auxiliary rays"""

    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020'
        self.pretty_name_short = 'Bangaru et al.'
        self.name = 'conv'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=16)


class ConvolutionWarp2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020 (2 aux. rays)'
        self.pretty_name_short = 'Bangaru et al. (2 aux. rays)'
        self.name = 'conv2'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=2)


class ConvolutionWarp4(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020 (4 aux. rays)'
        self.pretty_name_short = 'Bangaru et al. (4 aux. rays)'
        self.name = 'conv4'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=4)


class ConvolutionWarp8(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020 (8 aux. rays)'
        self.pretty_name_short = 'Bangaru et al. (8 aux. rays)'
        self.name = 'conv8'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=8)


class ConvolutionWarp16(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020 (16 aux. rays)'
        self.pretty_name_short = 'Bangaru et al. (16 aux. rays)'
        self.name = 'conv16'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=16)


class ConvolutionWarp32(BaseConfig):
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Bangaru et al. 2020 (32 aux. rays)'
        self.pretty_name_short = 'Bangaru et al. (32 aux. rays)'
        self.name = 'conv32'

    def get_warpfield(self, sdf_object):
        return WarpFieldConvolution(sdf_object, n_aux_rays=32)


class OnlyShadingGrad(BaseConfig):
    """ Naive autodiff method.
        This completely ignores discontinuities, which usually breaks optimization."""
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Only shading gradient'
        self.pretty_name_short = 'Only shading gradient'
        self.name = 'onlyshading'

    def get_warpfield(self, sdf_object):
        return DummyWarpField(sdf_object)


class FiniteDifferences(BaseConfig):
    """ Finite differences method. 
        This cannot really be used for any optimizations and is only useful for gradient validation."""
    def __init__(self):
        super().__init__()
        self.pretty_name = 'Finite differences'
        self.pretty_name_short = 'FD'
        self.name = 'fd'
        self.use_finite_differences = True

    def get_warpfield(self, sdf_object):
        return None


CONFIGS = {name.lower(): obj
           for name, obj in inspect.getmembers(sys.modules[__name__])
           if inspect.isclass(obj)}

def get_config(config):
    config = config.lower()
    if config in CONFIGS:
        return CONFIGS[config]()
    else:
        raise ValueError(f"Could not find config {config}!")


def apply_cmdline_args(config, unknown_args, return_dict=False):
    """Update flat dictionnary or object from unparsed argpase arguments"""
    
    # Always return a dict if input is a dict
    return_dict |= isinstance(unknown_args, dict)  
    
    unused_args = dict() if return_dict else list()
    if unknown_args is None:
        return unused_args

    def parse_value(dest_type, value):
        if value == 'None':
            return None
        if dest_type == bool:
            return value.lower() in ['true', '1']
        return dest_type(value)

    # Parse input list of strings key=value
    input_args = {}
    if isinstance(unknown_args, list):
        for s in unknown_args:
            if '=' in s:
                k = s[2:s.index('=')]
                v = s[s.index('=') + 1:]
            else:
                k = s[2:]
                v = True
            input_args[k] = v
    else:
        input_args = unknown_args

    for k, v in input_args.items():
        if isinstance(config, dict) and k in config:
            old_v = config[k]
            config[k] = parse_value(type(old_v), v)
            print(f"Overriden parameter: {k} = {old_v} -> {config[k]}")
        elif hasattr(config, k):
            old_v = getattr(config, k)
            setattr(config, k, parse_value(type(old_v), v))
            print(f"Overriden parameter: {k} = {old_v} -> {getattr(config, k)}")
        else:
            if return_dict:
                unused_args[k] = v
            else:
                unused_args.append('--' + k + '=' + v)
    return unused_args
