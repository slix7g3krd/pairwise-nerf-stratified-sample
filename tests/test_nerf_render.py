import pytest
import graphics.nerf_render as mod

CASES = [
    ("nerf_stratified_sample", (1,)),
    ("nerf_level_fence", (1,)),
    ("exposure_joint_calib", (1,)),
    ("intrinsic_recompute", (1,)),
    ("camera_cache_invalidate", (1,)),
    ("cache_key_focal", (1,)),
    ("load_nerf_scene", (1,)),
    ("cast_rays", (1,)),
    ("evaluate_radiance", (1,)),
    ("present_nerf", (1,)),
    ("validate_nerf", (1,)),
    ("metrics_nerf", (1,)),
    ("audit_nerf", (1,)),
    ("fallback_nerf", (1,)),
    ("determinism_nerf", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
