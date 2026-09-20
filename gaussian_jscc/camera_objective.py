"""Training-only camera supervision and teacher-position scene initialization.

Source coordinates/cameras are labels, never decoder inputs. No eigenvector
backward or per-point digital coordinate stream is introduced.
"""
import math
import torch
from torch.nn import functional as F
from .render_objective import MultiViewRenderTask, image_distortion


def camera_position_loss(xyz, source_xyz, camera, pixel_scale=2., depth_scale=.1):
    """Robust screen error in pixels plus relative camera-depth error.

    Source frustum selects rows (NOT true occlusion visibility). Predictions
    outside the frustum/behind the camera remain supervised. A source-relative
    safe projection denominator prevents a pole at predicted z=0. The two
    SmoothL1 terms have bounded residual-space slopes, not a guarantee of
    bounded parameter gradients. Source targets are detached.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (pixel_scale,depth_scale)):
        raise ValueError('pixel/depth scales must be positive')
    matrix = camera.world_view_transform.detach().to(xyz)
    def view(points):
        return torch.cat((points, torch.ones_like(points[:,:1])), -1) @ matrix
    src = view(source_xyz.detach()).detach()
    dst = view(xyz)
    near = float(getattr(camera,'znear',.01))
    far = float(getattr(camera,'zfar',float('inf')))
    focal = xyz.new_tensor([camera.image_width/(2*math.tan(camera.FoVx/2)),
                            camera.image_height/(2*math.tan(camera.FoVy/2))])
    half = xyz.new_tensor([camera.image_width/2,camera.image_height/2])
    target_xy = src[:,:2] / src[:,2:3].clamp_min(near) * focal
    valid = (src[:,2]>near) & (src[:,2]<far) & (target_xy.abs()<half).all(-1)
    if not valid.any():
        zero = xyz.sum()*0
        return zero, dict(pixel_loss=zero.detach(),depth_loss=zero.detach(),
                          pixel_error_mean=zero.detach(),source_frustum_count=0)
    src, dst, target_xy = src[valid],dst[valid],target_xy[valid]
    denominator = torch.maximum(dst[:,2:3],.1*src[:,2:3]).clamp_min(near)
    error = dst[:,:2]/denominator*focal-target_xy
    pixel = F.smooth_l1_loss(error/pixel_scale,torch.zeros_like(error))
    depth_error = (dst[:,2]-src[:,2])/src[:,2].clamp_min(near)/depth_scale
    depth = F.smooth_l1_loss(depth_error,torch.zeros_like(depth_error))
    return pixel+depth, dict(pixel_loss=pixel.detach(),depth_loss=depth.detach(),
                            pixel_error_mean=error.detach().norm(dim=-1).mean(),
                            source_frustum_count=int(valid.sum()))


class CameraInitializationTask(MultiViewRenderTask):
    """Full-scene RGB at source XYZ + camera-supervised predicted XYZ.

    Inherited streamed backward retains one rasterizer graph at a time. Replay
    then propagates the scene VJP through the codec with identical noise.
    Attributes (including covariance) get RGB gradients; XYZ gets projection
    gradients. Shared codec parameters can receive both, with explicit weight.
    """
    def __init__(self,cameras,reference,degree,source_xyz,white_background=False,
                 geometry_weight=.01,pixel_scale=2.,depth_scale=.1):
        super().__init__(cameras,reference,degree,white_background)
        if not math.isfinite(geometry_weight) or geometry_weight < 0:
            raise ValueError('geometry weight must be finite and nonnegative')
        self.source_xyz = source_xyz.detach()
        self.geometry_weight = geometry_weight
        self.pixel_scale, self.depth_scale = pixel_scale, depth_scale

    def _terms(self,scene):
        from .rendering import render
        if len(scene)!=len(self.source_xyz):
            raise ValueError('teacher XYZ must match retained scene rows exactly')
        source = self.source_xyz.to(scene)
        self.measurements = []
        for camera in self.cameras:
            # Construct per camera so streamed autograd frees each cat graph.
            teacher_scene = torch.cat((source,scene[:,3:]),-1)
            image = render(teacher_scene,camera,self.degree,self.white_background)
            rgb = image_distortion(image,self.reference.get(camera,scene.device))
            geometry,details = camera_position_loss(scene[:,:3],source,camera,
                                                    self.pixel_scale,self.depth_scale)
            self.measurements.append(dict(teacher_xyz_image_mse=float(rgb.detach()),
                                          camera_geometry_loss=float(geometry.detach()),
                                          geometry_contribution=float(geometry.detach())*self.geometry_weight,
                                          **{k:float(v) for k,v in details.items()}))
            yield rgb+self.geometry_weight*geometry

    def _report(self):
        self.stats = {k:sum(r[k] for r in self.measurements)/len(self.measurements)
                      for k in self.measurements[0]}
        self.stats['views_per_step'] = len(self.cameras)

    def __call__(self,scene,retained_ids=None):
        result = super().__call__(scene,retained_ids)
        self._report()
        return result

    def backward_scene(self,scene):
        result = super().backward_scene(scene)
        self._report()
        return result
