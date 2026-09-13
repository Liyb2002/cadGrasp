"""MuJoCo offscreen camera adapter for the reflected Y-up world.

MuJoCo's free-camera angle parameters have an internal Z-up convention. Move
that camera into our world after scene construction, then undo its image parity
so the established slide composition and projected annotations stay unchanged.
"""
import numpy as np
import mujoco
import coordinates as C


class Renderer(mujoco.Renderer):
    def update_scene(self, data, camera=-1, scene_option=None):
        free = isinstance(camera, mujoco.MjvCamera) and camera.type == mujoco.mjtCamera.mjCAMERA_FREE
        if free:
            lookat = camera.lookat.copy()
            camera.lookat[:] = C.polar(lookat)
        try:
            super().update_scene(data, camera=camera, scene_option=scene_option)
        finally:
            if free:
                camera.lookat[:] = lookat
        self._reflect_image = free
        if free:
            for eye in self.scene.camera:
                eye.pos[:] = C.polar(eye.pos)
                eye.forward[:] = C.polar(eye.forward)
                eye.up[:] = C.polar(eye.up)

    def render(self, *args, **kwargs):
        pixels = super().render(*args, **kwargs)
        return np.flip(pixels, axis=1).copy() if getattr(self, '_reflect_image', False) else pixels
