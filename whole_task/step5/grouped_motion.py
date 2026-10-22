"""Render the grouped result with a broader, still screened native grasp pool."""
from pathlib import Path
from .video.grasps import GraspPool


class ExpandedGraspPool(GraspPool):
    def choose(self, robot, source, target, work_faces=None, fixture=None, transport_target=None):
        original = self.candidate_limit
        try:
            for limit in (max(original, 3200), max(original, 9000), 50000):
                self.candidate_limit = limit
                try:
                    return super().choose(robot, source, target, work_faces, fixture, transport_target)
                except ValueError as error:
                    if 'No floor-clear, reachable parallel-jaw grasp' not in str(error):
                        raise
                    print('EXPAND native grasp pool', limit, str(error), flush=True)
            raise ValueError('All expanded native grasp candidates exhausted')
        finally:
            self.candidate_limit = original


def configure():
    from .grouped import configure as shared
    shared()
    from .video import workflow, motion_cache
    workflow.GraspPool = ExpandedGraspPool
    original_save = motion_cache.save
    if not getattr(original_save, '_grouped_motion_bound', False):
        def save(work, native_sources):
            native_sources.append(Path(__file__).resolve())
            return original_save(work, native_sources)
        save._grouped_motion_bound = True
        motion_cache.save = save


if __name__ == '__main__':
    configure()
    from .run_grouped import run
    run()
