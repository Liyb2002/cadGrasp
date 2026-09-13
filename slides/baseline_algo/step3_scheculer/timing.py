"""Wall-clock stage timings, separate from geometry and feasibility evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time


class StageTimings:
    def __init__(self, paths, settings):
        self.paths = list(map(Path, paths))
        self.started = time.monotonic()
        self.record = dict(schema='baseline_stage_timings_v1', complete=False,
                           started_at=datetime.now(timezone.utc).isoformat(),
                           settings=settings, stages=[])

    def save(self):
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.tmp')
            temporary.write_text(json.dumps(self.record, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)

    def call(self, stage, objects, operation):
        started = time.monotonic()
        row = dict(stage=stage, objects=list(objects), complete=False,
                   scope='Wall time of this command for all listed objects; includes interpreter startup and I/O')
        self.record['stages'].append(row)
        try:
            result = operation()
            row.update(complete=True, returncode=result.returncode)
            return result
        except BaseException as error:
            row['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic()-started
            self.save()
            print(stage, 'wall seconds', round(row['elapsed_seconds'], 3), flush=True)

    def finish(self, returncode=None, error=None):
        self.record.update(complete=error is None, returncode=returncode,
                           elapsed_seconds=time.monotonic()-self.started,
                           finished_at=datetime.now(timezone.utc).isoformat())
        totals = {}
        for row in self.record['stages']:
            number = row['stage'][4]
            totals[number] = totals.get(number, 0.)+row['elapsed_seconds']
        self.record['step_seconds'] = totals
        self.record['outside_stage_seconds'] = self.record['elapsed_seconds']-sum(totals.values())
        self.record['outside_stage_scope'] = 'Resume checks, final verdict checks and orchestration; reused stages are not fresh computation'
        if error is not None:
            self.record['error'] = type(error).__name__+': '+str(error)
        self.save()
