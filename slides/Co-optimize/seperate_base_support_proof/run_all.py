"""Reproducible entry: preserve completed workers if the old ledger reader fails."""
import json

import run_experiment as W
import resume_batch as R


if __name__ == '__main__':
    try:
        W.main()
    except ValueError as error:
        if 'invalid literal for int()' not in str(error) or 'pose_' not in str(error):
            raise
        spec = json.loads((W.PROOF/'experiment.json').read_text())
        for group in spec['groups']:
            path = W.BASE/group['id']/'step4/step4.2/proof_cold_v1/run_result.json'
            if not path.is_file() or not json.loads(path.read_text())['complete']:
                raise
        R.main()
