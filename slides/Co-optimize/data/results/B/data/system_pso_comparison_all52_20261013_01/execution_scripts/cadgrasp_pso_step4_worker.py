from pathlib import Path
import json,sys,time,resource,traceback
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
payload=json.loads(Path(sys.argv[1]).read_text());tree=Path(payload['frozen']);sys.path[:0]=[str(tree/'slides/Co-optimize/helper_func'),str(tree/'slides/Co-optimize/step4.2')]
import _bootstrap
from run_oracle import worker
began=time.perf_counter();result=worker((payload['id'],payload['options']));result['measured_wall_seconds']=time.perf_counter()-began
path=Path(payload['result']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
