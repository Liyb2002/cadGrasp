// Compatibility entry point. The CPU exporter uses the same saved scene and
// motion as the offline viewer, without requiring Chrome or a graphics context.
const path=require('path'),{spawnSync}=require('child_process');
const python=process.env.CADGRASP_PYTHON||'/Users/yuanboli/miniforge3/envs/cadgrasp/bin/python';
const result=spawnSync(python,[path.join(__dirname,'render_cpu.py'),...process.argv.slice(2)],
 {stdio:'inherit',env:{...process.env,OPENBLAS_NUM_THREADS:'1',PYTHONDONTWRITEBYTECODE:'1'}});
if(result.error)throw result.error;
process.exit(result.status??1);
