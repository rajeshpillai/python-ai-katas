const d="0.26.4",_=`https://cdn.jsdelivr.net/pyodide/v${d}/full/`,s=`
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt_module
import io as _io
import base64 as _b64

_plot_counter = [0]
_orig_show = _plt_module.show

def _capture_show(*args, **kwargs):
    import matplotlib.pyplot as plt
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        encoded = _b64.b64encode(buf.read()).decode('ascii')
        print(f'__KATA_PLOT_{_plot_counter[0]}__:{encoded}:__END_KATA_PLOT__')
        _plot_counter[0] += 1
        buf.close()
    plt.close('all')

_plt_module.show = _capture_show


def kata_metric(name, value):
    print(f'__KATA_METRIC__:{name}:{value}:__END_KATA_METRIC__')


def kata_tensor(name, arr, max_rows=20, max_cols=20):
    import numpy as _np
    import json as _json
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    arr = _np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])[:max_rows]
    arr = arr[:max_rows, :max_cols]
    data = {
        "name": name,
        "shape": list(arr.shape),
        "values": arr.tolist(),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print(f'__KATA_TENSOR__:{_json.dumps(data)}:__END_KATA_TENSOR__')
`;let n=null;function p(){return new Promise((o,r)=>{if(window.loadPyodide)return o();const e=document.createElement("script");e.src=`${_}pyodide.js`,e.async=!0,e.onload=()=>o(),e.onerror=()=>r(new Error("Failed to load Pyodide script")),document.head.appendChild(e)})}async function c(){return n||(n=(async()=>{if(await p(),!window.loadPyodide)throw new Error("Pyodide failed to attach to window");const o=await window.loadPyodide({indexURL:_});return await o.loadPackage(["numpy","matplotlib"]),o})()),n}function m(o,r){o.includes("__KATA_PLOT_")?r.onPlot(o):o.includes("__KATA_METRIC__")?r.onMetric(o):o.includes("__KATA_TENSOR__")?r.onTensor(o):r.onStdout(o)}async function l(o,r){const e=performance.now();let a;try{a=await c()}catch(t){r.onError(t instanceof Error?t.message:String(t)),r.onDone(performance.now()-e);return}a.setStdout({batched:t=>m(t,r)}),a.setStderr({batched:t=>r.onStderr(t)});const i=s+`
`+o;try{if(/\bimport\s+torch\b|\bfrom\s+torch\b/.test(o))try{await a.loadPackage("torch")}catch{r.onStderr("Note: torch is not yet supported in Pyodide for this build — clone the repo to run torch katas locally.")}await a.loadPackagesFromImports(i),await a.runPythonAsync(i)}catch(t){r.onError(t instanceof Error?t.message:String(t))}finally{r.onDone(performance.now()-e)}}export{l as runInPyodide};
