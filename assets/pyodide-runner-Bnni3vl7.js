const i=/__KATA_PLOT_(\d+)__:(.*):__END_KATA_PLOT__/,d=/__KATA_METRIC__:(.*?):(.*?):__END_KATA_METRIC__/,c=/__KATA_TENSOR__:(.*):__END_KATA_TENSOR__/,p="0.26.4",s=`https://cdn.jsdelivr.net/pyodide/v${p}/full/`,m=`
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
`;let _=null;function u(){return new Promise((r,t)=>{if(window.loadPyodide)return r();const a=document.createElement("script");a.src=`${s}pyodide.js`,a.async=!0,a.onload=()=>r(),a.onerror=()=>t(new Error("Failed to load Pyodide script")),document.head.appendChild(a)})}async function l(){return _||(_=(async()=>{if(await u(),!window.loadPyodide)throw new Error("Pyodide failed to attach to window");const r=await window.loadPyodide({indexURL:s});return await r.loadPackage(["numpy","matplotlib"]),r})()),_}function f(r,t){r.includes("__KATA_PLOT_")?t.onPlot(r):r.includes("__KATA_METRIC__")?t.onMetric(r):r.includes("__KATA_TENSOR__")?t.onTensor(r):t.onStdout(r)}async function h(r,t){const a=performance.now();let o;try{o=await l()}catch(e){t.onError(e instanceof Error?e.message:String(e)),t.onDone(performance.now()-a);return}o.setStdout({batched:e=>f(e,t)}),o.setStderr({batched:e=>t.onStderr(e)});const n=m+`
`+r;try{if(/\bimport\s+torch\b|\bfrom\s+torch\b/.test(r))try{await o.loadPackage("torch")}catch{t.onStderr("Note: torch is not yet supported in Pyodide for this build — clone the repo to run torch katas locally.")}await o.loadPackagesFromImports(n),await o.runPythonAsync(n)}catch(e){t.onError(e instanceof Error?e.message:String(e))}finally{t.onDone(performance.now()-a)}}async function y(r){const t={stdout:"",stderr:"",error:null,execution_time_ms:0,metrics:{},plots:[],tensors:[]};return await new Promise(a=>{h(r,{onStdout:o=>{t.stdout+=o+`
`},onStderr:o=>{t.stderr+=o+`
`},onPlot:o=>{const n=o.match(i);if(!n)return;const e={index:parseInt(n[1]),data:`data:image/png;base64,${n[2]}`,format:"png"};t.plots.push(e)},onMetric:o=>{const n=o.match(d);if(!n)return;const e=parseFloat(n[2]);t.metrics[n[1]]=isNaN(e)?n[2]:e},onTensor:o=>{const n=o.match(c);if(n)try{const e=JSON.parse(n[1]);t.tensors.push(e)}catch{}},onError:o=>{t.error=o},onDone:o=>{t.execution_time_ms=o,a()}})}),t}export{h as runInPyodide,y as runInPyodideOneShot};
