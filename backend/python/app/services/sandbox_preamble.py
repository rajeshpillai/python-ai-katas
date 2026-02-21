"""Python preamble injected before every user code execution.

Provides:
- Matplotlib plot capture (plt.show() -> base64 PNG sentinels)
- kata_metric(name, value) -> structured metric sentinels
- kata_tensor(name, array) -> structured tensor sentinels
"""

SANDBOX_PREAMBLE = r'''
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
    # Report a metric to the kata output panel.
    print(f'__KATA_METRIC__:{name}:{value}:__END_KATA_METRIC__')


def kata_tensor(name, arr, max_rows=20, max_cols=20):
    # Display a tensor/matrix as a visual heatmap in the output panel.
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
'''
