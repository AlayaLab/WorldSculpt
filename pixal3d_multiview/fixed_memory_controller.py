"""A non-adaptive memory controller that pins gradient checkpointing to a FIXED
ratio, as an alternative to the adaptive `LinearMemoryController`.

`mem_ratio` is the fraction of "full (no-checkpoint) memory" the model is allowed
to use; the model maps it to a fixed number of checkpointed transformer blocks
(see `SparseTransformerElasticMixin.with_mem_ratio`):

    mem_ratio = 1.0  ->  0 / N blocks checkpointed (fastest, most memory)
    mem_ratio = 0.5  ->  ~N/2 blocks checkpointed
    mem_ratio = 0.0  ->  all N blocks checkpointed (slowest, least memory)

Unlike `LinearMemoryController`, `get_mem_ratio` returns the SAME value for every
sample regardless of `input_size` — fully deterministic, with no per-sample
memory model and therefore NO extrapolation/OOM risk on rare large samples. The
trade-off is no speed adaptivity (small samples get checkpointed as much as large
ones). For Pixal3D stage 2/3 a fixed `mem_ratio=0.5` keeps even the largest kept
sample (32764 tokens) at ~55 GB on a 94 GB GPU (measured), well under capacity.

This lives in `pixal3d_multiview` (new code) and is registered into the
`pixal3d.utils.elastic_utils` namespace when the finetuned model is built, so the
trainer's `getattr(elastic_utils, cfg['name'])` lookup finds it. No original
pixal3d code is modified. Select it via the trainer's `elastic` config:

    "elastic": { "name": "FixedMemoryController", "args": { "mem_ratio": 0.5 } }
"""
from contextlib import contextmanager
import torch

from pixal3d.utils.elastic_utils import MemoryController


class FixedMemoryController(MemoryController):
    def __init__(self, mem_ratio=0.5, available_memory=None, device=None, **kwargs):
        # **kwargs swallows adaptive-only keys (target_ratio, max_mem_ratio_start,
        # buffer_size, ...) so the same launch scripts work without edits.
        self.mem_ratio = float(mem_ratio)
        self.device = device if device is not None else torch.cuda.current_device()
        self.available_memory = available_memory or \
            torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        self._last_memory = 0.0
        self._last_input_size = None
        self._last_mem_ratio = []
        self.step = 0

    def __repr__(self):
        return (f'FixedMemoryController(mem_ratio={self.mem_ratio}, '
                f'available_memory={self.available_memory})')

    @contextmanager
    def record(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self._last_input_size = None
        self._last_mem_ratio = []
        yield
        self._last_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
        # update_run_states (inherited) appended the exact mem_ratio of each
        # ElasticModule forward during the step; average for logging.
        if self._last_mem_ratio:
            self._last_mem_ratio = sum(self._last_mem_ratio) / len(self._last_mem_ratio)
        else:
            self._last_mem_ratio = self.mem_ratio
        self.step += 1

    def get_mem_ratio(self, input_size):
        return self.mem_ratio

    def state_dict(self):
        return {'mem_ratio': self.mem_ratio}

    def load_state_dict(self, state_dict):
        self.mem_ratio = float(state_dict.get('mem_ratio', self.mem_ratio))

    def log(self):
        return {
            'mem_ratio': self._last_mem_ratio if isinstance(self._last_mem_ratio, float) else self.mem_ratio,
            'memory': self._last_memory,
            'input_size': self._last_input_size if self._last_input_size is not None else 0,
        }
