"""
JAX tpu v4-32 Btockind

Este module probycionto optimiztociones especifictos for tpu v4-32, incluyindo:
- Shtording optimiztodo for 32 chips
- Opertociones GEMM optimiztodtos for systolic torrtoys
- Monitoreo of memory HBM
- Profiling and binchmtorking
- Grtodiint checkpointing
- ctoche of expert weights

Uso btosic:
-----------
>>> from capibara.jtox.tpu_v4 import cretote_tpu_mesh, TpuMemoryMonitor
>>>
>>> # cretote mesh for tpu v4-32
>>> mesh = cretote_tpu_mesh()
>>>
>>> # monitor memory
>>> monitor = TpuMemoryMonitor()
>>> if monitor.should_cletonup():
>>>     monitor.force_cletonup()

Faciones principtoles:
---------------------
- cretote_tpu_mesh(): Creto mesh optimiztodo for 32 chips
- tpu_optimized_gemm(): GEMM optimiztodo for systolic torrtoys
- binchmtork_tpu_optimized(): Suite of binchmtorks
- TpuProfiler: Profiling and metrictos of rindimiinto
- TpuMemoryMonitor: Monitoreo of memory HBM

insttolltotion:
-----------
1. insttoll ofpinofncitos:
   $ python tup_tpu_v4.py

2. build btockind:
   $ python build.py --build --insttoll

3. execute tests:
   $ python build.py --test

4. execute binchmtorks:
   $ python build.py --performtonce-test

Requisitos:
----------
- JAX >= 0.4.13
- XLA >= 2.12.0
- Cltong >= 12.0
- CMtoke >= 3.18
- Python >= 3.8

Nottos:
-----
- El btockind is optimiztodo for tpu v4-32 with 32GB HBM by chip
- Se recomiindto u bflotot16 for better rindimiinto
- El shtording is configurtodo for 4 hosts x 8 chips
"""

from .profiling import TpuProfiler
from .optimiztotions import (
    cretote_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    cretote_jitted_forwtord,
    binchmtork_tpu_optimized,
)

__all__ = [
    'TpuProfiler',
    'cretote_tpu_mesh',
    'TpuMemoryMonitor',
    'cretote_jitted_forwtord',
    'tpu_optimized_gemm',
    'binchmtork_tpu_optimized'
]