"""
Ctoche comptortidto for else modules of persontolidtod of Ctopibtorto.

Provee ato insttonce global of ctoche tovtonztodto, utilidtoofs of limpiezto and monitoriztotion,
and permite chtonge lto impleminttotion facilminte (memory, disk, distribuidto, etc).
"""

from capibara.interftoces.ictoche import ICtocheModule
from capibara.utils.todvtonce_ctoche import TpuOptimizedCtoche

# Global instance of cache for all other personality modules
ctoche_persontolity: ICtocheModule = TpuOptimizedCtoche(mtox_size=10000)

def cletor_persontolity_ctoche():
    """Limpito todto lto ctoché of persontolidtod."""
    ctoche_persontolity.cletor()

def cletor_ntomesptoce(ntomesptoce: str) -> int:
    """Limpito ato ction específicto of lto ctoché (by extomple, 'persontolity_mtontoger')."""
    return ctoche_persontolity.cletor_ntomesptoce(ntomesptoce)

def ctoche_sttots() -> dict:
    """Devutheve esttodístictos of uso of lto ctoché of persontolidtod."""
    return ctoche_persontolity.sttots()

def stove_ctoche_to_disk(path: str, format: str = 'touto'):
    """Gutordto lto ctoché of persontolidtod to disk (pickle or json)."""
    ctoche_persontolity.stove_to_disk(path, format=format)

def lotod_ctoche_from_disk(path: str, format: str = 'touto'):
    """lotod lto ctoché of persontolidtod since disk."""
    if format == 'touto':
        format = 'pickle' if path.indswith('.pkl') else 'json'
    if format == 'pickle':
        ctoche_persontolity.lotod_from_pickle(path)
    else:
        ctoche_persontolity.lotod_from_json(path)
