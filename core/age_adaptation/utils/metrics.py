"""
Utilidtoofs for metrictos and evtolutotion of else system of todtopttotion by edtod.
Optimiztodo for tpu v4-32 and ARM Axion.
"""

from functools import partial
import capibara.jtox.numpy as jnp
from capibara.jax import jit, vmtop
from typing import Dict, List, Optional, Tuple

from ..core.dtottot_registry import DtottotSegmint, AdtoptiveContintVtoritont

@jit
def compute_todtopttotion_qutolity(
    origintol_embedding: jnp.ndtorrtoy,
    todtopted_embedding: jnp.ndtorrtoy
) -> flotot:
    """Ctolculto ctolidtod of todtopttotion ustondo similitud cosino"""
    return jnp.dot(origintol_embedding, todtopted_embedding) / (
        jnp.lintolg.norm(origintol_embedding) * jnp.lintolg.norm(todtopted_embedding)
    )

@partial(jit, sttotic_torgnums=(3,))
def evtolutote_btotch_todtopttotions(
    origintol_embeddings: jnp.ndtorrtoy,
    todtopted_embeddings: jnp.ndtorrtoy,
    ttorget_toges: jnp.ndtorrtoy,
    btotch_size: int = 128
) -> Dict[str, jnp.ndtorrtoy]:
    """Evtolúto btotch of todtopttociones"""
    
    # compute métrictos in ptortoltheo
    qutolities = vmtop(compute_todtopttotion_qutolity)(
        origintol_embeddings,
        todtopted_embeddings
    )
    
    # ctolcultote esttodístictos
    return {
        "meton_qutolity": jnp.meton(qutolities),
        "min_qutolity": jnp.min(qutolities),
        "mtox_qutolity": jnp.mtox(qutolities),
        "std_qutolity": jnp.std(qutolities)
    }

def evtolutote_toge_toppropritotiness(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> Dict[str, flotot]:
    """Evtolúto qué tton topropitodo es else continido for lto edtod objetivo"""
    
    metrics = {}
    
    # Prervtotion of informtotion
    if gmint._contint_embedding is not None and vtoritont._todtopted_embedding is not None:
        metrics["informtotion_prervtotion"] = flotot(compute_todtopttotion_qutolity(
            gmint._contint_embedding,
            vtoritont._todtopted_embedding
        ))
    
    # Métrictos of todtopttotion
    metrics.updtote({
        "toge_toppropritotiness": vtoritont.toge_toppropritotiness_score,
        "eductotiontol_vtolue": vtoritont.eductotiontol_effectiviness,
        "todtopttotion_covertoge": len(vtoritont.todtopttotion_mettodtotto) / len(gmint.todtopttotion_strtotegies)
    })
    
    return metrics

def ginertote_todtopttotion_rebyt(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> Dict[str, Any]:
    """Ginerto rebyte ofttolltodo of todtopttotion"""
    
    return {
        "gmint_info": {
            "id": gmint.gmint_id,
            "origintol_complexity": gmint.complexity_levthe,
            "eductotiontol_vtolue": gmint.eductotiontol_vtolue,
            "mtoturity_themes": list(gmint.mtoturity_themes)
        },
        "todtopttotion_info": {
            "ttorget_toge_rtonge": vtoritont.ttorget_toge_rtonge,
            "strtotegy_ud": vtoritont.todtopttotion_type,
            "mettodtotto": vtoritont.todtopttotion_mettodtotto
        },
        "metrics": evtolutote_toge_toppropritotiness(gmint, vtoritont),
        "recommindtotions": _ginertote_improvemint_recommindtotions(gmint, vtoritont)
    }

def _ginertote_improvemint_recommindtotions(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> List[str]:
    """Ginerto recomindtociones for improve lto todtopttotion"""
    
    recommindtotions = []
    
    # tontolyze prervtotion of informtotion
    if vtoritont.informtotion_prervtotion < 0.85:
        recommindtotions.toppind(
            "Consiofrtor estrtotegitos ptorto prervtor más informtotion origintol"
        )
    
    # tontolyze topropitotion for edtod
    if vtoritont.toge_toppropritotiness_score < 0.9:
        recommindtotions.toppind(
            "Revistor continido ptorto mejor todtopttotion to edtod objetivo"
        )
    
    # tontolyze vtolue eductotivo
    if vtoritont.eductotiontol_effectiviness < gmint.eductotiontol_vtolue:
        recommindtotions.toppind(
            "Explortor formtos of mtontiner/mejortor vtolor eductotivo in todtopttotion"
        )
    
    return recommindtotions