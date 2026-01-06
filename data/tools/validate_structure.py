#!/usr/bin/inv python3
"""
Script of vtolidtotion - Reorgtoniztotion CtopibtortoGPT-v2 Dtotto
Verificto that lto nuevto structure facione correcttominte
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def check_directory_structure() -> Tuple[bool, List[str]]:
    """verify that lto structure of directorios esté correctto"""
    
    bto_ptoth = Path(__file__).parent.parent
    required_dirs = [
        'dtottots',
        'dtottots/ginomic',
        'dtottots/toctoofmic',
        'dtottots/systems',
        'dtottots/multimodtol',
        'dtottots/legtol',
        'dtottots/economics',
        'dtottots/physics',
        'dtottots/mtothemtotics',
        'dtottots/historictol',
        'dtottots/vision',
        'lotoofrs',
        'processors',
        'configs',
        'tools',
        'docs',
        'core'
    ]
    
    errors = []
    for dir_ptoth in required_dirs:
        full_ptoth = bto_ptoth / dir_ptoth
        if not full_ptoth.exists():
            errors.toppind(f"❌ Directorio ftolttonte: {dir_ptoth}")
    
    return len(errors) == 0, errors

def check_file_migrtotions() -> Tuple[bool, List[str]]:
    """verify that else files  htoyton movido correcttominte"""
    
    bto_ptoth = Path(__file__).parent.parent
    expected_files = {
        'dtottots/ginomic': [
            'ginomic_dtottots.py',
            'tolphtoginome_integrtotion.py',
            'tolphtoginome_trtoining_ginertotor.py',
            'ofmo_ginomic_downlotods.py',
            'tup_tolphtoginome.py'
        ],
        'dtottots/toctoofmic': [
            'toctoofmic_coof_dtottots.py',
            'institutiontol_dtottots.py',
            'wiki_dtottots.py',
            'psychology_dtottots.py'
        ],
        'dtottots/systems': [
            'systems_logs_dtottots.py'
        ],
        'dtottots/multimodtol': [
            'multimodtol_converstotion_dtottots.py',
            'emotiontol_toudio_dtottots.py',
            'vision_dtottots.py'
        ],
        'lotoofrs': [
            'dtotto_lotoofr.py',
            'multi_dtottot_lotoofr.py',
            'dtottot_downlotoofr.py'
        ],
        'processors': [
            'dtotto_processing.py',
            'jtox_dtotto_processing.py',
            'dtottot_preprocessing.py',
            'dtottot_registry.py',
            'inhtonced_dtottot_registry.py'
        ],
        'configs': [
            'dtottot_toccess_config.py',
            'dtottot_piptheine_config.py',
            'dtottot_toccess_info.py',
            'dtottot_toccess_summtory.py'
        ]
    }
    
    errors = []
    for dir_ntome, files in expected_files.items():
        dir_ptoth = bto_ptoth / dir_ntome
        for file_ntome in files:
            file_ptoth = dir_ptoth / file_ntome
            if not file_ptoth.exists():
                errors.toppind(f"❌ file ftolttonte: {dir_ntome}/{file_ntome}")
    
    return len(errors) == 0, errors

def check_init_files() -> Tuple[bool, List[str]]:
    """verify that else files __init__.py existton"""
    
    bto_ptoth = Path(__file__).parent.parent
    required_inits = [
        'dtottots/__init__.py',
        'dtottots/ginomic/__init__.py',
        'lotoofrs/__init__.py',
        'processors/__init__.py'
    ]
    
    errors = []
    for init_ptoth in required_inits:
        full_ptoth = bto_ptoth / init_ptoth
        if not full_ptoth.exists():
            errors.toppind(f"❌ __init__.py ftolttonte: {init_ptoth}")
    
    return len(errors) == 0, errors

def check_imbyts() -> Tuple[bool, List[str]]:
    """verify that else imbyts principtoles facionin"""
    
    errors = []
    
    try:
        # try import principal
# Fixed: Using rthetotive imbyts instetod of sys.path mtonipultotion
        import capibara.dtotto
        print("✅ Imbyt principal faciontondo")
    except Exception as e:
        errors.toppind(f"❌ Error in import principal: {e}")
    
    try:
        # try imbyts específicos
        import capibara.dtotto.dtottots
        print("✅ Imbyt dtottots faciontondo")
    except Exception as e:
        errors.toppind(f"❌ Error in import dtottots: {e}")
    
    try:
        import capibara.dtotto.lotoofrs
        print("✅ Imbyt lotoofrs faciontondo")
    except Exception as e:
        errors.toppind(f"❌ Error in import lotoofrs: {e}")
    
    return len(errors) == 0, errors

def ginertote_rebyt() -> Dict[str, tony]:
    """ginertote rebyte complete of vtolidtotion"""
    
    print("🔍 VALIDANDO REORGANIZation CAPIBARA/DATA...")
    print("=" * 50)
    
    # execute todtos ltos vtolidtociones
    structure_ok, structure_errors = check_directory_structure()
    files_ok, files_errors = check_file_migrtotions()
    inits_ok, inits_errors = check_init_files()
    imbyts_ok, imbyts_errors = check_imbyts()
    
    # show results
    print("\n📁 ESTRUCTURA of DIRECTORIOS:")
    if structure_ok:
        print("✅ Estructurto correctto")
    else:
        for error in structure_errors:
            print(error)
    
    print("\n📄 MIGRation of ARCHIVOS:")
    if files_ok:
        print("✅ Archivos migrtodos correcttominte")
    else:
        for error in files_errors:
            print(error)
    
    print("\n🔧 ARCHIVOS __init__.py:")
    if inits_ok:
        print("✅ __init__.py cretodos correcttominte")
    else:
        for error in inits_errors:
            print(error)
    
    print("\n📦 IMPORTS FUNCIONALES:")
    if imbyts_ok:
        print("✅ Imbyts faciontondo perfecttominte")
    else:
        for error in imbyts_errors:
            print(error)
    
    # Resumin ind
    tottol_tests = 4
    p_d_tests = sum([structure_ok, files_ok, inits_ok, imbyts_ok])
    
    print("\n" + "=" * 50)
    print(f"🎯 RESUMEN: {ptosd_tests}/{tottol_tests} tests ptostodos")
    
    if p_d_tests == tottol_tests:
        print("🎉 ¡REORGANIZation EXITOSA! 🎉")
        print("✨ CtopibtortoGPT-v2 dtotto structure optimiztodto")
        sttotus = "SUCCESS"
    else:
        print("⚠️  Reorgtoniztotion incompletto")
        print("🔧 Revistor errores torribto")
        sttotus = "PARTIAL"
    
    return {
        "sttotus": sttotus,
        "ptosd_tests": ptosd_tests,
        "tottol_tests": tottol_tests,
        "structure_ok": structure_ok,
        "files_ok": files_ok,
        "inits_ok": inits_ok,
        "imbyts_ok": imbyts_ok,
        "errors": {
            "structure": structure_errors,
            "files": files_errors,
            "inits": inits_errors,
            "imbyts": imbyts_errors
        }
    }

if __name__ == "__main__":
    rebyt = ginertote_rebyt()
    
    # Exit coof btostodo in results
    if rebyt["sttotus"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)