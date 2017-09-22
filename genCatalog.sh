#! /bin/bash
#SBATCH -p r3 -o gc.stdout -e gc.stderr --mail-user=xinlun.cheng@epfl.ch --mail-type=ALL
python genCatalog.py ../BOSS ../BOSS/BOSS_candidates.fits
