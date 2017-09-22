import argparse
import os
import numpy as np
import pyfits as pf
from astropy.io import fits


# Commandline argument parser
parser = argparse.ArgumentParser()
parser.add_argument("dataDir", help="Directory for data", type=str)
parser.add_argument("saveFile", help="Output file", type=str)
args = parser.parse_args()
dataDir = args.dataDir
saveFile = args.saveFile
# Generate catalog
er = []
sc = []
zs = []
ra = []
dec = []
z = []
plate = []
mjd = []
fid = []
w = []
snp = []
snn = []
sns = []
o3w = []
o3s = []
for each in os.listdir(dataDir):
    tmpDir = os.path.join(dataDir, each)
    if os.path.isdir(tmpDir):
        dDir = os.path.join(tmpDir, "candidates_doublet.txt")
        try:
            temp = np.loadtxt(dDir)
            if temp.ndim == 2:
                er.extend(temp[:, 0])
                sc.extend(temp[:, 1])
                zs.extend(temp[:, 2])
                ra.extend(temp[:, 3])
                dec.extend(temp[:, 4])
                z.extend(temp[:, 5])
                plate.extend(temp[:, 6])
                mjd.extend(temp[:, 7])
                fid.extend(temp[:, 8])
                w.extend(temp[:, 9])
                snp.extend(temp[:, 10])
                snn.extend(temp[:, 11])
                sns.extend(temp[:, 12])
                o3w.extend(temp[:, 13])
                o3s.extend(temp[:, 14])
            else:
                er.append(temp[0])
                sc.append(temp[1])
                zs.append(temp[2])
                ra.append(temp[3])
                dec.append(temp[4])
                z.append(temp[5])
                plate.append(temp[6])
                mjd.append(temp[7])
                fid.append(temp[8])
                w.append(temp[9])
                snp.append(temp[10])
                snn.append(temp[11])
                sns.append(temp[12])
                o3w.append(temp[13])
                o3s.append(temp[14])
        except Exception as reason:
            print(each + ' ' + str(reason))
c0 = fits.Column(name="einsteinRadius", array=np.array(er), format='D')
c1 = fits.Column(name='score', array=np.array(sc), format='D')
c2 = fits.Column(name='zsource', array=np.array(zs), format='D')
c3 = fits.Column(name='ra', array=np.array(ra), format='D')
c4 = fits.Column(name='dec', array=np.array(dec), format='D')
c5 = fits.Column(name='z', array=np.array(z), format='D')
c6 = fits.Column(name='plate', array=np.array(plate), format='D')
c7 = fits.Column(name='mjd', array=np.array(mjd), format='D')
c8 = fits.Column(name='fiberid', array=np.array(fid), format='D')
c9 = fits.Column(name='wavelength', array=np.array(w), format='D')
ca = fits.Column(name='snPrevFiber', array=np.array(snp), format='D')
cb = fits.Column(name='snNextFiber', array=np.array(snn), format='D')
cc = fits.Column(name='snSpectra', array=np.array(sns), format='D')
cd = fits.Column(name='o3FoundWave', array=np.array(o3w), format='D')
ce = fits.Column(name='o3FoundSig', array=np.array(o3s), format='D')
# Magnitude and ebv
uCan = []
gCan = []
rCan = []
iCan = []
zCan = []
ebv = []
for p, m, f in zip(*[plate, mjd, fid]):
    sppDir = os.path.join('/SCRATCH', 'BOSS', 'data', 'v5_7_0', str(int(p)),
                          'spPlate-' + str(int(p)) + "-" + str(int(m)) +
                          '.fits')
    sppFile = pf.open(sppDir)
    uCan.append(sppFile[5].data['MAG'][int(f) - 1][0])
    gCan.append(sppFile[5].data['MAG'][int(f) - 1][1])
    rCan.append(sppFile[5].data['MAG'][int(f) - 1][2])
    iCan.append(sppFile[5].data['MAG'][int(f) - 1][3])
    zCan.append(sppFile[5].data['MAG'][int(f) - 1][4])
    ebv.append(sppFile[5].data['SFD_EBV'][int(f) - 1])
    sppFile.close()
d1 = fits.Column(name='magU', array=np.array(uCan), format='D')
d2 = fits.Column(name='magG', array=np.array(gCan), format='D')
d3 = fits.Column(name='magR', array=np.array(rCan), format='D')
d4 = fits.Column(name='magI', array=np.array(iCan), format='D')
d5 = fits.Column(name='magZ', array=np.array(zCan), format='D')
d6 = fits.Column(name='ebv', array=np.array(ebv), format='D')
# Save to file
t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb,
                                   cc, cd, ce, d1, d2, d3, d4, d5, d6])
t.writeto(saveFile, clobber=True)
