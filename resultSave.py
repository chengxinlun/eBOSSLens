import itertools as it
import numpy as np
from utils import make_sure_path_exists
from SDSSObject import SDSSObject
# Matplotlib trick
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def galSave(doublet, obj, peak_candidates, doublet_index, savedir, em_lines):
    detection = False
    if doublet:
        z_s = peak_candidates[doublet_index].wavelength / 3727.24 - 1.0
        detection = _doubletSave(obj, z_s, peak_candidates, doublet_index,
                                 savedir)
        if len(peak_candidates):
            detection = _dblmultSave(obj, z_s, peak_candidates, savedir,
                                     detection, em_lines)
    elif len(peak_candidates) > 1:
        detection = _multletSave(obj, peak_candidates, savedir, em_lines)
    return detection


def _doubletSave(obj, z_s, peak_candidates, doublet_index, savedir):
    score = 0.0
    detection = False
    fileD = open(savedir + '/candidates_doublet.txt', 'a')
    if z_s > obj.z + 0.05:
        detection = True
        score += peak_candidates[doublet_index].chi
        fileD.write(str(obj.radEinstein(z_s)) + " " + str(score) +
                    " " + str(z_s) + " " + str(obj.RA) + " " +
                    str(obj.DEC) + " " + str(obj.plate) + " " +
                    str(obj.mjd) + " " + str(obj.fiberid) + " " +
                    str(peak_candidates[doublet_index].wavDoublet[0]) +
                    "\n")
    fileD.close()
    return detection


def _dblmultSave(obj, z_s, peak_candidates, savedir, detection, em_lines):
    confirmed_lines = []
    score = 0.0
    det = detection
    fileM = open(savedir + '/candidates_DM.txt', 'a')
    # Generating all combinations of lines from above list to compare with
    # candidates
    temp = [peak for peak in peak_candidates if peak.chi != peak.chiDoublet]
    compare = em_lines[1: 5]
    if z_s > obj.z + 0.05:
        for peak in temp:
            for line in compare:
                if abs(peak.wavelength/line - 1.0 - z_s) < 0.01:
                    det = True
                    confirmed_lines.append(line)
                    score += peak.chiDoublet
    if confirmed_lines != []:
        fileM.write(str(obj.radEinstein(z_s)) + " " + str(score) + " " +
                    str(z_s) + " " + str(obj.RA) + " " + str(obj.DEC) +
                    " " + str(obj.plate) + " " + str(obj.mjd) + " " +
                    str(obj.fiberid) + " " + str(confirmed_lines) + "\n")
    fileM.close()
    return det


def _multletSave(obj, peak_candidates, savedir, em_lines):
    confirmed_lines = []
    score = 0.0
    detection = False
    compare = it.combinations(em_lines, len(peak_candidates))
    fileM = open(savedir + '/candidates_multi.txt', 'a')
    for group in compare:
        for k in range(len(peak_candidates)):
            for j in range(k + 1, len(peak_candidates)):
                crit1 = peak_candidates[k].wavelength / group[k]
                crit2 = crit1 - peak_candidates[j].wavelength / group[j]
                if abs(crit2) < 0.01 and crit1 - 1.05 > obj.z:
                    detection = True
                    z_s = peak_candidates[k].wavelength / group[k] - 1.0
                    confirmed_lines.append([group,
                                            peak_candidates[k].wavelength /
                                            group[k] - 1.0])
                    score += peak_candidates[j].chi ** 2.0 + \
                        peak_candidates[k].chi ** 2.0
    if confirmed_lines != []:
        fileM.write(str(obj.radEinstein(z_s)) + " " + str(score) + " " +
                    str(z_s) + " " + str(obj.RA) + " " + str(obj.DEC) +
                    " " + str(obj.plate) + " " + str(obj.mjd) + " " +
                    str(obj.fiberid) + " " + str(confirmed_lines) + "\n")
    fileM.close()
    return detection


def plotGalaxyLens(doublet, obj, savedir, peak_candidates, doublet_index, fit):
    if not doublet:
        ax = plt.subplot(1, 1, 1)
        plt.title('RA=' + str(obj.RA) + ', Dec=' + str(obj.DEC) + ', Plate=' +
                  str(obj.plate) + ', Fiber=' + str(obj.fiberid) +
                  ', MJD=' + str(obj.mjd) + '\n$z=' + str(obj.z) + ' \pm' +
                  str(obj.z_err) + '$, Class=' + str(obj.obj_class))
        ax.plot(obj.wave, obj.reduced_flux, 'k')
        plt.xlabel('$Wavelength\, (Angstroms)$')
        plt.ylabel('$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2} Ang^{-1}$')
        ax.plot(obj.wave, fit, 'r')
        make_sure_path_exists(savedir + '/plots/')
        plt.savefig(savedir + '/plots/' + str(obj.plate) + '-' + str(obj.mjd) +
                    '-' + str(obj.fiberid) + '.png')
        plt.close()
    # If doublet, plot in two different windows
    else:
        # Plot currently inspecting spectra
        x_doublet = np.mean(peak_candidates[doublet_index].wavDoublet)
        bounds = np.linspace(obj.wave2bin(x_doublet) - 10,
                             obj.wave2bin(x_doublet) + 10, 21, dtype=np.int16)
        f = open(savedir + '/doublet_ML.txt', 'a')
        f.write(str(obj.plate) + ' ' + str(obj.mjd) + ' ' + str(obj.fiberid) +
                " " + str(obj.reduced_flux[bounds]))
        f.close()
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        plt.suptitle('RA=' + str(obj.RA) + ', Dec=' + str(obj.DEC) +
                     ', Plate=' + str(obj.plate) + ', Fiber='+str(obj.fiberid) +
                     ', MJD=' + str(obj.mjd) + '\n$z=' + str(obj.z) + ' \pm' +
                     str(obj.z_err) + '$, Class=' + str(obj.obj_class))
        ax2 = plt.subplot2grid((1, 3), (0, 2))
        ax1.plot(obj.wave[10:-10], obj.reduced_flux[10:-10], 'k')
        ax1.plot(obj.wave, fit, 'r')
        ax1.set_xlabel('$\lambda \, [\AA]$ ')
        ax1.set_ylabel(
            '$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2} Ang^{-1}$')
        ax2.set_xlabel('$\lambda \, [\AA]$ ')
        ax2.locator_params(tight=True)
        ax2.set_xlim([peak_candidates[doublet_index].wavelength - 30.0,
                      peak_candidates[doublet_index].wavelength + 30.0])
        ax2.plot(obj.wave, obj.reduced_flux, 'k')
        ax2.plot(obj.wave, fit, 'r')
        ax2.set_ylim([-5, 10])
        ax2.vlines(x=obj.zline['linewave'] * (1.0 + obj.z), ymin=-10, ymax=10,
                   colors='g', linestyles='dashed')
        ax1.set_xlim([np.min(obj.wave), np.max(obj.wave)])
        # Plot previous one
        if obj.fiberid != 1:
            objPre = SDSSObject(obj.plate, obj.mjd, obj.fiberid - 1,
                                obj.dataVersion, obj.baseDir)
            ax2.plot(objPre.wave, objPre.reduced_flux, 'b')
        # Plot next one
        if obj.fiberid != 1000:
            objNxt = SDSSObject(obj.plate, obj.mjd, obj.fiberid + 1,
                                obj.dataVersion, obj.baseDir)
            ax2.plot(objNxt.wave, objNxt.reduced_flux, 'g')
        # Save to file
        make_sure_path_exists(savedir + '/plots/')
        plt.savefig(savedir + '/plots/' + str(obj.plate) + '-' + str(obj.mjd) +
                    '-' + str(obj.fiberid) + '.png')
        plt.close()


def plot_Jackpot(obj, peak, savedir, counter):
    # TODO: complete the function
    '''
    em_lines = np.array([3726.5,4861.325,4958.911,5006.843,6562.801])
    if show==False:
        mpl.use('Agg')
    fontP = FontProperties()
    fontP.set_size('medium')
    plt.suptitle(SDSSname(RA,DEC)+'\n'+'RA='+str(RA)+', Dec='+str(DEC) +', $z_{QSO}='+'{:03.3}'.format(z)+ '$')

    gs = gridspec.GridSpec(1,4)
    p1 = plt.subplot(gs[0,:4])

    smoothed_flux = np.array([np.mean(flux[ii-2:ii+3]) for ii in range(len(flux)) if (ii>4 and ii<len(flux)-4)])

    p1.plot(wave[5:-4], smoothed_flux, 'k', label = 'BOSS Flux', drawstyle='steps-mid')
    #p1.plot(wave,  flux, 'k', label = 'BOSS Flux')
    p1.plot(wave, synflux, 'r', label = 'PCA fit')
    box = p1.get_position()
    p1.set_position([box.x0,box.y0+0.02,box.width*0.9,box.height])
    p1.set_ylim(np.min(synflux)-3, np.max(synflux)+3)
    p1.vlines(x = em_lines*(1+peak[2]),ymin= -100,ymax= 100,colors= 'g',linestyles='dashed')
    p1.vlines(x = em_lines*(1+peak[3]),ymin= -100,ymax= 100,colors= 'b',linestyles='dashed')

    p1.legend(loc='upper right', bbox_to_anchor = (1.2,1), ncol = 1, prop=fontP)
    p1.set_xlim(3500,10500)
    plt.ylabel('Flux [$10^{-17} erg\, s^{-1} cm^{-2}  \AA^{-1}]$')


    make_sure_path_exists(topdir + savedir +'/plots/')
    #plt.show()
    plt.savefig(topdir + savedir +'/plots/'+SDSSname(RA,DEC)+ '-' + str(plate) + '-' + str(mjd) + '-' + str(fiberid) + '-'+str(counter) +'.png')
    plt.close()
    '''
