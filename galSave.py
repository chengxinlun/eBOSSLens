import os
import pickle
import itertools as it
import numpy as np
from scipy.optimize import minimize
from utils import make_sure_path_exists, gauss, chi2T, gauss3
from SDSSObject import SDSSObject
# Matplotlib trick
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def galSave(doublet, obj, peak_candidates, doublet_index, savedir, em_lines,
            doPlot, prodCrit):
    detection = False
    preProd = 0.0
    nxtProd = 0.0
    if doublet:
        preProd, nxtProd = fitcSpec(obj, peak_candidates[doublet_index])
        if preProd + nxtProd > prodCrit:
            raise Exception("Rejected by comparing to other fibers")
        z_s = peak_candidates[doublet_index].wavelength / 3727.09 - 1.0
        # Find peak near infered OIII and Hbeta by fitting
        fitChi, fitRes = _findPeak(obj, z_s, width=20.0)
        print(fitChi, fitRes)
        detection = _doubletSave(obj, z_s, peak_candidates, doublet_index,
                                 savedir, preProd, nxtProd, fitChi, fitRes)
        detection = _dblmultSave(obj, z_s, peak_candidates, savedir,
                                 detection, em_lines)
    elif len(peak_candidates) > 1:
        detection = _multletSave(obj, peak_candidates, savedir, em_lines)
    if not detection:
        raise Exception("Rejected since source too near")
    peaks = []
    for k in range(len(peak_candidates)):
        peak = peak_candidates[k]
        if k == doublet_index and doublet:
            peaks.append([peak.wavDoublet[0], peak.ampDoublet[0],
                          peak.varDoublet])
            peaks.append([peak.wavDoublet[1], peak.ampDoublet[1],
                          peak.varDoublet])
        else:
            peaks.append([peak.wavSinglet, peak.ampSinglet, peak.varSinglet])
    peak_number = len(peak_candidates)
    if (peak_number > 1 or doublet) and detection:
        if doPlot:
            fit = 0.0
            for k in np.arange(len(peaks)):
                fit = fit + gauss(obj.wave, x_0=peaks[k][0], A=peaks[k][1],
                                  var=peaks[k][2])
            o3hbflux = gauss3(obj.wave, fitRes, 4862.68 * (1.0 + z_s),
                              4960.30 * (1.0 + z_s), 5008.24 * (1.0 + z_s))
            o3b = [4842.68 * (1.0 + z_s), 5028.24 * (1.0 + z_s)]
            o3hbwave = [4862.68 * (1.0 + z_s) - fitRes[0],
                        5008.24 * (1.0 + z_s) - fitRes[3]]
            plotGalaxyLens(doublet, obj, savedir, peak_candidates, preProd,
                           nxtProd, doublet_index, fit, o3hbflux, fitChi, o3b,
                           o3hbwave)
        if doublet:
            x_doublet = np.mean(peak_candidates[doublet_index].wavDoublet)
            bd = np.linspace(obj.wave2bin(x_doublet) - 10,
                             obj.wave2bin(x_doublet) + 10, 21, dtype=np.int16)
            galSaveflux(obj.reduced_flux[bd], obj.fiberid, savedir)


def _findPeak(obj, zsource, width=20.0):
    hb = 4862.68 * (1.0 + zsource)
    o31 = 4960.30 * (1.0 + zsource)
    o32 = 5008.24 * (1.0 + zsource)
    tmp = np.linspace(obj.wave2bin(hb - width * (1.0 + zsource)),
                      obj.wave2bin(o32 + width * (1.0 + zsource)),
                      dtype=np.int16)
    if len(tmp) < 10:
        return 1000.0, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    bounds = [(-10.0, 10.0), (0.0, 5.0), (1.0, 8.0), (-10.0, 10.0), (0.0, 5.0),
              (1.0, 8.0), (0.0, 5.0)]
    pWave = obj.wave[tmp]
    pFlux = obj.reduced_flux[tmp]
    pIvar = obj.ivar[tmp]
    res = minimize(chi2T, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0], args=(pWave,
                                                                     pFlux,
                                                                     pIvar, hb,
                                                                     o31, o32),
                   method='SLSQP', bounds=bounds)
    if res.fun == 0.0:
        return 1000.0, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    return res.fun, res.x


def _doubletSave(obj, z_s, peak_candidates, doublet_index, savedir, pP, nP,
                 fitChi, fitRes):
    # List to string
    text = ""
    for each in fitRes:
        text = text + " " + str(each)
    score = 0.0
    detection = False
    fileD = open(os.path.join(savedir, 'candidates_doublet.txt'), 'a')
    if z_s > obj.z + 0.05:
        detection = True
        score += peak_candidates[doublet_index].chi
        fileD.write(str(obj.radEinstein(z_s)) + " " + str(score) +
                    " " + str(z_s) + " " + str(obj.RA) + " " +
                    str(obj.DEC) + " " + str(obj.z) + " " + str(obj.plate) +
                    " " + str(obj.mjd) + " " + str(obj.fiberid) + " " +
                    str(peak_candidates[doublet_index].wavDoublet[0]) + " " +
                    str(pP) + " " + str(nP) + " " + str(obj.snSpectra) + " " +
                    str(fitChi) + text + "\n")
    fileD.close()
    return detection


def _dblmultSave(obj, z_s, peak_candidates, savedir, detection, em_lines):
    confirmed_lines = []
    score = 0.0
    det = detection
    fileM = open(os.path.join(savedir, 'candidates_DM.txt'), 'a')
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
    fileM = open(os.path.join(savedir, 'candidates_multi.txt'), 'a')
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


def galSaveflux(fList, fid, savedir):
    fileDir = os.path.join(savedir, "doublet_ML")
    make_sure_path_exists(fileDir)
    fileDir = os.path.join(fileDir, str(fid) + ".pkl")
    f = open(fileDir, "wb")
    pickle.dump(fList, f)
    f.close()


def plotGalaxyLens(doublet, obj, savedir, peak_candidates, preProd, nxtProd,
                   doublet_index, fit, o3hb, fitChi, o3b, o3hbw):
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
        if fitChi != 1000.0:
            fs = (22.5, 10)
            gd = (2, 9)
            mcol = 4
            o2col = 2
            o2p = (0, 4)
        else:
            fs = (15, 5)
            gd = (1, 3)
            mcol = 2
            o2col = 1
            o2p = (0, 2)
        # Plot currently inspecting spectra
        plt.figure(figsize=fs)
        plt.suptitle('RA=' + str(obj.RA) + ', Dec=' + str(obj.DEC) +
                     ', Plate=' + str(obj.plate) + ', Fiber='+str(obj.fiberid) +
                     ', MJD=' + str(obj.mjd) + '\n$z=' + str(obj.z) + ' \pm' +
                     str(obj.z_err) + '$, Class=' + str(obj.obj_class))
        # Reduced flux overall
        ax1 = plt.subplot2grid(gd, (0, 0), colspan=mcol)
        ax1.plot(obj.wave[10:-10], obj.reduced_flux[10:-10], 'k')
        ax1.plot(obj.wave, fit, 'r')
        ax1.set_xlabel('$\lambda \, [\AA]$ ')
        ax1.set_ylabel(
            '$f_{\lambda}\, (10^{-17} erg\, s^{-1} cm^{-2} Ang^{-1}$')
        ax1.set_xlim([np.min(obj.wave), np.max(obj.wave)])
        # Reduced flux detail
        ax2 = plt.subplot2grid(gd, o2p, colspan=o2col)
        ax2.set_xlabel('$\lambda \, [\AA]$ ')
        ax2.locator_params(tight=True)
        ax2.set_xlim([peak_candidates[doublet_index].wavelength - 30.0,
                      peak_candidates[doublet_index].wavelength + 30.0])
        ax2.plot(obj.wave, obj.reduced_flux, 'k')
        ax2.plot(obj.wave, fit, 'r')
        ax2.set_ylim([-5, 10])
        ax2.vlines(x=obj.zline['linewave'] * (1.0 + obj.z), ymin=-10, ymax=10,
                   colors='g', linestyles='dashed')
        if fitChi != 1000.0:
            ax3 = plt.subplot2grid(gd, (0, 6), colspan=3)
            ax3.set_xlabel('$\lambda \, [\AA]$ ')
            ax3.locator_params(tight=True)
            ax3.set_xlim(o3b)
            ax3.plot(obj.wave, obj.reduced_flux, 'k')
            ax3.plot(obj.wave, o3hb, 'r')
            ax3.set_ylim([-5, 10])
            ax3.vlines(x=o3hbw[0], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
            ax3.vlines(x=o3hbw[1], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
            ax4 = plt.subplot2grid(gd, (1, 0), colspan=4)
            ax4.plot(obj.wave[10:-10], obj.ivar[10: -10])
            ax4.vlines(x=o3hbw[0], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
            ax4.vlines(x=o3hbw[1], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
            ax5 = plt.subplot2grid(gd, (1, 4), colspan=o2col)
            ax5.set_xlabel('$\lambda \, [\AA]$ ')
            ax5.locator_params(tight=True)
            ax5.set_xlim([peak_candidates[doublet_index].wavelength - 30.0,
                          peak_candidates[doublet_index].wavelength + 30.0])
            ax5.plot(obj.wave, obj.ivar, 'k')
            ax5.set_ylim([-5, 10])
            ax5.vlines(x=obj.zline['linewave'] * (1.0 + obj.z), ymin=-10, ymax=10,
                       colors='g', linestyles='dashed')
            ax6 = plt.subplot2grid(gd, (1, 6), colspan=3)
            ax6.set_xlabel('$\lambda \, [\AA]$ ')
            ax6.locator_params(tight=True)
            ax6.set_xlim(o3b)
            ax6.plot(obj.wave, obj.ivar, 'k')
            ax6.vlines(x=o3hbw[0], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
            ax6.vlines(x=o3hbw[1], ymin=-5, ymax=10, colors='g',
                       linestyles='dashed')
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
        make_sure_path_exists(os.path.join(savedir, 'plots'))
        plt.savefig(os.path.join(savedir, 'plots', str(obj.plate) + '-' +
                                 str(obj.mjd) + '-' + str(obj.fiberid) +
                                 '.png'))
        plt.close()


def fitcSpec(obj, peak, width=2.0):
    initP = [peak.ampDoublet[0], peak.varDoublet, peak.ampDoublet[1],
             peak.wavDoublet[0], peak.wavDoublet[1]]
    limP = [(0.0, 5.0), (1.0, 8.0), (0.0, 5.0),
            (peak.wavDoublet[0] - width * np.sqrt(peak.varDoublet),
             peak.wavDoublet[0] + width * np.sqrt(peak.varDoublet)),
            (peak.wavDoublet[1] - width * np.sqrt(peak.varDoublet),
             peak.wavDoublet[1] + width * np.sqrt(peak.varDoublet))]
    bounds = np.arange(obj.wave2bin(peak.wavDoublet.min()) - 15,
                       obj.wave2bin(peak.wavDoublet.max()) + 15, 1.0,
                       dtype=int)
    if obj.fiberid != 1 and (len(obj.ivarPre[bounds].nonzero()[0]) > 6):
        resp, preChi2 = obj.doubletFit(bounds, initP, limP, "pre")
        preAmp = (resp[0] + resp[2]) * np.sqrt(resp[1]) / \
            (np.sum(peak.ampDoublet) * np.sqrt(peak.varDoublet))
    else:
        preAmp = 0.0
    if obj.fiberid != 1000 and (len(obj.ivarNxt[bounds].nonzero()[0]) > 6):
        resn, nxtChi2 = obj.doubletFit(bounds, initP, limP, "nxt")
        nxtAmp = (resn[0] + resn[2]) * np.sqrt(resn[1]) / \
            (np.sum(peak.ampDoublet) * np.sqrt(peak.varDoublet))
    else:
        nxtAmp = 0.0
    return [preAmp, nxtAmp]
