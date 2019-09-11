#!/usr/bin/env python3

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.linalg import norm, det
from plot_implicit import plot_implicit
import numpy as np
from gwfn import Gwfn


Be_gwfn = Gwfn('gwfn.data')


def gaussian_1s(r, spin):
    return Be_gwfn.wfn(r, 0, spin)


def gaussian_2s(r, spin):
    return Be_gwfn.wfn(r, 1, spin)


def gaussian_2px(r, spin):
    return Be_gwfn.wfn(r, 2, spin)


def gaussian_2py(r, spin):
    return Be_gwfn.wfn(r, 3, spin)


def gaussian_2pz(r, spin):
    return Be_gwfn.wfn(r, 4, spin)


def Be_1s2s(r1, r2, spin):
    return gaussian_1s(r1, spin) * gaussian_2s(r2, spin) - gaussian_1s(r2, spin) * gaussian_2s(r1, spin)


def Be_1s2px(r1, r2, spin):
    return gaussian_1s(r1, spin) * gaussian_2px(r2, spin) - gaussian_1s(r2, spin) * gaussian_2px(r1, spin)


def Be_1s2py(r1, r2, spin):
    return gaussian_1s(r1, spin) * gaussian_2py(r2, spin) - gaussian_1s(r2, spin) * gaussian_2py(r1, spin)


def Be_1s2pz(r1, r2, spin):
    return gaussian_1s(r1, spin) * gaussian_2pz(r2, spin) - gaussian_1s(r2, spin) * gaussian_2pz(r1, spin)


def Be(r12_minus, r34_minus, r12_plus):
    u"""HF = |1s(r1)2s(r2)| * |1s(r3)2s(r4)|"""
    # r12_plus = r12_plus + 3.0
    r34_plus = np.full((100, 100), 6.0)
    vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    phi = np.pi/4.0
    theta = np.pi/2.0
    vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])[:, np.newaxis, np.newaxis]
    r1n = (r12_plus + r12_minus)/2.0
    r2n = (r12_plus - r12_minus)/2.0
    r1 = vec_1 * r1n[np.newaxis]
    r2 = vec_2 * r2n[np.newaxis]
    r3n = (r34_plus + r34_minus)/2.0
    r4n = (r34_plus - r34_minus)/2.0
    r3 = vec_3 * r3n[np.newaxis]
    r4 = vec_4 * r4n[np.newaxis]
    return Be_1s2s(r1, r2, spin='up') * Be_1s2s(r3, r4, spin='down')


def Be_4det(r12_minus, r34_minus, r12_plus):
    u"""CI = phi(1s2, 2s2) + C * phi(1s2, 2p2)"""
    # r12_plus = r12_plus + 3.0
    r34_plus = np.full((100, 100), 1.0)
    vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    phi = np.pi/4.0
    theta = np.pi/2.0
    vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])[:, np.newaxis, np.newaxis]
    C = -0.18
    r1n = (r12_plus + r12_minus)/2.0
    r2n = (r12_plus - r12_minus)/2.0
    r1 = vec_1 * r1n[np.newaxis]
    r2 = vec_2 * r2n[np.newaxis]
    r3n = (r34_plus + r34_minus)/2.0
    r4n = (r34_plus - r34_minus)/2.0
    r3 = vec_3 * r3n[np.newaxis]
    r4 = vec_4 * r4n[np.newaxis]
    return Be_1s2s(r1, r2, spin='up') * Be_1s2s(r3, r4, spin='down') + \
           C * Be_1s2pz(r1, r2, spin='up') * Be_1s2pz(r3, r4, spin='down') + \
           C * Be_1s2py(r1, r2, spin='up') * Be_1s2py(r3, r4, spin='down') + \
           C * Be_1s2px(r1, r2, spin='up') * Be_1s2px(r3, r4, spin='down')


if __name__ == '__main__':
    plot_implicit(Be)
    #plot_implicit(Be_4det)
