#!/usr/bin/env python3

import numpy as np
from numpy.polynomial.polynomial import polyval, polyval3d
from numpy.linalg import norm


# TODO: implement a method to write a backflow file
# TODO: write a generator of monomials of THETA and PHI polynom. decomposition
# TODO: interpolate backflow of order X_Y_ZZ, with polynomial functions.
# TODO: interpolate backflow of order X_Y_ZZ, non-polynomial functions.

class Backflow:
    """Backflow reader from file.
    Inhomogeneous backflow transformations in quantum Monte Carlo.
    P. Lopez Rıos, A. Ma, N. D. Drummond, M. D. Towler, and R. J. Needs
    http://www.tcm.phy.cam.ac.uk/~mdt26/downloads/lopezrios_backflow.pdf
    """

    def __init__(self, file):
        """Init."""
        self.ETA_TERM = None
        self.MU_TERM = []
        self.PHI_TERM = []
        self.THETA_TERM = []
        self.ETA_L = None
        self.MU_L = []
        self.MU_CUSP = []
        self.MU_order = []
        self.MU_spin_dep = []
        self.PHI_L = []
        self.PHI_CUSP = []
        self.PHI_irrotational = []
        self.PHI_ee_order = []
        self.PHI_en_order = []
        self.PHI_spin_dep = []
        self.AE_L = None
        self.read(file)

    def ETA_powers(self):
        """Generates ETA polynomial powers."""
        for i in range(1, self.ETA_spin_dep+1):
            for j in range(self.ETA_order):
                if (j, i) == (1, 1):
                    continue
                yield j, i

    def MU_powers(self):
        """Generates MU polynomial powers."""
        for i in range(1, self.MU_spin_dep+1):
            for j in range(self.MU_order):
                if j < 2:
                    continue
                yield j, i

    def PHI_powers(self):
        """Generates PHI polynomial powers."""
        for i in range(1, self.PHI_spin_dep+1):
            for j in range(self.PHI_ee_order):
                for k in range(self.PHI_en_order):
                    for l in range(self.PHI_en_order):
                        if (l < 2 or k < 2) and j == 0 and i == 1:
                            continue
                        if (l < 1 or l > self.PHI_en_order-2 or k < 2) and j == 1 and i == 1:
                            continue
                        if abs(k-l-1) > self.PHI_en_order-4 and j == 1 and i == 1:
                            continue
                        yield l, k, j, i

    def THETA_powers(self):
        """Generates THETA polynomial powers."""
        for i in range(1, self.PHI_spin_dep+1):
            for j in range(self.PHI_ee_order):
                for k in range(self.PHI_en_order):
                    for l in range(self.PHI_en_order):
                        yield l, k, j, i

    def read(self, file):
        """Open file and read backflow data."""
        with open(file, 'r') as f:
            backflow = False
            AE = False
            ETA = ETA_params = False
            MU = MU_params = MU_set = False
            PHI = PHI_params = PHI_set = False
            line = f.readline()
            eta_powers = self.ETA_powers()
            # mu_powers = self.MU_powers()
            # phi_powers = self.PHI_powers()
            # theta_powers = self.THETA_powers()
            while line:
                line = f.readline()
                if line.strip().startswith('START BACKFLOW'):
                    backflow = True
                elif line.strip().startswith('Truncation order'):
                    self.C = float(f.readline().split()[0])
                elif line.strip().startswith('START ETA TERM'):
                    ETA = True
                elif line.strip().startswith('END ETA TERM'):
                    ETA = False
                elif line.strip().startswith('START MU TERM'):
                    MU = True
                elif line.strip().startswith('END MU TERM'):
                    MU = False
                elif line.strip().startswith('START PHI TERM'):
                    PHI = True
                elif line.strip().startswith('END PHI TERM'):
                    PHI = False
                elif line.strip().startswith('START AE CUTOFFS'):
                    AE = True
                elif line.strip().startswith('END AE CUTOFFS'):
                    AE = False
                elif ETA:
                    if line.strip().startswith('Expansion order'):
                        self.ETA_order = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Spin dep'):
                        self.ETA_spin_dep = int(f.readline().split()[0]) + 1
                        self.ETA_L = np.zeros((self.ETA_spin_dep), 'd')
                    elif line.strip().startswith('Cut-off radii'):
                        line = f.readline().split()
                        self.ETA_L[0] = float(line[0])
                        for i in range(1, self.ETA_spin_dep):
                            # Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                            if line[1] == '2':
                                self.ETA_L[i] = float(line[0])
                            else:
                                self.ETA_L[i] = float(f.readline().split()[0])
                    elif line.strip().startswith('Parameter'):
                        self.ETA_TERM = np.zeros((self.ETA_order, self.ETA_spin_dep), 'd')
                        ETA_params = True
                    elif ETA_params:
                        a, b = map(int, line.split()[3].split('_')[1].split(','))
                        assert (a, b) == next(eta_powers)
                        self.ETA_TERM[a][b-1] = float(line.split()[0])
                elif MU:
                    if line.strip().startswith('Number of sets'):
                        self.MU_sets = int(f.readline().split()[0])
                        set = 0
                    if line.strip().startswith('START SET'):
                        MU_set = True
                    if line.strip().startswith('END SET'):
                        MU_set = MU_params = False
                        set += 1
                    if MU_set:
                        if line.strip().startswith('Type of e-N cusp conditions'):
                            self.MU_CUSP.append(bool(int(f.readline().split()[0])))
                        elif line.strip().startswith('Expansion order'):
                            self.MU_order.append(int(f.readline().split()[0]) + 1)
                        elif line.strip().startswith('Spin dep'):
                            self.MU_spin_dep.append(int(f.readline().split()[0]) + 1)
                        elif line.strip().startswith('Cutoff (a.u.)'):
                            self.MU_L.append(float(f.readline().split()[0]))
                        elif line.strip().startswith('Parameter values'):
                            self.MU_TERM.append(np.zeros((self.MU_order[set], self.MU_spin_dep[set]), 'd'))
                            MU_params = True
                        elif MU_params:
                            a, b = map(int, line.split()[3].split('_')[1].split(','))
                            # assert (a, b) == next(mu_powers)
                            self.MU_TERM[set][a][b-1] = float(line.split()[0])
                elif PHI:
                    if line.strip().startswith('Number of sets'):
                        self.PHI_sets = int(f.readline().split()[0])
                        set = 0
                    if line.strip().startswith('START SET'):
                        PHI_set = True
                    if line.strip().startswith('END SET'):
                        PHI_set = PHI_params = False
                        set += 1
                    if PHI_set:
                        if line.strip().startswith('Type of e-N cusp conditions'):
                            self.PHI_CUSP.append(bool(int(f.readline().split()[0])))
                        elif line.strip().startswith('Irrotational Phi'):
                            self.PHI_irrotational.append(bool(int(f.readline().split()[0])))
                        elif line.strip().startswith('Electron-nucleus expansion order'):
                            self.PHI_en_order.append(int(f.readline().split()[0]) + 1)
                        elif line.strip().startswith('Electron-electron expansion order'):
                            self.PHI_ee_order.append(int(f.readline().split()[0]) + 1)
                        elif line.strip().startswith('Spin dep'):
                            self.PHI_spin_dep.append(int(f.readline().split()[0]) + 1)
                        elif line.strip().startswith('Cutoff (a.u.)'):
                            self.PHI_L.append(float(f.readline().split()[0]))
                        elif line.strip().startswith('Parameter values'):
                            self.PHI_TERM.append(np.zeros((self.PHI_en_order[set], self.PHI_en_order[set], self.PHI_ee_order[set], self.PHI_spin_dep[set]), 'd'))
                            if not self.PHI_irrotational[set]:
                                self.THETA_TERM.append(np.zeros((self.PHI_en_order[set], self.PHI_en_order[set], self.PHI_ee_order[set], self.PHI_spin_dep[set]), 'd'))
                            PHI_params = True
                        elif PHI_params:
                            a, b, c, d = map(int, line.split()[3].split('_')[1].split(','))
                            if line.split()[3].split('_')[0] == 'phi':
                                # print((a, b, c, d), next(phi_powers))
                                self.PHI_TERM[set][a][b][c][d-1] = float(line.split()[0])
                            elif line.split()[3].split('_')[0] == 'theta':
                                # print((a, b, c, d), next(theta_powers))
                                self.THETA_TERM[set][a][b][c][d-1] = float(line.split()[0])
                elif AE:
                    if line.strip().startswith('Nucleus'):
                        self.AE_L = float(f.readline().split()[2])
            if not backflow:
                print('No BACKLOW section found')
                exit(0)

    def cutoff(self, r, L):
        """General cutoff"""
        return (1 - r/L)**self.C * np.heaviside(L-r, 0.0)

    def AE_cutoff(self, r, L):
        """All electron atom cutoff"""
        return np.where(r < L, (r/L)**2 * (6 - 8 * (r/L) + 3 * (r/L)**2), 1.0)

    def ETA(self, ri, rj, rI_list, ri_spin, rj_spin):
        """ETA term.
        Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param ri_spin: 0 - up, 1 - down
        :param rj_spin: 0 - up, 1 - down
        :return:
        """
        rij = norm(ri - rj, axis=0)
        if self.ETA_spin_dep == 1:
            channel = 0
        elif self.ETA_spin_dep == 2:
            if ri_spin == rj_spin:
                channel = 0
            else:
                channel = 1
        elif self.ETA_spin_dep == 3:
            if ri_spin == rj_spin == 0:
                channel = 0
            elif ri_spin == rj_spin == 1:
                channel = 1
            else:
                channel = 2
        result = self.cutoff(rij, self.ETA_L[channel]) * polyval(rij, self.ETA_TERM[:, channel])
        for rI in rI_list:
            riI = norm(ri - rI, axis=0)
            result *= self.AE_cutoff(riI, self.AE_L)
        return result * (rj - ri)

    def MU(self, ri, rI_list, ri_spin, set):
        """MU term
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param ri_spin: 0 - up, 1 - down
        :param set: int
        :return:
        """
        result = 0
        for k, rI in enumerate(rI_list):
            riI = norm(ri - rI, axis=0)
            channel = min(self.MU_spin_dep[set]-1, ri_spin)
            mu = self.cutoff(riI, self.MU_L[set]) * polyval(riI, self.MU_TERM[set][:, channel])
            for l, rI_other in enumerate(rI_list):
                if k != l:
                    riI_other = norm(ri - rI_other, axis=0)
                    mu *= self.AE_cutoff(riI_other, self.AE_L)
            result += mu * (rI - ri)
        return result

    def PHI(self, ri, rj, rI_list, ri_spin, rj_spin, set):
        """PHI term
        Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param ri_spin: 0 - up, 1 - down
        :param rj_spin: 0 - up, 1 - down
        :param set: int
        :return:
        """
        result = 1
        for rI in rI_list:
            if self.PHI_spin_dep[set] == 1:
                channel = 0
            elif self.PHI_spin_dep[set] == 2:
                if ri_spin == rj_spin:
                    channel = 0
                else:
                    channel = 1
            elif self.PHI_spin_dep[set] == 3:
                if ri_spin == rj_spin == 0:
                    channel = 0
                elif ri_spin == rj_spin == 1:
                    channel = 1
                else:
                    channel = 2
            rij = norm(ri - rj, axis=0)
            riI = norm(ri - rI, axis=0)
            rjI = norm(rj - rI, axis=0)
            result *= self.cutoff(riI, self.PHI_L[set]) * self.cutoff(rjI, self.PHI_L[set])
            result *= polyval3d(riI, rjI, rij, self.PHI_TERM[set][:, :, :, channel])
            if self.PHI_CUSP[set]:
                result *= self.AE_cutoff(riI, self.AE_L)
        return result * (rj - ri)

    def THETA(self, ri, rj, rI_list, ri_spin, rj_spin, set):
        """THETA term
        Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param ri_spin: 0 - up, 1 - down
        :param rj_spin: 0 - up, 1 - down
        :param set: 0, 1, 2
        :return:
        """
        result = 0
        for k, rI in enumerate(rI_list):
            if self.PHI_spin_dep[set] == 1:
                channel = 0
            elif self.PHI_spin_dep[set] == 2:
                if ri_spin == rj_spin:
                    channel = 0
                else:
                    channel = 1
            elif self.PHI_spin_dep[set] == 3:
                if ri_spin == rj_spin == 0:
                    channel = 0
                elif ri_spin == rj_spin == 1:
                    channel = 1
                else:
                    channel = 2
            rij = norm(ri - rj, axis=0)
            riI = norm(ri - rI, axis=0)
            rjI = norm(rj - rI, axis=0)
            theta = self.cutoff(riI, self.PHI_L[set]) * self.cutoff(rjI, self.PHI_L[set])
            theta *= polyval3d(riI, rjI, rij, self.THETA_TERM[set][:, :, :, channel])
            for l, rI_other in enumerate(rI_list):
                if k != l:
                    riI_other = norm(ri - rI_other, axis=0)
                    theta *= self.AE_cutoff(riI_other, self.AE_L)
            result += theta * (rI - ri)
        return result

    def ALL(self, ri, rj, rI_list, ri_spin, rj_spin, set):
        """Total displacement"""
        result = 0
        if self.ETA_TERM is not None:
            result += self.ETA(ri, rj, rI_list, ri_spin, rj_spin)
        if self.MU_TERM:
            result += self.MU(ri, rI_list, ri_spin, set)
        if self.PHI_TERM is not None:
            result += self.PHI(ri, rj, rI_list, ri_spin, rj_spin, set)
            if not self.PHI_irrotational[set]:
                result += self.THETA(ri, rj, rI_list, ri_spin, rj_spin, set)
        return result

    def Be(self, r1, r2, r3, r4):
        """Be backflow"""
        r1 = r1 + self.ALL(r1, r2, [0, 0, 0], 0, 0, 0) + self.ALL(r1, r3, [0, 0, 0], 1, 0, 0) + self.ALL(r1, r4, [0, 0, 0], 1, 0, 0)
        r2 = r2 + self.ALL(r2, r1, [0, 0, 0], 0, 0, 0) + self.ALL(r2, r3, [0, 0, 0], 1, 0, 0) + self.ALL(r2, r4, [0, 0, 0], 1, 0, 0)
        r3 = r3 + self.ALL(r3, r4, [0, 0, 0], 0, 0, 0) + self.ALL(r3, r1, [0, 0, 0], 0, 0, 0) + self.ALL(r3, r2, [0, 0, 0], 0, 0, 0)
        r4 = r4 + self.ALL(r4, r3, [0, 0, 0], 0, 0, 0) + self.ALL(r4, r1, [0, 0, 0], 0, 0, 0) + self.ALL(r4, r2, [0, 0, 0], 0, 0, 0)
        return r1, r2, r3, r4
