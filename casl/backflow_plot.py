#!/usr/bin/env python3

import argparse
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval3d
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D


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

    def __init__(self):
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
            mu_powers = self.MU_powers()
            phi_powers = self.PHI_powers()
            theta_powers = self.THETA_powers()
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

    def ETA(self, ri, rj, rI_list, channel):
        """ETA term.
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param channel: u-u, u-d
        :return:
        """
        rij = norm(ri - rj, axis=0)
        channel = min(self.ETA_spin_dep-1, channel)
        result = (self.cutoff(rij, self.ETA_L[channel]) *
                  polyval(rij, self.ETA_TERM[:, channel]))
        for rI in rI_list:
            riI = norm(ri - rI, axis=0)
            result *= self.AE_cutoff(riI, self.AE_L)
        return result * (rj - ri)

    def MU(self, ri, rI_list, channel, set):
        """MU term
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param channel: u, d
        :param set: 0, 1, 2
        :return:
        """
        result = 0
        for k, rI in enumerate(rI_list):
            riI = norm(ri - rI, axis=0)
            channel = min(self.MU_spin_dep[set]-1, channel)
            mu = self.cutoff(riI, self.MU_L[set])
            mu *= polyval(riI, self.MU_TERM[set][:, channel])
            for l, rI_other in enumerate(rI_list):
                if k != l:
                    riI_other = norm(ri - rI_other, axis=0)
                    mu *= self.AE_cutoff(riI_other, self.AE_L)
            result += mu * (rI - ri)
        return result

    def PHI(self, ri, rj, rI_list, channel, set):
        """PHI term
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param channel: u, d
        :param set: 0, 1, 2
        :return:
        """
        result = 1
        for rI in rI_list:
            rij = norm(ri - rj, axis=0)
            riI = norm(ri - rI, axis=0)
            rjI = norm(rj - rI, axis=0)
            result *= self.cutoff(riI, self.PHI_L[set]) * self.cutoff(rjI, self.PHI_L[set])
            result *= polyval3d(riI, rjI, rij, self.PHI_TERM[set][:, :, :, channel])
            if self.PHI_CUSP[set]:
                result *= self.AE_cutoff(riI, self.AE_L)
        return result * (rj - ri)

    def THETA(self, ri, rj, rI_list, channel, set):
        """THETA term
        :param ri: shape([1,2,3], n, m, l, ...)
        :param rj: shape([1,2,3], n, m, l, ...)
        :param rI_list: list of shape([1,2,3], n, m, l, ...)
        :param channel: u, d
                :param set: 0, 1, 2
        :return:
        """
        result = 0
        for k, rI in enumerate(rI_list):
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

    def ALL(self, ri, rj, rI_list, channel, set):
        """Total displacement"""
        result = 0
        if self.ETA_TERM is not None:
            result += self.ETA(ri, rj, rI_list, channel)
        if self.MU_TERM:
            result += self.MU(ri, rI_list, channel, set)
        if self.PHI_TERM is not None:
            result += self.PHI(ri, rj, rI_list, channel, set)
            if not self.PHI_irrotational[set]:
                result += self.THETA(ri, rj, rI_list, channel, set)
        return result


class Plot1D(Backflow):
    """Plot along the line."""

    def __init__(self, term, file):
        """Initialize plotter.
        :param term: term to plot (ETA, MU)
        :param file: backflow data file.
        """
        super().__init__()
        self.term = term
        self.read(file)
        self.x_min = 0.0
        self.x_max = 10.0
        self.x_steps = 101
        self.xy_elec = [0.0]
        self.xy_nucl = [0.0]

    def grid(self):
        """Electron positions (grid).
        :param indexing: cartesian or matrix indexing of output
        :return:
        """
        return np.linspace(self.x_min, self.x_max, self.x_steps)

    def backflow(self, grid, channel, set):
        """Backflow.
        :param grid: electron positions
        :param channel: [u] or [d]
        :return:
        """
        xy_elec = np.array(self.xy_elec)[:, np.newaxis]
        xy_nucl = np.array(self.xy_nucl)[:, np.newaxis]
        if self.term == 'ETA':
            return self.ETA(grid, xy_elec, [], channel)
        elif self.term == 'MU':
            return self.MU(grid, [xy_nucl], channel, set)

    def plot(self):
        """
        electron is at (0,0,0) for ETA term
        nucleus is at (0,0,0) for MU term
        """
        if self.term == 'ETA':
            for channel in range(self.ETA_spin_dep):
                plt.plot(self.grid(),
                         self.backflow(self.grid(), channel)[0], label=['u-u', 'u-d'][channel])
        elif self.term == 'MU':
            for set in range(self.MU_sets):
                for channel in range(self.MU_spin_dep[set]):
                    label = 'set: {} channel: {}'.format(set, ['u', 'd'][channel])
                    plt.plot(self.grid(), self.backflow(self.grid(), channel, set)[0], label=label)
        plt.xlabel('r (au)')
        plt.ylabel('displacement (au)')
        plt.title('{} backflow term'.format(self.term))
        plt.grid(True)
        plt.legend()
        plt.show()


class Plot2D(Backflow):
    """Plot along the 2D grid."""

    def __init__(self, term, file):
        """Initialize plotter.
        :param term: term to plot (PHI, THETA, ALL)
        :param file: backflow data file.
        """
        super().__init__()
        self.plot_type = 0
        self.plot_3d_type = 0
        self.channel_3D = 0
        self.set = 0
        self.plot_cutoff = False
        self.term = term
        self.read(file)
        self.xy_elec = [0.0, 0.0]
        self.xy_nucl = [0.0, 0.0]

    @property
    def max_L(self):
        """max cutoff"""
        return max(np.max(self.ETA_L) or 0, self.MU_L[self.set] or 0, self.PHI_L[self.set] or 0)

    def grid(self, indexing='xy'):
        """First electron positions (grid).
        :param indexing: cartesian or matrix indexing of output
        :return:
        """
        self.x_max = self.y_max = self.max_L
        self.x_min = self.y_min = -self.max_L
        if self.plot_type in (0, 1):
            self.x_steps = 25
            self.y_steps = 25
            x = np.linspace(self.x_min, self.x_max, self.x_steps)
            y = np.linspace(self.y_min, self.y_max, self.y_steps)
            return np.meshgrid(x, y, indexing=indexing)
        elif self.plot_type == 2:
            if indexing == 'xy':
                self.x_steps = 10
                self.y_steps = 25
                r = np.linspace(0, self.x_max, self.x_steps)[:, np.newaxis]
                theta = np.linspace(0, 2*np.pi, self.y_steps)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                return x, y
            elif indexing == 'ij':
                self.x_steps = 25
                self.y_steps = 10
                theta = np.linspace(0, 2*np.pi, self.x_steps)[:, np.newaxis]
                r = np.linspace(0, self.x_max, self.y_steps)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                return x, y

    def grid_3D(self, indexing='xy'):
        """First electron positions (grid).
        :return:
        """
        self.x_steps = 100
        self.y_steps = 100
        x = np.linspace(self.x_min, self.x_max, self.x_steps)
        y = np.linspace(self.y_min, self.y_max, self.y_steps)
        self.z_steps = 3
        self.z_max = 2*self.max_L/(self.x_steps-1)
        self.z_min = - self.z_max
        z = np.linspace(self.z_min, self.z_max, self.z_steps)
        return np.meshgrid(x, y, z, indexing=indexing)

    def backflow(self, grid, channel, set=0):
        """Backflow.
        :param grid: first electron positions
        :param channel: [u-u] or [u-d]
        :return:
        """
        xy_elec = np.array(self.xy_elec)[:, np.newaxis, np.newaxis]
        xy_nucl = np.array(self.xy_nucl)[:, np.newaxis, np.newaxis]
        if self.term == 'PHI':
            return self.PHI(grid, xy_elec, [xy_nucl], channel, set)
        elif self.term == 'THETA':
            return self.THETA(grid, xy_elec, [xy_nucl], channel, set)
        elif self.term == 'ALL':
            return self.ALL(grid, xy_elec, [xy_nucl], channel, set)

    def backflow_3D(self, grid, channel, set):
        """Backflow.
        :param grid: first electron positions
        :param channel: [u-u] or [u-d]
        :return:
        """
        xy_elec = np.array(self.xy_elec + [0])[:, np.newaxis, np.newaxis, np.newaxis]
        xy_nucl = np.array(self.xy_nucl + [0])[:, np.newaxis, np.newaxis, np.newaxis]
        if self.term == 'PHI':
            return self.PHI(grid, xy_elec, [xy_nucl], channel, set)
        elif self.term == 'THETA':
            return self.THETA(grid, xy_elec, [xy_nucl], channel, set)
        elif self.term == 'ALL':
            return self.ALL(grid, xy_elec, [xy_nucl], channel, set)

    def jacobian_det(self, indexing, channel, set):
        """Jacobian matrix & determinant
        first electron is at the self.grid() i.e. (x,y,0)
        second electron is at the position (self.xy_elec[0],self.xy_elec[1],0)
        nucleus is at the position (0,0,0)

        jacobian[i][j] = dvect[xi]/dxj

        :return:
        """
        grid = self.grid_3D(indexing)
        vect = self.backflow_3D(grid, channel, set) + np.array(grid)
        jacobian = [np.gradient(comp, 2*self.max_L/(self.x_steps-1)) for comp in vect]
        jacobian = np.moveaxis(jacobian, 0, -1)
        jacobian = np.moveaxis(jacobian, 0, -1)
        det_sign = 1 if indexing == 'ij' else -1
        return det_sign * np.linalg.det(jacobian)

    def div(self, indexing, channel, set):
        """Divergence"""
        grid = self.grid_3D(indexing)
        vect = self.backflow_3D(grid, channel, set)
        jacobian = [np.gradient(comp, 2*self.max_L/(self.x_steps-1)) for comp in vect]
        return jacobian[0][0] + jacobian[1][1] + jacobian[2][2]

    def curl(self, indexing, channel, set):
        """Curl"""
        grid = self.grid_3D(indexing)
        vect = self.backflow_3D(grid, channel, set)
        jacobian = [np.gradient(comp, 2*self.max_L/(self.x_steps-1)) for comp in vect]
        return jacobian[2][1]-jacobian[1][2], jacobian[0][2]-jacobian[2][0], jacobian[1][0] - jacobian[0][1]

    def plot2D(self, replot=False):
        """Plot backflow.
        first electron is on the self.grid()
        second electron is at (self.xy_elec[0],self.xy_elec[1],0)
        nucleus is at (0,0,0)
        """
        if not replot:
            self.fig_2D, self.axs = plt.subplots(1, 2)
        for channel in range(2):
            self.axs[channel].clear()
            self.axs[channel].set_title('{} backflow {} term'.format(self.term, ['u-u', 'u-d'][channel]))
            self.axs[channel].set_aspect('equal', adjustable='box')
            self.axs[channel].plot(*self.xy_nucl, 'ro', label='nucleus')
            self.axs[channel].plot(*self.xy_elec, 'mo', label='electron')
            self.axs[channel].set_xlabel('X axis')
            self.axs[channel].set_ylabel('Y axis')
            if self.plot_type == 0:
                self.axs[channel].quiver(
                    *self.grid('xy'),
                    *self.backflow(self.grid('xy'), channel, self.set),
                    angles='xy', scale_units='xy',
                    scale=1, color=['blue', 'green'][channel]
                )
            elif self.plot_type == 1:
                for indexing in 'xy', 'ij':
                    self.axs[channel].plot(
                        *(self.grid(indexing) + self.backflow(self.grid(indexing), channel, self.set)),
                        color=['blue', 'green'][channel]
                    )
            elif self.plot_type == 2:
                for indexing in 'xy', 'ij':
                    self.axs[channel].plot(
                        *(self.grid(indexing) + self.backflow(self.grid(indexing), channel, self.set)),
                        color=['blue', 'green'][channel]
                    )
            elif self.plot_type == 3:
                contours = self.axs[channel].contour(
                    self.grid_3D('ij')[0][:, :, 1],
                    self.grid_3D('ij')[1][:, :, 1],
                    self.jacobian_det('ij', channel, self.set)[:, :, 1],
                    10,
                    colors='black'
                )
                plt.clabel(contours, inline=True, fontsize=8)
                # indexing = 'ij' for origin = 'lower' or indexing = 'xy' for origin = 'upper'
                img = self.axs[channel].imshow(
                    self.jacobian_det('ij', channel, self.set)[:, :, 1],
                    extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                    origin='lower', cmap='summer'
                )
                # plt.colorbar(img)
            if self.plot_cutoff:
                if self.ETA_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_elec, self.ETA_L[channel], fill=False, linestyle=':', label='ETA e-e cutoff'))
                if self.MU_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_nucl, self.MU_L[self.set], fill=False, color='c', label='MU e-n cutoff'))
                if self.PHI_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_nucl, self.PHI_L[self.set], fill=False, color='y', label='PHI e-n cutoff'))
                if self.PHI_CUSP and self.AE_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_nucl, self.AE_L, fill=False, label='AE cutoff'))
            self.axs[channel].legend()
        if replot:
            self.fig_2D.canvas.draw()
        else:
            self.fig_2D.canvas.mpl_connect('button_press_event', self.onclick_2D)
            self.fig_2D.canvas.mpl_connect('key_press_event', self.onpress)

    def plot3D(self, replot=False):
        """Plot backflow.
        first electron is on the self.grid()
        second electron is at (self.xy_elec[0], self.xy_elec[1], 0)
        nucleus is at (0,0,0)
        """
        channel = self.channel_3D
        if not replot:
            self.fig_3D = plt.figure()
            self.ax = self.fig_3D.add_subplot(111, projection='3d')
        self.ax.clear()
        if self.plot_3d_type == 0:
            self.ax.set_title('Jacobian {} backflow {} term'.format(self.term, ['u-u', 'u-d'][channel]))
            # https://github.com/matplotlib/matplotlib/issues/487 - masked array not supported
            mask = (self.grid_3D('ij')[0] < 0) | (self.grid_3D('ij')[1] > 0)
            jacobian = np.where(mask, self.jacobian_det('ij', channel, self.set), np.nan)
            self.ax.plot_wireframe(
                self.grid_3D('ij')[0][:, :, 1],
                self.grid_3D('ij')[1][:, :, 1],
                jacobian[:, :, 1],
                color=['blue', 'green'][channel]
            )
        elif self.plot_3d_type == 1:
            self.ax.set_title('Curl[Z] {} backflow {} term'.format(self.term, ['u-u', 'u-d'][channel]))
            self.ax.plot_wireframe(
                self.grid_3D('ij')[0][:, :, 1],
                self.grid_3D('ij')[1][:, :, 1],
                self.curl('ij', channel, self.set)[2][:, :, 1],
                color=['blue', 'green'][channel]
            )
            # self.ax.quiver(
            #     self.grid_3D('ij')[0][:, :, 1],
            #     self.grid_3D('ij')[1][:, :, 1],
            #     self.grid_3D('ij')[2][:, :, 1],
            #     self.curl('ij', channel)[0][:, :, 1],
            #     self.curl('ij', channel)[1][:, :, 1],
            #     self.curl('ij', channel)[2][:, :, 1],
            #     pivot='tail', color=['blue', 'green'][channel]
            # )
        elif self.plot_3d_type == 2:
            self.ax.set_title('Div {} backflow {} term'.format(self.term, ['u-u', 'u-d'][channel]))
            self.ax.plot_wireframe(
                self.grid_3D('ij')[0][:, :, 1],
                self.grid_3D('ij')[1][:, :, 1],
                self.div('ij', channel, self.set)[:, :, 1],
                color=['blue', 'green'][channel]
            )
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        if replot:
            self.fig_3D.canvas.draw()
        else:
            self.fig_3D.canvas.mpl_connect('key_press_event', self.onpress)

    def plot(self, replot=False):
        self.plot2D(replot=replot)
        self.plot3D(replot=replot)
        if not replot:
            plt.show()

    def onclick_2D(self, event):
        """On click"""
        if not (event.xdata and event.ydata):
            return
        if not (self.x_min < event.xdata < self.x_max or self.y_min < event.xdata < self.y_max):
            return
        self.xy_elec = [event.xdata, event.ydata]
        self.plot(replot=True)

    def onpress(self, event):
        """On key pressed"""
        if event.key == 'f1':
            self.plot_type = (self.plot_type + 1) % 4
        elif event.key == 'f2':
            self.plot_cutoff = not self.plot_cutoff
        elif event.key == 'f3':
            self.channel_3D = (self.channel_3D + 1) % 2
        elif event.key == 'f4':
            self.plot_3d_type = (self.plot_3d_type + 1) % 3
        elif event.key == 'f5':
            self.set = (self.set + 1) % self.MU_sets
        self.plot(replot=True)


def main():
    parser = argparse.ArgumentParser(
        description="This script to plot backflow terms.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'term',
        type=str,
        help="BACKFLOW term ETA, MU, PHI, THETA, ALL"
    )
    parser.add_argument(
        'backflow_file',
        type=str,
        default='correlation.data',
        nargs='?',
        help="name of correlation.* file"
    )

    args = parser.parse_args()

    if args.term in ('ETA', 'MU'):
        Plot1D(args.term, args.backflow_file).plot()
    elif args.term in ('PHI', 'THETA', 'ALL'):
        Plot2D(args.term, args.backflow_file).plot()


if __name__ == "__main__":
    main()
