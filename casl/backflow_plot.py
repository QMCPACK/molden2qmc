#!/usr/bin/env python3

import argparse
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D


# TODO: implement a method to write a backflow file
# plot a CURL
# TODO: write a generator of monomials in THETA and PHI decomposition
# TODO: интерполировать backflow порядка X_Y_ZZ, более меньшим порядком.
# TODO: интерполировать backflow порядка X_Y_ZZ, не полиномиальными функциями.

class Backflow:
    """Backflow reader from file.
    Inhomogeneous backflow transformations in quantum Monte Carlo.
    P. Lopez Rıos, A. Ma, N. D. Drummond, M. D. Towler, and R. J. Needs
    http://www.tcm.phy.cam.ac.uk/~mdt26/downloads/lopezrios_backflow.pdf
    """

    def __init__(self):
        """Init."""
        self.ETA_TERM = None
        self.MU_TERM = None
        self.PHI_TERM = None
        self.THETA_TERM = None
        self.ETA_L = None
        self.MU_L = None
        self.PHI_L = None
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
            AE = False
            ETA = ETA_params = False
            MU = MU_params = False
            PHI = PHI_params = False
            line = f.readline()
            eta_powers = self.ETA_powers()
            mu_powers = self.MU_powers()
            phi_powers = self.PHI_powers()
            theta_powers = self.THETA_powers()
            while line:
                line = f.readline()
                if line.strip().startswith('Truncation order'):
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
                    if line.strip().startswith('Type of e-N cusp conditions'):
                        self.MU_CUSP = bool(int(f.readline().split()[0]))
                    elif line.strip().startswith('Expansion order'):
                        self.MU_order = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Spin dep'):
                        self.MU_spin_dep = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Cutoff (a.u.)'):
                        self.MU_L = float(f.readline().split()[0])
                    elif line.strip().startswith('Parameter values'):
                        self.MU_TERM = np.zeros((self.MU_order, self.MU_spin_dep), 'd')
                        MU_params = True
                    elif MU_params:
                        if line.strip().startswith('END SET'):
                            MU_params = False
                        else:
                            a, b = map(int, line.split()[3].split('_')[1].split(','))
                            assert (a, b) == next(mu_powers)
                            self.MU_TERM[a][b-1] = float(line.split()[0])
                elif PHI:
                    if line.strip().startswith('Type of e-N cusp conditions'):
                        self.PHI_CUSP = bool(int(f.readline().split()[0]))
                    elif line.strip().startswith('Irrotational Phi'):
                        self.PHI_irrotational = bool(int(f.readline().split()[0]))
                    elif line.strip().startswith('Electron-nucleus expansion order'):
                        self.PHI_en_order = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Electron-electron expansion order'):
                        self.PHI_ee_order = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Spin dep'):
                        self.PHI_spin_dep = int(f.readline().split()[0]) + 1
                    elif line.strip().startswith('Cutoff (a.u.)'):
                        self.PHI_L = float(f.readline().split()[0])
                    elif line.strip().startswith('Parameter values'):
                        self.PHI_TERM = np.zeros((self.PHI_en_order, self.PHI_en_order, self.PHI_ee_order, self.PHI_spin_dep), 'd')
                        if not self.PHI_irrotational:
                            self.THETA_TERM = np.zeros((self.PHI_en_order, self.PHI_en_order, self.PHI_ee_order, self.PHI_spin_dep), 'd')
                        PHI_params = True
                    elif PHI_params:
                        if line.strip().startswith('END SET'):
                            PHI_params = False
                        else:
                            a, b, c, d = map(int, line.split()[3].split('_')[1].split(','))
                            if line.split()[3].split('_')[0] == 'phi':
                                self.PHI_TERM[a][b][c][d-1] = float(line.split()[0])
                            elif line.split()[3].split('_')[0] == 'theta':
                                # print((a, b, c, d), next(theta_powers))
                                self.THETA_TERM[a][b][c][d-1] = float(line.split()[0])
                elif AE:
                    if line.strip().startswith('Nucleus'):
                        self.AE_L = float(f.readline().split()[2])

    def cutoff(self, r, L):
        """General cutoff"""
        return (1 - r/L)**self.C * np.heaviside(L-r, 0.0)

    def AE_cutoff(self, r, L):
        """All electron atom cutoff"""
        return np.where(r < L, (r/L)**2 * (6 - 8 * (r/L) + 3 * (r/L)**2), 1.0)

    def ETA(self, ri, rj, rI, channel):
        """ETA term"""
        rij = np.hypot(*(ri - rj))
        channel = min(self.ETA_spin_dep-1, channel)
        result = (self.cutoff(rij, self.ETA_L[channel]) *
                  polyval(rij, self.ETA_TERM[:, channel]))
        if rI is not None:
            riI = np.hypot(*(ri - rI))
            result *= self.AE_cutoff(riI, self.AE_L)
        return result

    def MU(self, ri, rI, channel):
        """MU term"""
        riI = np.hypot(*(ri - rI))
        channel = min(self.MU_spin_dep-1, channel)
        return (self.cutoff(riI, self.MU_L) *
                polyval(riI, self.MU_TERM[:, channel]))

    def PHI(self, ri, rj, rI, channel):
        """PHI term"""
        rij = np.hypot(*(ri - rj))
        riI = np.hypot(*(ri - rI))
        rjI = np.hypot(*(rj - rI))
        result = (self.cutoff(riI, self.PHI_L) *
                  self.cutoff(rjI, self.PHI_L) *
                  polyval3d(riI, rjI, rij, self.PHI_TERM[:, :, :, channel]))
        if self.PHI_CUSP:
            result *= self.AE_cutoff(riI, self.AE_L)
        return result

    def THETA(self, ri, rj, rI, channel):
        """THETA term"""
        rij = np.hypot(*(ri - rj))
        riI = np.hypot(*(ri - rI))
        rjI = np.hypot(*(rj - rI))
        return (self.cutoff(riI, self.PHI_L) *
                self.cutoff(rjI, self.PHI_L) *
                polyval3d(riI, rjI, rij, self.THETA_TERM[:, :, :, channel]))

    def ALL(self, ri, rj, rI, channel):
        """Total displacement"""
        result = 0
        if self.ETA_TERM is not None:
            result += self.ETA(ri, rj, rI, channel) * (rj - ri)
        if self.MU_TERM is not None:
            result += self.MU(ri, rI, channel) * (rI - ri)
        if self.PHI_TERM is not None:
            result += self.PHI(ri, rj, rI, channel) * (rj - ri)
            if not self.PHI_irrotational:
                result += self.THETA(ri, rj, rI, channel) * (rI - ri)
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
        self.x_min = self.y_min = self.y_max = 0.0
        self.x_max = 10.0
        self.x_steps = 101
        self.y_steps = 1
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]

    def grid(self, indexing='xy'):
        """Electron positions (grid).
        :param indexing: cartesian or matrix indexing of output
        :return:
        """
        x = np.linspace(self.x_min, self.x_max, self.x_steps)
        y = np.linspace(self.y_min, self.y_max, self.y_steps)
        return np.meshgrid(x, y, indexing=indexing)

    def backflow(self, grid, channel):
        """Backflow.
        :param grid: electron positions
        :param channel: [u] or [d]
        :return:
        """
        if self.term == 'ETA':
            return self.ETA(grid, self.xy_elec, None, channel) * (self.xy_elec - grid)
        elif self.term == 'MU':
            return self.MU(grid, self.xy_nucl, channel) * (self.xy_nucl - grid)

    @property
    def spin_dep(self):
        if self.term == 'ETA':
            return self.ETA_spin_dep
        elif self.term == 'MU':
            return self.MU_spin_dep

    def label(self, channel):
        if self.term == 'ETA':
            return ['u-u', 'u-d'][channel]
        elif self.term == 'MU':
            return ['u', 'd'][channel]

    def plot(self):
        """
        electron is at (0,0,0) for ETA term
        nucleus is at (0,0,0) for MU term
        """
        for channel in range(self.spin_dep):
            plt.plot(self.grid()[0][0], self.backflow(self.grid(), channel)[0][0], label=self.label(channel))
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
        self.term = term
        self.read(file)
        self.x_max = self.y_max = self.max_L
        self.x_min = self.y_min = -self.max_L
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.plot_cutoff = False
        self.plot_type = 0
        self.channel_3D = 0

    @property
    def max_L(self):
        """max cutoff"""
        return max(np.max(self.ETA_L) or 0, self.MU_L or 0, self.PHI_L or 0)

    def grid(self, indexing='xy'):
        """First electron positions (grid).
        :param indexing: cartesian or matrix indexing of output
        :return:
        """
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
        return np.meshgrid(x, y, indexing=indexing)

    def backflow(self, grid, channel):
        """Backflow.
        :param grid: first electron positions
        :param channel: [u-u] or [u-d]
        :return:
        """
        if self.term == 'PHI':
            return self.PHI(grid, self.xy_elec, self.xy_nucl, channel) * (self.xy_elec - grid)
        elif self.term == 'THETA':
            return self.THETA(grid, self.xy_elec, self.xy_nucl, channel) * (self.xy_nucl - grid)
        elif self.term == 'ALL':
            return self.ALL(grid, self.xy_elec, self.xy_nucl, channel)

    def backflow_z(self, ri, rj, rI, channel):
        if self.term == 'PHI':
            return self.PHI(ri, rj, rI, channel)
        elif self.term == 'THETA':
            return self.THETA(ri, rj, rI, channel)
        elif self.term == 'ALL':
            result = 0
            if self.ETA_TERM is not None:
                result += self.ETA(ri, rj, rI, channel)
            if self.MU_TERM is not None:
                result += self.MU(ri, rI, channel)
            if self.PHI_TERM is not None:
                result += self.PHI(ri, rj, rI, channel)
                if not self.PHI_irrotational:
                    result += self.THETA(ri, rj, rI, channel)
            return result

    def jacobian_det(self, indexing, channel):
        """Jacobian matrix & det
        first electron is at the self.grid() i.e. (x,y,0)
        second electron is at the position (self.xy_elec[0],self.xy_elec[1],0)
        nucleus is at the position (0,0,0)
        For backflow displacement:
          Backflow[Z](x, y, 0) = z * (ETA(ri) + MU(ri-rj) + PHI(ri) + THETA(ri-rj)) = 0
         dBackflow[Z](x, y, 0)/dx = 0
         dBackflow[Z](x, y, 0)/dy = 0
         dBackflow[Z](x, y, 0)/dz = ETA(ri) + MU(ri-rj) + PHI(ri) + THETA(ri-rj)

                     | dBackflow[X]/dx  dBackflow[X]/dy     no matter    |   | 1  0  0 |
          jacobian = | dBackflow[Y]/dx  dBackflow[Y]/dy     no matter    | + | 0  1  0 |
                     |         0                0        dBackflow[Z]/dz |   | 0  0  1 |
        :return:
        """
        # self.z_steps = 3
        # self.z_max = self.max_L / (self.x_steps - 1)
        # self.z_min = - self.z_max
        # z = np.linspace(self.z_min, self.z_max, self.z_steps)

        grid = self.grid_3D(indexing)
        func = self.backflow(grid, channel) + np.array(grid)
        jacobian = [np.gradient(f, 2*self.max_L/(self.x_steps-1)) for f in func]
        det_xy = jacobian[0][0] * jacobian[1][1] - jacobian[1][0] * jacobian[0][1]
        det_z = 1 + self.backflow_z(grid, self.xy_elec, self.xy_nucl, channel)
        det_sign = 1 if indexing == 'ij' else -1
        return det_sign * det_xy * det_z

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
                self.axs[channel].quiver(*self.grid('xy'), *self.backflow(self.grid('xy'), channel), angles='xy', scale_units='xy', scale=1, color=['blue', 'green'][channel])
            elif self.plot_type == 1:
                for indexing in 'xy', 'ij':
                    self.axs[channel].plot(*(self.grid(indexing) + self.backflow(self.grid(indexing), channel)), color=['blue', 'green'][channel])
            elif self.plot_type == 2:
                for indexing in 'xy', 'ij':
                    self.axs[channel].plot(*(self.grid(indexing) + self.backflow(self.grid(indexing), channel)), color=['blue', 'green'][channel])
            elif self.plot_type == 3:
                contours = self.axs[channel].contour(*self.grid_3D('ij'), self.jacobian_det('ij', channel), 10, colors='black')
                plt.clabel(contours, inline=True, fontsize=8)
                # indexing = 'ij' for origin = 'lower' or indexing = 'xy' for origin = 'upper'
                img = self.axs[channel].imshow(self.jacobian_det('ij', channel), extent=[self.x_min, self.x_max, self.y_min, self.y_max], origin='lower', cmap='summer')
                # plt.colorbar(img)
            if self.plot_cutoff:
                if self.ETA_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_elec, self.ETA_L[channel], fill=False, linestyle=':', label='ETA e-e cutoff'))
                if self.MU_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_nucl, self.MU_L, fill=False, color='c', label='MU e-n cutoff'))
                if self.PHI_L is not None:
                    self.axs[channel].add_patch(Circle(self.xy_nucl, self.PHI_L, fill=False, color='y', label='PHI e-n cutoff'))
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
        # https://github.com/matplotlib/matplotlib/issues/487 - masked array not supported
        mask = (self.grid_3D('ij')[0] < 0) | (self.grid_3D('ij')[1] > 0)
        jacobian = np.where(mask, self.jacobian_det('ij', channel), np.nan)
        self.ax.plot_wireframe(*self.grid_3D('ij'), jacobian, color=['blue', 'green'][channel])
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
        self.xy_elec = np.array([event.xdata, event.ydata])[:, np.newaxis, np.newaxis]
        self.plot(replot=True)

    def onpress(self, event):
        """On key pressed"""
        if event.key == 'f1':
            self.plot_type = (self.plot_type + 1) % 4
        elif event.key == 'f2':
            self.plot_cutoff = not self.plot_cutoff
        elif event.key == 'f3':
            self.channel_3D = (self.channel_3D + 1) % 2
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
