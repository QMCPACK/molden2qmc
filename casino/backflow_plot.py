#!/usr/bin/env python3

import argparse
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval3d
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

from backflow import Backflow

# TODO: implement a method to write a backflow file
# TODO: write a generator of monomials of THETA and PHI polynom. decomposition
# TODO: interpolate backflow of order X_Y_ZZ, with polynomial functions.
# TODO: interpolate backflow of order X_Y_ZZ, non-polynomial functions.


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

    def backflow(self, grid, channel, set=None):
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
