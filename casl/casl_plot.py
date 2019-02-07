#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d, polyval3d
import sys
from itertools import combinations
from math import exp
from yaml import safe_load
from mpl_toolkits.mplot3d import Axes3D


class Jastrow:
    """Jastrow reader from file.
    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703 – Published 27 September 2012
    """

    def __init__(self, term_num, file):
        self.file = file
        self.term_num = term_num
        self.jastrow_data = self.read_input_file()
        self.channels = self.term['Linear parameters'].keys()

    def read_input_file(self):
        with open(self.file, 'r') as f:
            casl = safe_load(f)
            return casl['JASTROW']

    @property
    def term(self):
        try:
            return self.jastrow_data['TERM {}'.format(self.term_num)]
        except KeyError:
            print('TERM {} not found in the input file'.format(self.term_num))

    def rank(self, term):
        if isinstance(term['Rank'], list):
            return term['Rank']
        elif isinstance(term['Rank'], str):
            return list(map(int, term['Rank'].split()))

    def type(self, term):
        try:
            return term['Type']
        except TypeError:
            return term[0]['Type']

    def order(self, term):
        try:
            return term['Order']
        except TypeError:
            return term[1]['Order']

    def rules(self, term):
        if isinstance(term['Rules'], list):
            return term['Rules']
        elif isinstance(term['Rules'], str):
            return list(map(int, term['Rules'].split()))

    def C(self, term):
        try:
            return term['C']
        except TypeError:
            return term[0]['C']

    def linear_parameters(self, term, channel):
        """Load linear parameters into multidimensional array."""
        e_rank, n_rank = self.rank(term)
        dims = []
        if e_rank > 1:
            e_order = self.order(term['e-e basis'])
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.order(term['e-n basis'])
            dims += [n_order] * e_rank * n_rank

        linear_parameters = np.zeros(dims, 'd')
        for key, val in term['Linear parameters'][channel].items():
            index = map(lambda x: x-1, map(int, key.split('_')[1].split(',')))
            linear_parameters[tuple(index)] = val[0]
        return linear_parameters

    def basis(self, term, channel):
        """type of functional bases should be:
        natural_power
        cosine
        cosine with k-cutoff
        r/(r^b+a) power
        1/(r+a) power
        r/(r+a) power
        """
        if self.type(term) == 'natural power':
            return lambda r: r
        elif self.type(term) == 'r/(r^b+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            b = parameters[channel]['b'][0]
            return lambda r: r/(r**b+a)
        elif self.type(term) == '1/(r+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            return lambda r: 1/(r+a)
        elif self.type(term) == 'r/(r+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            return lambda r: r/(r+a)
        else:
            print('basis with a {} type is not supported'.format(self.type(term)))
            sys.exit(0)

    def cutoff(self, term, channel):
        """type of cutoff functions should be:
        polynomial
        alt polynomial
        gaussian
        anisotropic polynomial
        """
        if term is None:
            return lambda r: 1.0
        elif self.type(term) in ('polynomial', 'anisotropic polynomial'):
            C = self.C(term['Constants'])
            L = term['Parameters'][channel]['L'][0]
            return lambda r: (1-r/L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'alt polynomial':
            C = self.C(term['Constants'])
            L = term['Parameters'][channel]['L'][0]
            return lambda r: (r-L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'gaussian':
            L_hard = term['Parameters'][channel]['L_hard'][0]
            return lambda r: exp(-(r/L)**2) * np.heaviside(L_hard-r/L, 0.0)
        else:
            print('cutoff with {} type is not supported'.format(self.type(term)))
            sys.exit(0)

    def cutoff_channel(self, term, channel, i, j):
        """make cutoff Parameters channel from Linear parameters channel"""
        prefix, suffix = channel.split(' ')
        split_suffix = suffix.split('-')
        if self.rank(term) == [1, 2]:
            return prefix + ' ' + split_suffix[i] + '-' + split_suffix[1]
        else:
            return prefix + ' ' + split_suffix[i] + '-' + split_suffix[j]

    def jastrow(self, term, channel, *args):
        u"""JASTROW
        :param rank: jastrow rank
        :param channel:
        :param args: [ri1, ... rin, rI1, ...rIm]
        :return:
        """
        e_rank, n_rank = self.rank(term)
        p_args = []
        cutoff = 1.0
        if e_rank > 1:
            ee_basis = self.basis(term['e-e basis'], channel)
            for i, ri, in enumerate(args[:e_rank]):
                for j, rj in enumerate(args[:e_rank]):
                    if i < j:
                        rij = np.hypot(*(ri - rj))
                        p_args.append(ee_basis(rij))
                        ee_cutoff = self.cutoff(
                            term.get('e-e cutoff'),
                            self.cutoff_channel(term, channel, i, j)
                        )
                        cutoff *= ee_cutoff(rij)

        if n_rank > 0:
            en_basis = self.basis(term['e-n basis'], channel)
            for i, ri in enumerate(args[:e_rank]):
                for j, rI in enumerate(args[e_rank:], e_rank):
                    riI = np.hypot(*(ri - rI))
                    p_args.append(en_basis(riI))
                    en_cutoff = self.cutoff(
                        term.get('e-n cutoff'),
                        self.cutoff_channel(term, channel, i, j)
                    )
                    cutoff *= en_cutoff(riI)

        if len(p_args) == 1:
            result = polyval(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 2:
            result = polyval2d(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 3:
            result = polyval3d(*p_args, self.linear_parameters(term, channel))
        return np.exp(result * cutoff)


class Plot(Jastrow):
    """Plot along the line."""

    def __init__(self, term_num, file):
        super().__init__(term_num, file)

    @property
    def grid(self):
        x = np.linspace(self.x_min, self.x_max, self.x_steps)
        y = np.linspace(self.y_min, self.y_max, self.y_steps)
        return np.meshgrid(x, y)

    def plot1D_a(self, replot=False):
        self.x_min = 0.0
        self.x_max = 5.0
        self.y_min = 0.0
        self.y_max = 0.0
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = 200
        self.y_steps = 1
        for channel in self.channels:
            plt.plot(self.grid[0][0], self.jastrow(self.term, channel, self.grid, self.xy_elec)[0], label=channel)
        plt.xlabel('r (au)')
        plt.ylabel('value')
        plt.title('JASTROW term')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot1D_b(self, replot=False):
        self.x_min = 0.0
        self.x_max = 5.0
        self.y_min = 0.0
        self.y_max = 0.0
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = 200
        self.y_steps = 1
        for channel in self.channels:
            plt.plot(self.grid[0][0], self.jastrow(self.term, channel, self.grid, self.xy_nucl)[0], label=channel)
        plt.xlabel('r (au)')
        plt.ylabel('value')
        plt.title('JASTROW term')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot2D(self, replot=False):
        self.x_min = -5.0
        self.x_max = 5.0
        self.y_min = -5.0
        self.y_max = 5.0
        self.xy_nucl_1 = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl_2 = np.array([2.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = self.y_steps = 40
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        for channel in self.channels:
            self.ax.plot_wireframe(*self.grid, self.jastrow(self.term, channel, self.grid, self.xy_nucl_1, self.xy_nucl_2))
        fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        plt.show()

    def plot3D(self, replot=False):

        if not replot:
            self.x_min = -3.0
            self.x_max = 3.0
            self.y_min = -3.0
            self.y_max = 3.0
            self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
            self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
            self.x_steps = self.y_steps = 30
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            # self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        for i, channel in enumerate(self.channels):
            self.ax.plot_wireframe(*self.grid, self.jastrow(self.term, channel, self.grid, self.xy_elec, self.xy_nucl), color=['blue', 'green'][i])
            #
            # contours = self.ax.contour(*self.xy_grid(), self.jastrow3D(np.array(self.xy_grid()), self.xy_elec, self.xy_nucl, channel), 10, colors='black')
            # plt.clabel(contours, inline=True, fontsize=8)
            # indexing = 'ij' for origin = 'lower' or indexing = 'xy' for origin = 'upper'
            # img = self.ax.imshow(self.jastrow3D(np.array(self.xy_grid()), self.xy_elec, self.xy_nucl, channel), extent=[self.x_min, self.x_max, self.y_min, self.y_max], origin='lower', cmap='summer')
            # plt.colorbar(img)

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        if replot:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onpress)
            plt.show()

    def plot(self, replot=False):
        """Plot single JASTROW term"""
        if self.rank(self.term) == [2, 0]:
            self.plot1D_a(replot)
        elif self.rank(self.term) == [1, 1]:
            self.plot1D_b(replot)
        elif self.rank(self.term) == [1, 2]:
            self.plot2D(replot)
        elif self.rank(self.term) == [2, 1]:
            self.plot3D(replot)
        else:
            print('term with {} rank is not supported'.format(self.rank(self.term)))

    def onclick(self, event):
        """On click"""
        if not (event.xdata and event.ydata):
            return
        if not (self.x_min < event.xdata < self.x_max or self.y_min < event.xdata < self.y_max):
            return
        self.xy_elec = np.array([event.xdata, event.ydata])[:, np.newaxis, np.newaxis]
        self.plot(replot=True)

    def onpress(self, event):
        """On key pressed"""
        delta = 0.2
        if event.key == 'right':
            self.xy_elec[0] += delta
        elif event.key == 'left':
            self.xy_elec[0] -= delta
        elif event.key == 'up':
            self.xy_elec[1] += delta
        elif event.key == 'down':
            self.xy_elec[1] -= delta
        self.plot(replot=True)


def main():
    parser = argparse.ArgumentParser(
        description="This script plot JASTOW terms.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'term_num',
        type=int,
        help="JASTROW term number"
    )
    parser.add_argument(
        'casl_file',
        type=str,
        default='parameters.casl',
        nargs='?',
        help="name of *.casl file"
    )

    args = parser.parse_args()

    Plot(args.term_num, args.casl_file).plot()


if __name__ == "__main__":
    main()
