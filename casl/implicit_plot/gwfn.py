#!/usr/bin/env python3

import argparse
import numpy as np
from numpy import exp
from numpy.linalg import norm


class Gwfn:
    """Gaussian wfn reader from file."""

    def __init__(self, file_name):
        """Init."""
        self.atoms = list()
        self.atom_numbers = list()
        self.shell_code = list()
        self.shell_primitives = list()
        self.primitives = list()
        self.coeffs = list()
        self.mo = list()
        self.read(file_name)

    def read(self, file_name):
        """Open file and read gwfn data."""
        def read_bool():
            return fp.readline().strip() == '.true.'

        def read_str():
            return str(fp.readline())

        def read_int():
            return int(fp.readline())

        def read_float():
            return float(fp.readline())

        def read_ints(n):
            result = list()
            while len(result) < n:
                line = fp.readline()
                result += map(int, line.split())
            return result

        def read_floats(n):
            result = list()
            while len(result) < n:
                line = fp.readline()
                result += map(float, [line[i*20:(i+1)*20] for i in range(len(line)//20)])
            return result

        with open(file_name, 'r') as fp:
            for line in fp:
                if line.startswith('TITLE'):
                    self.title = read_str()
                # BASIC_INFO
                # ----------
                elif line.startswith('Spin unrestricted'):
                    self.unrestricted = read_bool()
                elif line.startswith('Number of electrons'):
                    self.nelec = read_int()
                # GEOMETRY
                # --------
                elif line.startswith('Number of atoms'):
                    self.natom = read_int()
                elif line.startswith('Atomic positions'):
                    self.atoms = read_floats(self.natom * 3)
                elif line.startswith('Atomic numbers for each atom'):
                    self.atom_numbers = read_ints(self.natom)
                # BASIS SET
                # ---------
                elif line.startswith('Number of shells per primitive cell'):
                    self.nshell = read_int()
                elif line.startswith('Number of basis functions'):
                    self.nbasis = read_int()
                elif line.startswith('Number of Gaussian primitives'):
                    self.nprimitives = read_int()
                elif line.startswith('Code for shell types'):
                    self.shell_code = read_ints(self.nshell)
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self.shell_primitives = read_ints(self.nshell)
                elif line.startswith('Exponents of Gaussian primitives'):
                    self.primitives = read_floats(self.nprimitives)
                elif line.startswith('Normalized contraction coefficients'):
                    self.coeffs = read_floats(self.nprimitives)
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    line = fp.readline()
                    self.mo = read_floats(self.nbasis * self.nbasis)

    def s(self, alpha, r):
        return exp(-alpha * np.sum(r**2))

    def p_x(self, alpha, r):
        return r[0] * exp(-alpha * np.sum(r**2))

    def p_y(self, alpha, r):
        return r[1] * exp(-alpha * np.sum(r**2))

    def p_z(self, alpha, r):
        return r[2] * exp(-alpha * np.sum(r**2))

    def wfn(self, r, fn):
        return sum(self.coeffs[i] * fn(self.primitives[i], r) for i in range(self.nprimitives))


def main():
    parser = argparse.ArgumentParser(
        description="This script to read gwfn.data file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'gwfn_file',
        type=str,
        default='gwfn.data',
        nargs='?',
        help="name of correlation.* file"
    )

    args = parser.parse_args()
    gwfn = Gwfn(args.gwfn_file)
    r = np.array([1,1,1])
    print(gwfn.wfn(r, gwfn.s))
    print(gwfn.atoms)

if __name__ == "__main__":
    main()
