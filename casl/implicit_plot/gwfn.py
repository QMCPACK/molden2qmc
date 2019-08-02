#!/usr/bin/env python3

import argparse
from numpy import exp
from numpy.linalg import norm


class Gwfn:
    """Gaussian wfn reader from file."""

    def __init__(self, file_name):
        """Init."""
        self.atoms = list()
        self.atom_numbers = list()
        self.primitives = list()
        self.coeffs = list()
        self.mo = list()
        self.read(file_name)

    def read(self, file_name):
        """Open file and read gwfn data."""
        with open(file_name, 'r') as fp:
            for line in fp:
                if line.startswith('TITLE'):
                    self.title = fp.readline()
                elif line.startswith('Spin unrestricted'):
                    self.unrestricted = True
                elif line.startswith('Number of electrons'):
                    self.nelec = int(fp.readline())
                elif line.startswith('Number of atoms'):
                    self.natom = int(fp.readline())
                elif line.startswith('Atomic positions'):
                    for i in range(self.natom):
                        line = fp.readline()
                        self.atoms.append([float(line[i*20:(i+1)*20]) for i in range(3)])
                elif line.startswith('Atomic numbers for each atom'):
                    while len(self.atom_numbers) < self.natom:
                        line = fp.readline()
                        self.atom_numbers += map(int, line.split())
                elif line.startswith('Number of basis functions'):
                    self.nbasis = int(fp.readline())
                elif line.startswith('Number of Gaussian primitives'):
                    self.nprimitives = int(fp.readline())
                elif line.startswith('Exponents of Gaussian primitives'):
                    while len(self.primitives) < self.nprimitives:
                        line = fp.readline()
                        self.primitives += [float(line[i*20:(i+1)*20]) for i in range(len(line)//20)]
                elif line.startswith('Normalized contraction coefficients'):
                    while len(self.coeffs) < self.nprimitives:
                        line = fp.readline()
                        self.coeffs += [float(line[i*20:(i+1)*20]) for i in range(len(line)//20)]
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    line = fp.readline()
                    while len(self.mo) < self.nbasis * self.nbasis:
                        line = fp.readline()
                        self.mo += [float(line[i*20:(i+1)*20]) for i in range(len(line)//20)]

    def wfn(self, r):
        result = 0
        for i in range(self.nprimitives):
            result += self.coeffs[i] * exp(-self.primitives[i]*norm(r, axis=0))
        return result

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
    print(gwfn.wfn([1,1,1]))

if __name__ == "__main__":
    main()
