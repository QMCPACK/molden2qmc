#!/usr/bin/env python3

import argparse


class Gwfn:
    """Gaussian wfn reader from file."""

    def __init__(self):
        """Init."""
        self.atoms = list()
        self.atom_numbers = list()
        self.primitives = list()
        self.coeffs = list()

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
                elif line.startswith('Number of Gaussian primitives'):
                    self.nprimitives = int(fp.readline())
                elif line.startswith('Exponents of Gaussian primitives'):
                    while len(self.primitives) < self.nprimitives:
                        line = fp.readline()
                        self.primitives += [float(line[i*20:(i+1)*20]) for i in range(4)]
                elif line.startswith('Normalized contraction coefficients'):
                    while len(self.coeffs) < self.nprimitives:
                        line = fp.readline()
                        self.coeffs += [float(line[i*20:(i+1)*20]) for i in range(4)]


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
    Gwfn().read(args.gwfn_file)

if __name__ == "__main__":
    main()
