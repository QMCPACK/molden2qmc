#!/usr/bin/env python3

import argparse
import numpy as np


class Gwfn:
    """Gaussian wfn reader from file."""

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> 2l+1
    shell_map = {1: 1, 2: 3, 3: 3, 4: 5, 5: 7, 6: 9}

    def __init__(self, file_name, mult=1):
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
                elif line.startswith('Number of Gaussian centres'):
                    self.natom = read_int()
                elif line.startswith('Number of shells per primitive cell'):
                    self.nshell = read_int()
                elif line.startswith('Number of basis functions'):
                    self.nbasis_functions = read_int()
                elif line.startswith('Number of Gaussian primitives'):
                    self.nprimitives = read_int()
                elif line.startswith('Code for shell types'):
                    self.shell_types = read_ints(self.nshell)
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self.primitives = read_ints(self.nshell)
                elif line.startswith('Exponents of Gaussian primitives'):
                    tail = read_floats(self.nprimitives)
                    self.exponents = np.zeros((self.nbasis_functions, max(self.primitives)))
                    n = 0
                    for primitive, shell_type in zip(self.primitives, self.shell_types):
                        head, tail = tail[:primitive], tail[primitive:]
                        for j in range(self.shell_map[shell_type]):
                            self.exponents[n, :primitive] = head
                            n += 1
                elif line.startswith('Normalized contraction coefficients'):
                    tail = read_floats(self.nprimitives)
                    self.contraction_coefficients = np.zeros((self.nbasis_functions, max(self.primitives)))
                    n = 0
                    for primitive, shell_type in zip(self.primitives, self.shell_types):
                        head, tail = tail[:primitive], tail[primitive:]
                        for j in range(self.shell_map[shell_type]):
                            self.contraction_coefficients[n, :primitive] = head
                            n += 1
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    line = fp.readline()
                    mo = read_floats(self.nbasis_functions * self.nbasis_functions)
                    self.mo = np.array(mo).reshape((self.nbasis_functions, self.nbasis_functions))

    @staticmethod
    def angular_part(shell_type, r):
        """
        from CASINO/examples/generic/gauss_dfg/README
        :param shell_type: shell types (s/sp/p/d/f... 1/2/3/4/5...)
        :param r: electron coordinates with shape (3, n, m, l, ...)
        :return:
        """
        harmonic = np.array([0])
        if shell_type == 1:
            harmonic = np.array([1])
        if shell_type == 3:
            harmonic = np.array(r)
        x, y, z = r
        if shell_type == 4:
            harmonic = np.array([
                       3 * z**2 - r**2,
                       x * z,
                       y * z,
                       x**2 + y**2,
                       x * y
                   ])
        if shell_type == 5:
            harmonic = np.array([
                       z * (5 * z**2 - 3 * r**2) / 2,
                       3 * x * (5 * z**2 - 3 * r**2) / 2,
                       3 * y * (5 * z**2 - 3 * r**2) / 2,
                       15 * z * (x**2 - y**2),
                       30 * x * y * z,
                       15 * x * (x**2 - 3 * y**2),
                       15 * y * (3 * x**2 - y*2),
                   ])
        if shell_type == 6:
            harmonic = np.array([
                       (35 * z*z*z*z - 30 * z*z*r**2 + 3 * r**4) / 8,
                       5 * x * z * (7 * z * z - 3 * r**2) / 2,
                       5 * y * z * (7 * z * z - 3 * r**2) / 2,
                       15 * (x*x - y*y) * (7 * z * z - r**2) / 2,
                       30 * x * y * (7 * z * z - r**2) / 2,
                       105 * x * z * (x*x - 3 * y*y),
                       105 * y * z * (3 * x*x - y*y),
                       105 * (x*x*x*x - 6 * x*x*y*y + y*y*y*y),
                       420 * x * y * (x*x - y*y)
                   ])
        return np.einsum('i...,j...->j...', radial, harmonic)

    def wfn(self, r, mo):
        """
        ao - array[self.nbasis_functions]
        angular part of spherical harmonic - array[self.nbasis_functions]
        coeff - array[self.nprimitives]
        np.exp(-alpha*r^2) - array[self.nprimitives]

        r -> [3, a, b, c]
        ao -> array[self.nbasis_functions]
        coeff -> array[self.nbasis_functions, max_nprim_inbasis_wfn]
        alpha -> array[self.nbasis_functions, max_nprim_inbasis_wfn]
        coeff * np.exp(-alpha*r^2) -> array[self.nbasis_functions, max_nprim_inbasis_wfn, a, b, c]
        angular part of spherical harmonic > array[self.nbasis_functions, a, b, c]

        SUM along [self.nbasis_functions, max_nprim_inbasis_wfn]
        """
        r2 = np.sum(r**2, axis=0)[:, :, np.newaxis, np.newaxis]
        radial = np.sum(self.contraction_coefficients * np.exp(-self.exponents * r2), axis=-1)
        angular = mo[0]
        return np.sum(angular * radial, axis=-1)


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
    r = np.array([[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]])
    print(gwfn.wfn(r, gwfn.mo[0]))


if __name__ == "__main__":
    main()
