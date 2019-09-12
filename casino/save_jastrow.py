#!/usr/bin/env python3

import argparse
import copy
from yaml import dump, Dumper
from yaml.events import SequenceEndEvent

parser = argparse.ArgumentParser(
    description="This script creates parameters.casl file from existion gwfn.data file and rules.",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument('gwfn_file', type=str, default='gwfn.data', nargs='?', help="name of gwfn.data file")
parser.add_argument('casl_file', type=str, default='parameters.casl', nargs='?', help="name of parameters.casl file")

args = parser.parse_args()

with open(args.gwfn_file, "r") as gwfn:
    result = ''
    line = gwfn.readline()
    while line and not line.startswith('Atomic numbers for each atom:'):
        line = gwfn.readline()
    if not line:
        raise Exception
    line = gwfn.readline()
    while line and not line.startswith('Valence charges for each atom:'):
        result += line
        line = gwfn.readline()

atomic_numbers = map(int, result.split())

Rules = []
Erules = []

for atom in set(atomic_numbers):
    first_occurense = None
    for position, next_atom in enumerate(atomic_numbers):
        if atom == next_atom:
            if first_occurense is None:
                first_occurense = position+1
                Erules.append('1-n%s=2-n%s' % (first_occurense, position+1))
            else:
                Rules.append('n%s=n%s' % (first_occurense, position+1))

Rules_1_1 = copy.copy(Rules)
Rules_1_1.append('1=2')

Rules_2_1 = copy.copy(Rules)
Rules_2_1.extend(Erules)
Rules_2_1.append('1-1=2-2')

casl = {
    'JASTROW': {
        'Title': 'al',
        'TERM 1': {
            'Rules': ['1-1=2-2'],
            'e-e basis': [{'Type': 'natural power'}, {'Order': 2}],
            'e-e cusp': 'T',
            'e-e cutoff': [{'Type': 'alt polynomial'}],
            'Rank': [2, 0]
        },
        'TERM 2': {
            'Rules': Rules_1_1,
            'e-n basis': [{'Type': 'natural power'}, {'Order': 2}],
            'e-n cutoff': [{'Type': 'alt polynomial'}],
            'Rank': [1, 1]
        },
        'TERM 3': {
            'Rules': Rules_2_1,
            'e-e basis': [{'Type': 'natural power'}, {'Order': 4}],
            'e-n basis': [{'Type': 'natural power'}, {'Order': 4}],
            'e-n cutoff': [{'Type': 'alt polynomial'}],
            'Rank': [2, 1]
        }
    }
}


class CaslDumper(Dumper):

    def expect_block_sequence_item(self, first=False):
        if not first and isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        else:
            self.write_indent()
            self.write_indicator(u' ', True, indention=True)
            self.states.append(self.expect_block_sequence_item)
            self.expect_node(sequence=True)

with open(args.casl_file, 'w') as f:
    dump(casl, f, default_flow_style=False, Dumper=CaslDumper)
