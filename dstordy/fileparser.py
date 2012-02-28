#!/usr/bin/python
from protein import Protein

def parse(filename):
    f = open(filename)
    proteins = []
    valid = False
    current = Protein()

    for line in f:
        if line.startswith("#") or line == "end\n":
            continue

        elif line == "<>\n":
            valid = True
            if len(current._value) > 0:
                proteins.append(current)
            current = Protein()

        elif line == "<end>\n":
            valid = False
            proteins.append(current)
            current = Protein()

        elif valid:
            current.append_data(line[0],line[2])
    return proteins
