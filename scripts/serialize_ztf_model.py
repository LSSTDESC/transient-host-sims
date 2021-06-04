#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Alexandre Boucaud <aboucaud@lal.in2p3.fr>
# Edited by Patrick Aleo (Github: patrickaleo) for ZTF
# Only g,r,(i) filters

import argparse
import os
import glob
from pathlib import Path

from validutils_ztf.io import save_compressed #Made validutils_ztf for only g,r filters
from validutils_ztf.table import parse_lsst_model


def parse_args():

    parser = argparse.ArgumentParser("LSST supernovae simulations serializer")
    parser.add_argument('directory', type=str,
                        help="Simulation directory (MODEL)")
    parser.add_argument('--destination', type=str, default='.',
                        help="Path to the output files")
    parser.add_argument('--timed', action='store_true',
                        help="Add a timer")
    parser.add_argument('--maxitems', type=int, default=0,
                        help="Limit maximum number of objects to be serialized")
    parser.add_argument('--recursive', action='store_true',
                        help="Serialize a lot of models")
    parser.add_argument('--root', default='LSST',
                        help='Specify a root name for the files to parse')
    parser.add_argument('--model', default='*',
                        help='Specify a model number string to parse')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.maxitems <= 0:
        maxitems = None
    else:
        maxitems = args.maxitems

    if args.timed:
        import time
        start = time.time()

    rootstr = args.root
    if rootstr not in ('LSST', 'IDEAL'):
        message = 'Unknown root string {}'.format(rootstr)
        raise ValueError(message)

    modelstr = args.model

    rootdir = str(Path(args.directory))
    if args.recursive:
        dirpattern = os.path.join(rootdir, '{}_*_MODEL{}'.format(rootstr, modelstr))
    else:
        dirpattern = rootdir
    dirs = glob.glob(dirpattern)

    for thisdir in dirs:
        message = 'Serializing light curves in: {}'.format(thisdir)
        print(message)

        dirpath = Path(thisdir)

        # Output directory
        outdir = Path(args.destination)
        if not outdir.exists():
            outdir.mkdir()

        # Use model directory as base name for the output file
        filename = "{}.pkl.gz".format(dirpath.name)
        outfile = outdir / filename

        table = parse_lsst_model(dirpath, maxitems=maxitems)
        save_compressed(table, outfile)

        print("Light curve data from {} saved to {}".format(dirpath.name, outfile))

    if args.timed:
        end = time.time()
        secs = end - start
        print("LSST light curves serialized in {:.1f} seconds".format(secs))


if __name__ == '__main__':
    main()
