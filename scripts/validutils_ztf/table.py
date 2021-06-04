"""
Parsers for astro tables
"""
from collections import defaultdict

from astropy.table import Table


LSST_FILTERS = 'ugriz'


def parse_phot_table(table, rows):
    """
    Retrieve filter information from the photometric file

    Parameters
    ----------
    path : str
        path to LSST light curve file
    rows : slice
        range of rows for this SN

    Returns
    -------
    dict
        dictionary of filter data for the light curve

    """
    fitstable = table[rows]

    data = {}

    for filt in LSST_FILTERS:
        data[filt] = defaultdict(list)

    for row in fitstable:
        filt = row['BAND'].strip() #was FLT
        if filt not in LSST_FILTERS:
            continue
        data[filt]['mjd'].append(row['MJD'])
        data[filt]['fluxcal'].append(row['FLUXCAL'])
        data[filt]['fluxcalerr'].append(row['FLUXCALERR'])
        data[filt]['photflag'].append(row['PHOTFLAG'])

    return data


def parse_header_table(table, index=0):
    """
    Retrieve metadata from header file

    Parameters
    ----------
    path : str
        Pointer to the header FITS file
    index : int, optional
        Row number in the header table (default is 0)
        Corresponds to the SN id in this exposure.

    Returns
    -------
    header : dict
        stripped version of the header
    rows : slice
        range of rows in the corresponding PHOT file
    istarget : bool
        boolean flag to distinguish between train/test sample

    """
    head = table[index]

    header = {}
    # SN ID
    header['snid'] = int(head['SNID'])
    # Redshift
    header['z'] = head['SIM_REDSHIFT_HOST']
    header['host_specz'] = head['HOSTGAL_SPECZ']
    header['host_photoz'] = head['HOSTGAL_PHOTOZ']
    # Type
    header['type'] = head['SIM_TYPE_INDEX']
    # Peak MJD
    header['pkmjd'] = head['SIM_PEAKMJD']
    # Libid
    header['libid'] = head['SIM_LIBID']

    # Peak magnitudes
    for filt in LSST_FILTERS:
        header['pkmag_%s' % filt] = head['SIM_PEAKMAG_%s' % filt]

    # Add all the SIM_ info for validation purposes
    simkeys = [colname
               for colname in head.colnames
               if colname.startswith("SIM_")]
    for key in simkeys:
        header[key] = head[key]

    # Corresponding rows in the photometric file
    rows = slice(head['PTROBS_MIN'] - 1, head['PTROBS_MAX'] - 1)

    return header, rows


def parse_lsst_model(dirpath, maxitems=None):
    """
    Parse all files in the provided directory and save relevant
    information in a dictionary saved back to disk (and compressed)

    Parameters
    ----------
    dirpath : Path
        path to a LSST simulated FITS directory
    destination : str
        path to save the output files
    save : bool, optional
        save the result into a compressed pickle file (default is False)

    Returns
    -------
    dataframe or tuple of dataframes

    """
    # List all header files in the given directory
    header_files = dirpath.glob('*HEAD.FITS*')

    ctr = 0

    datadict = {}
    for hfile in header_files:
        pfile = hfile.as_posix().replace('HEAD', 'PHOT')
        htable = Table.read(hfile, format='fits')
        ptable = Table.read(pfile, format='fits', memmap=True)
        for idx in range(len(htable)):
            header, rows = parse_header_table(htable, idx)
            data = parse_phot_table(ptable, rows)
            # Add the header info to the light curve data
            data['header'] = header
            # Use the SN id to index the dictionary
            datadict[header['snid']] = data
            ctr += 1
            if maxitems is not None:
                if ctr >= maxitems:
                    break
        if maxitems is not None:
            if ctr >= maxitems:
                break
    message = 'Done reading {:n} objects'.format(ctr)
    print(message)
    return datadict
