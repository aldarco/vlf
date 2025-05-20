"""
Script to extract txt files from tar.gz files recorded by the RedPitaya.
This script extract the files in the same folder as the targz files.

author: Aldo Arriola @ CONIDA-DIACE
"""


import sys
import os
import tarfile



def decompress(dir):
	fnames = os.listdir(dir)
	nfiles = len(fnames)
	nextracted = 0
	for fname in fnames:
		if fname.endswith('.tar.gz'):
			tar = tarfile.open(dir+'/'+fname, 'r:gz')
			tar.extractall(dir)
			tar.close()
			nextracted += 1

	print(">>> Extracted {} files out of {} files in folder".format(nextracted, nfiles))

if __name__ == '__main__':
	dir = sys.argv[1] # change with the path 
	decompress(dir)
