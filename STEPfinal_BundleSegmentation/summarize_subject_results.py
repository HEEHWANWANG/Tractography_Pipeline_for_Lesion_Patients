#!/bin/python3
import os
import glob 
import argparse

import numpy as np 
import pandas as pd 

import math

# ---------------------------------------------------- Uutil functions ----------------------------------------------------- #
def check_nan(bundle_track: np.array, bundle_track_name: str):
    if not bundle_track.tolist():
        print(Warning("There isn't any estimated track streamlines involved in '%s'. Please check '_%s.tck' file first." % (bundle_track_name, bundle_track_name))) 
        

def read_ascii(file):
    with open(file) as f:
        for line in f:
            array = np.fromstring(line.strip(), dtype=float, sep=" ")
    return array


def get_tract_name(file_name): 
    track_name = os.path.splitext(file_name)[0] # remove ".txt" 
    track_name = track_name[1:]   # remove "_" in the first position of file name 
    return track_name


def save_results(dataframe: pd.DataFrame, subject_name):
    current_dir = os.getcwd() 
    save_file = subject_name + '_summary.csv'
    dataframe.to_csv(os.path.join(current_dir, save_file), na_rep='NaN', index=False)
    print('RESULT IS SAVED')

# ------------------------------------------------------------------------------------------------------------------------- #





# ---------------------------------------------------- main functions ----------------------------------------------------- #
def main(args): 
    # grab the result files from tcksample
    bundle_track_files = glob.glob('*.txt')

    # create result template 
    summary = {'tract_bundle':[], args.metric+'_mean':[], args.metric+'_stdev':[]}

    for bundle_track_file in bundle_track_files:
        bundle_track_name = get_tract_name(bundle_track_file)   # get the name of tract_bundle
        
        bundle_track_file = os.path.join(file_dir, bundle_track_file)   # get the file directory of tcksample results 
        bundle_track = read_ascii(bundle_track_file)    # read the tcksample results as numpy array 

        check_nan(bundle_track, bundle_track_name)      # check whether numpy array is empty or not 
        
        bundle_track_mean = bundle_track.mean()         
        bundle_track_stdev = math.sqrt(bundle_track.var())

        summary['tract_bundle'].append(bundle_track_name)
        summary[args.metric+'_mean'].append(bundle_track_mean)
        summary[args.metric+'_stdev'].append(bundle_track_stdev)

    summary = pd.DataFrame(summary)
    save_results(summary, args.subject) 
# ------------------------------------------------------------------------------------------------------------------------- #





# ------------------------------------------------------- Arguments ------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--file_dir", type=str,required=True,help='Directories containing tract_bundle.tck files should be indicated')
parser.add_argument("--metric", type=str,required=True,help='DTI metric should be indicated')
parser.add_argument("--subject", type=str,required=True,help='The name of subject should be indicated')
args = parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------------- #





if __name__ == "__main__":
    file_dir = os.path.join(args.file_dir, args.metric)
    os.chdir(file_dir)
    main(args)

