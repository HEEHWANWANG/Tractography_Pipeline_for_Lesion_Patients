import os
import glob 
import argparse

import pandas as pd 

# ---------------------------------------------------- Uutil functions ----------------------------------------------------- #
def get_subject_result(work_dir, subject_name, metric='count'): 
    
    
    if metric == 'count': 
        subject_file = subject_name + '_summary.txt'
        subject_file = os.path.join(*[work_dir,subject_name, 'tract_bundle',subject_file])
        subject_file = pd.read_table(subject_file, sep=" ", header=None)
        subject_data = pd.DataFrame([subject_file[1].values], columns=subject_file[0].values)
    else:
        subject_file = subject_name + '_summary.csv'
        subject_file = os.path.join(*[work_dir,subject_name, 'tract_bundle', metric, subject_file])
        subject_file = pd.read_csv(subject_file)
        subject_data = pd.DataFrame([subject_file[metric+'_mean'].values], columns=subject_file['tract_bundle'].values)
        
    return subject_data 

def save_result(dataframe: pd.DataFrame, metric, args):
    file_name = args.group_name + '_summary_' + metric + '.csv'
    file_name = os.path.join(args.work_dir, file_name) 
    dataframe.to_csv(file_name, na_rep='NaN')
    print('SUMMARIZING RESULTS of %s FOR EACH SUBJECT IS DONE.\n' % metric)
    
# ------------------------------------------------------------------------------------------------------------------------- #





# ---------------------------------------------------- main functions ----------------------------------------------------- #
def main(args): 
    for metric in args.metric_list: 
        summary_result = pd.DataFrame()
        for subject_name in args.subject_list:
            subject_data = get_subject_result(args.work_dir, subject_name, metric)
            summary_result = pd.concat([summary_result, subject_data], ignore_index=True)
        summary_result.index = args.subject_list
        save_result(summary_result, metric, args)
# ------------------------------------------------------------------------------------------------------------------------- #





# ------------------------------------------------------- Arguments ------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str,required=True,help="Directories containing all subject directories (group directories) containing tract_bundle.tck files should be indicated")
parser.add_argument("--group_name", type=str,required=True,help='Directories containing tract_bundle.tck files should be indicated')
parser.add_argument("--subject_list", type=str, nargs='*',required=True,help='The name of subject should be indicated')
parser.add_argument("--metric_list", type=str, nargs='*',required=True,help='DTI metric should be indicated')
args = parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------------- #





if __name__ == "__main__":
    os.chdir(args.work_dir)
    main(args)