import os
import shutil
import glob

def remove_working_dir(save_dir): 
    shutil.rmtree(os.path.join(save_dir, 'workdir'))


def check_T1(data_dir, save_dir):
    os.chdir(data_dir)
    subject_list = glob.glob('*')

    missing_subject = []
    for subject in subject_list:
        T1_file = '%s_T1w.nii.gz' % subject
        T1_file = os.path.join(*[save_dir, subject, T1_file])  
        if os.path.isfile(T1_file) == False:
            missing_subject.append(subject)

    print('FOLLOWING SUBJECTS DOES NOT HAVE **Talairach space warpped T1 nifti images**. PLZ CHECK RAW IMAGES \n {}'.format(missing_subject))
    
