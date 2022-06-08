from bin.autorecon1_without_skullstripping import *
from bin.move_T1 import *
from utils import *
import os
import glob
import shutil

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

def autorecon1_without_skullstripping(data_dir='/scratch/connectome/dhkdgmlghks/lesion_tract_pipeline/BIDS/Data_Testing_300', save_dir='/Users/wangheehwan/Desktop/lesion_tract_pipeline/BIDS/Data_Testing_300/../lesion_tract', n_process=1):
    os.chdir(data_dir)
    # ================
    wf = pe.Workflow(name="autorecon1_without_skullstripping")
    wf.base_dir = os.path.join(save_dir, 'workdir')

    # ================
    subject_list = glob.glob('*')

    # ================
    inputspec = pe.Node(
        interface=IdentityInterface(['subject_id']), name="inputspec")
    inputspec.iterables = ("subject_id", subject_list)

    datasource = pe.Node(
        interface=nio.DataGrabber(infields=['subject_id'], outfields=['struct']),
        name='datasource')
    datasource.inputs.base_directory = data_dir
    datasource.inputs.template = '%s/ses-1/anat/%s_%s_%s.nii.gz'
    datasource.inputs.template_args = dict(struct=[['subject_id', 'subject_id','ses-1','T1w']])
    datasource.inputs.subject_id = subject_list
    datasource.inputs.sort_filelist = True

    wf.connect(inputspec, 'subject_id', datasource, 'subject_id')

    # ================
    autorecon1 = create_AutoRecon1_workflow()
    autorecon1.inputs.inputspec.subjects_dir = save_dir
    wf.connect(inputspec, 'plugin_args', autorecon1, 'inputspec.plugin_args')
    wf.connect(inputspec, 'subject_id', autorecon1, 'inputspec.subject_id')
    wf.connect(datasource, 'struct', autorecon1, 'inputspec.T1_files')

    # ================
    if n_process > 1: 
        wf.run("MultiProc", plugin_args={'n_procs': n_process})
    else:
        wf.run()


# ================# ================# ================# ================# ================# ================# ================# ================# ================# ================# ================# ================# ================
def mgz2nifti(save_dir='/scratch/connectome/dhkdgmlghks/lesion_tract_pipeline/BIDS/lesion_tract', n_process=1):
    if not os.path.isdir(save_dir):
        raise ValueError("WARPING RAW T1 IMAGE TO TALAIRACH SPACE SHOULD BE DONE FIRST. ")

    # ================
    wf = pe.Workflow(name="convert_mgz_to_nifti")
    wf.base_dir = os.path.join(save_dir, 'workdir')
    
    if os.path.exists(wf.base_dir): # reomve working directory used for former stage
        shutil.rmtree(wf.base_dir)


    # ================
    os.chdir(save_dir)
    subject_list = glob.glob('*')



    # ================
    inputspec = pe.Node(
        interface=IdentityInterface(['subject_id']), name="inputspec")
    inputspec.iterables = ("subject_id", subject_list)

    datasource = pe.Node(
        interface=nio.DataGrabber(infields=['subject_id'], outfields=['struct']),
        name='datasource')
    datasource.inputs.base_directory = save_dir
    datasource.inputs.template = '%s/mri/%s.mgz'
    datasource.inputs.template_args = dict(struct=[['subject_id','T1']])
    datasource.inputs.subject_id = subject_list
    datasource.inputs.sort_filelist = True

    wf.connect(inputspec, 'subject_id', datasource, 'subject_id')


    # ================
    mgz2nifti_wf = create_mgz2nifti_workflow()
    mgz2nifti_wf.inputs.inputspec.save_dir = save_dir
    wf.connect(inputspec, 'subject_id', mgz2nifti_wf, 'inputspec.subject_id')
    wf.connect(datasource, 'struct', mgz2nifti_wf, 'inputspec.T1_files')
    wf.connect(inputspec, 'plugin_args', mgz2nifti_wf, 'inputspec.plugin_args')

    # ================
    #wf.run("MultiProc", plugin_args={'n_procs': 1})

    wf.run()


if __name__ == '__main__':
    data_dir = '/Users/wangheehwan/Desktop/lesion_tract_pipeline/BIDS/Data_Testing_300'
    
    os.chdir(data_dir)
    save_dir = os.path.abspath('../lesion_tract')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # warping to Talairach space 
    autorecon1_without_skullstripping(data_dir = data_dir, save_dir = save_dir, n_process=1) 
    remove_working_dir(save_dir) # reomve working directory used for former stage
    
    # Change image type from mgz to nifti
    mgz2nifti(save_dir = save_dir, n_process=1)
    remove_working_dir(save_dir)
