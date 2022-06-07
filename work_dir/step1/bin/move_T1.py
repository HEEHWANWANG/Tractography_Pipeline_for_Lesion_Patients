#!/usr/bin/env python3
import os
import glob

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.interfaces.utility import Function,IdentityInterface




"""
print(os.environ['FREESURFER_HOME'])
os.chdir('/Users/wangheehwan/Desktop')
save_dir = os.path.abspath('lesion_tract_pipeline/lesion_tract')

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# ================
wf = pe.Workflow(name="convert_mgz_to_nifti")
wf.base_dir = os.path.join(save_dir, 'workdir')


# ================
data_dir = os.path.abspath('lesion_tract_pipeline/BIDS/Data_Testing_300')
os.chdir(data_dir)
subject_list = glob.glob('*')
subjects_dir = os.path.join(save_dir, 'Data_Testing_300')
if not os.path.exists(subjects_dir):
    os.mkdir(subjects_dir)


# ================
inputspec = pe.Node(
    interface=IdentityInterface(['subject_id']), name="inputspec")
inputspec.iterables = ("subject_id", subject_list)

datasource = pe.Node(
    interface=nio.DataGrabber(infields=['subject_id'], outfields=['struct']),
    name='datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '%s/mri/%s.mgz'
datasource.inputs.template_args = dict(struct=[['subject_id','T1']])
datasource.inputs.subject_id = subject_list
datasource.inputs.sort_filelist = True

wf.connect(inputspec, 'subject_id', datasource, 'subject_id')
"""
# ================

def create_mgz2nifti_workflow(name='mgz2nifti_workflow'):
    wf = pe.Workflow(name=name)

    inputspec = pe.Node(
        interface=IdentityInterface(fields=[
            'T1_files', 'num_threads',
            'save_dir', 'subject_id', 'plugin_args'
        ]),
        run_without_submitting=True,
        name='inputspec')
    
    def convert_to_nifti(in_file=None, out_file=None, save_dir=None, subject_id=None):
        #Returns an undefined output if the in_file is not defined
        from nipype.interfaces.freesurfer import MRIConvert
        import os
        import copy
        if in_file:
            convert = MRIConvert()
            convert.inputs.in_file = in_file
            convert.inputs.out_file = os.path.join(save_dir, '%s/%s_T1w.nii.gz' % (subject_id, subject_id))
            convert.inputs.out_type = 'niigz'
            out = convert.run()

        return out.outputs.out_file
    
    mgz2nifti = pe.Node(
        Function(['in_file', 'out_file','save_dir', 'subject_id'], ['out_file'],
                        convert_to_nifti),
        name="mgz2nifti")

#wf.connect([(datasource, mgz2nifti, [('struct', 'in_file'), ('subject_id', 'subject_id')])])
    wf.connect([(inputspec, mgz2nifti, [('T1_files', 'in_file'), ('save_dir', 'save_dir'), ('subject_id','subject_id')])])
    
    return wf 
"""
# ================
mgz2nifti_wf = create_mgz2nifti_workflow()
mgz2nifti_wf.inputs.inputspec.save_dir = subjects_dir
wf.connect(inputspec, 'subject_id', mgz2nifti_wf, 'inputspec.subject_id')
wf.connect(datasource, 'struct', mgz2nifti_wf, 'inputspec.T1_files')
wf.connect(inputspec, 'plugin_args', mgz2nifti_wf, 'inputspec.plugin_args')

# ================
#wf.run("MultiProc", plugin_args={'n_procs': 1})

wf.run()

# ================
import shutil
shutil.rmtree(wf.base_dir)
"""