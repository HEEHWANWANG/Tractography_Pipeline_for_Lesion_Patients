#! /usr/bin/env python3
import os
import glob

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from niflow.nipype1.workflows.smri.freesurfer.autorecon1 import *
#from niflow.nipype1.workflows.smri.freesurfer.autorecon1 import *
from nipype.interfaces.freesurfer import Info
from nipype.interfaces import utility as niu
from nipype.interfaces.io import DataSink

import logging



from nipype.pipeline import engine as pe
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.freesurfer import *
from niflow.nipype1.workflows.smri.freesurfer.utils import copy_file

def create_AutoRecon1_without_skullstripping(name="AutoRecon1",
                      longitudinal=False,
                      distance=None,
                      custom_atlas=None,
                      plugin_args=None,
                      shrink=None,
                      stop=None,
                      fsvernum=5.3):
    """Creates the AutoRecon1 workflow in nipype.
    Inputs::
           inputspec.T1_files : T1 files (mandatory)
           inputspec.T2_file : T2 file (optional)
           inputspec.FLAIR_file : FLAIR file (optional)
           inputspec.cw256 : Conform inputs to 256 FOV (optional)
           inputspec.num_threads: Number of threads to use with EM Register (default=1)
    Outpus::
    """
    ar1_wf = pe.Workflow(name=name)
    inputspec = pe.Node(
        interface=IdentityInterface(fields=[
            'T1_files', 'T2_file', 'FLAIR_file', 'cw256', 'num_threads'
        ]),
        run_without_submitting=True,
        name='inputspec')

    if not longitudinal:
        # single session processing
        verify_inputs = pe.Node(
            Function(["T1_files", "cw256"],
                     ["T1_files", "cw256", "resample_type", "origvol_names"],
                     checkT1s),
            name="Check_T1s")
        ar1_wf.connect([(inputspec, verify_inputs, [('T1_files', 'T1_files'),
                                                    ('cw256', 'cw256')])])

        # T1 image preparation
        # For all T1's mri_convert ${InputVol} ${out_file}
        T1_image_preparation = pe.MapNode(
            MRIConvert(), iterfield=['in_file', 'out_file'], name="T1_prep")

        ar1_wf.connect([
            (verify_inputs, T1_image_preparation,
             [('T1_files', 'in_file'), ('origvol_names', 'out_file')]),
        ])

        def convert_modalities(in_file=None, out_file=None):
            """Returns an undefined output if the in_file is not defined"""
            from nipype.interfaces.freesurfer import MRIConvert
            import os
            if in_file:
                convert = MRIConvert()
                convert.inputs.in_file = in_file
                convert.inputs.out_file = out_file
                convert.inputs.no_scale = True
                out = convert.run()
                out_file = os.path.abspath(out.outputs.out_file)
            return out_file

        T2_convert = pe.Node(
            Function(['in_file', 'out_file'], ['out_file'],
                     convert_modalities),
            name="T2_Convert")
        T2_convert.inputs.out_file = 'T2raw.mgz'
        ar1_wf.connect([(inputspec, T2_convert, [('T2_file', 'in_file')])])

        FLAIR_convert = pe.Node(
            Function(['in_file', 'out_file'], ['out_file'],
                     convert_modalities),
            name="FLAIR_Convert")
        FLAIR_convert.inputs.out_file = 'FLAIRraw.mgz'
        ar1_wf.connect([(inputspec, FLAIR_convert, [('FLAIR_file',
                                                     'in_file')])])
    else:
        # longitudinal inputs
        inputspec = pe.Node(
            interface=IdentityInterface(fields=[
                'T1_files', 'iscales', 'ltas', 'subj_to_template_lta',
                'template_talairach_xfm', 'template_brainmask'
            ]),
            run_without_submitting=True,
            name='inputspec')

        def output_names(T1_files):
            """Create file names that are dependent on the number of T1 inputs"""
            iscale_names = list()
            lta_names = list()
            for i, t1 in enumerate(T1_files):
                # assign an input number
                file_num = str(i + 1)
                while len(file_num) < 3:
                    file_num = '0' + file_num
                iscale_names.append("{0}-iscale.txt".format(file_num))
                lta_names.append("{0}.lta".format(file_num))
            return iscale_names, lta_names

        filenames = pe.Node(
            Function(['T1_files'], ['iscale_names', 'lta_names'],
                     output_names),
            name="Longitudinal_Filenames")
        ar1_wf.connect([(inputspec, filenames, [('T1_files', 'T1_files')])])

        copy_ltas = pe.MapNode(
            Function(['in_file', 'out_file'], ['out_file'], copy_file),
            iterfield=['in_file', 'out_file'],
            name='Copy_ltas')
        ar1_wf.connect([(inputspec, copy_ltas, [('ltas', 'in_file')]),
                        (filenames, copy_ltas, [('lta_names', 'out_file')])])

        copy_iscales = pe.MapNode(
            Function(['in_file', 'out_file'], ['out_file'], copy_file),
            iterfield=['in_file', 'out_file'],
            name='Copy_iscales')
        ar1_wf.connect([(inputspec, copy_iscales, [('iscales', 'in_file')]),
                        (filenames, copy_iscales, [('iscale_names',
                                                    'out_file')])])

        concatenate_lta = pe.MapNode(
            ConcatenateLTA(), iterfield=['in_file'], name="Concatenate_ltas")
        ar1_wf.connect([(copy_ltas, concatenate_lta, [('out_file',
                                                       'in_file')]),
                        (inputspec, concatenate_lta, [('subj_to_template_lta',
                                                       'subj_to_base')])])

    # Motion Correction
    """
    When there are multiple source volumes, this step will correct for small
    motions between them and then average them together.  The output of the
    motion corrected average is mri/rawavg.mgz which is then conformed to
    255 cubed char images (1mm isotropic voxels) in mri/orig.mgz.
    """

    def createTemplate(in_files, out_file):
        import os
        import shutil
        if len(in_files) == 1:
            # if only 1 T1 scan given, no need to run RobustTemplate
            print(
                "WARNING: only one run found. This is OK, but motion correction "
                +
                "cannot be performed on one run, so I'll copy the run to rawavg "
                + "and continue.")
            shutil.copyfile(in_files[0], out_file)
            intensity_scales = None
            transforms = None
        else:
            from nipype.interfaces.freesurfer import RobustTemplate
            # if multiple T1 scans are given run RobustTemplate
            intensity_scales = [
                os.path.basename(f.replace('.mgz', '-iscale.txt'))
                for f in in_files
            ]
            transforms = [
                os.path.basename(f.replace('.mgz', '.lta')) for f in in_files
            ]
            robtemp = RobustTemplate()
            robtemp.inputs.in_files = in_files
            robtemp.inputs.average_metric = 'median'
            robtemp.inputs.out_file = out_file
            robtemp.inputs.no_iteration = True
            robtemp.inputs.fixed_timepoint = True
            robtemp.inputs.auto_detect_sensitivity = True
            robtemp.inputs.initial_timepoint = 1
            robtemp.inputs.scaled_intensity_outputs = intensity_scales
            robtemp.inputs.transform_outputs = transforms
            robtemp.inputs.subsample_threshold = 200
            robtemp.inputs.intensity_scaling = True
            robtemp_result = robtemp.run()
            # collect the outputs from RobustTemplate
            out_file = robtemp_result.outputs.out_file
            intensity_scales = [
                os.path.abspath(f)
                for f in robtemp_result.outputs.scaled_intensity_outputs
            ]
            transforms = [
                os.path.abspath(f)
                for f in robtemp_result.outputs.transform_outputs
            ]
        out_file = os.path.abspath(out_file)
        return out_file, intensity_scales, transforms

    if not longitudinal:
        create_template = pe.Node(
            Function(['in_files', 'out_file'],
                     ['out_file', 'intensity_scales', 'transforms'],
                     createTemplate),
            name="Robust_Template")
        create_template.inputs.out_file = 'rawavg.mgz'
        ar1_wf.connect([(T1_image_preparation, create_template,
                         [('out_file', 'in_files')])])
    else:
        create_template = pe.Node(RobustTemplate(), name="Robust_Template")
        create_template.inputs.average_metric = 'median'
        create_template.inputs.out_file = 'rawavg.mgz'
        create_template.inputs.no_iteration = True
        ar1_wf.connect([(concatenate_lta, create_template,
                         [('out_file', 'initial_transforms')]),
                        (inputSpec, create_template, [('in_t1s', 'in_files')]),
                        (copy_iscales, create_template,
                         [('out_file', 'in_intensity_scales')])])

    # mri_convert
    conform_template = pe.Node(MRIConvert(), name='Conform_Template')
    conform_template.inputs.out_file = 'orig.mgz'
    if not longitudinal:
        conform_template.inputs.conform = True
        ar1_wf.connect([(verify_inputs, conform_template,
                         [('cw256', 'cw256'), ('resample_type',
                                               'resample_type')])])
    else:
        conform_template.inputs.out_datatype = 'uchar'

    ar1_wf.connect([(create_template, conform_template, [('out_file',
                                                          'in_file')])])

    # Talairach
    """
    This computes the affine transform from the orig volume to the MNI305 atlas using Avi Snyders 4dfp
    suite of image registration tools, through a FreeSurfer script called talairach_avi.
    Several of the downstream programs use talairach coordinates as seed points.
    """

    bias_correction = pe.Node(MNIBiasCorrection(), name="Bias_correction")
    bias_correction.inputs.iterations = 1
    bias_correction.inputs.protocol_iterations = 1000
    bias_correction.inputs.distance = distance
    if stop:
        bias_correction.inputs.stop = stop
    if shrink:
        bias_correction.inputs.shrink = shrink
    bias_correction.inputs.no_rescale = True
    bias_correction.inputs.out_file = 'orig_nu.mgz'

    ar1_wf.connect([
        (conform_template, bias_correction, [('out_file', 'in_file')]),
    ])

    if not longitudinal:
        # single session processing
        talairach_avi = pe.Node(TalairachAVI(), name="Compute_Transform")
        if custom_atlas is not None:
            # allows to specify a custom atlas
            talairach_avi.inputs.atlas = custom_atlas
        talairach_avi.inputs.out_file = 'talairach.auto.xfm'
        ar1_wf.connect([(bias_correction, talairach_avi, [('out_file',
                                                           'in_file')])])
    else:
        # longitudinal processing
        # Just copy the template xfm
        talairach_avi = pe.Node(
            Function(['in_file', 'out_file'], ['out_file'], copy_file),
            name='Copy_Template_Transform')
        talairach_avi.inputs.out_file = 'talairach.auto.xfm'

        ar1_wf.connect([(inputspec, talairach_avi, [('template_talairach_xfm',
                                                     'in_file')])])

    copy_transform = pe.Node(
        Function(['in_file', 'out_file'], ['out_file'], copy_file),
        name='Copy_Transform')
    copy_transform.inputs.out_file = 'talairach.xfm'

    ar1_wf.connect([(talairach_avi, copy_transform, [('out_file',
                                                      'in_file')])])

    # In recon-all the talairach.xfm is added to orig.mgz, even though
    # it does not exist yet. This is a compromise to keep from
    # having to change the time stamp of the orig volume after talairaching.
    # Here we are going to add xfm to the header after the xfm has been created.
    # This may mess up the timestamp.

    add_xform_to_orig = pe.Node(
        AddXFormToHeader(), name="Add_Transform_to_Orig")
    add_xform_to_orig.inputs.copy_name = True
    add_xform_to_orig.inputs.out_file = conform_template.inputs.out_file

    ar1_wf.connect(
        [(conform_template, add_xform_to_orig, [('out_file', 'in_file')]),
         (copy_transform, add_xform_to_orig, [('out_file', 'transform')])])

    # This node adds the transform to the orig_nu.mgz file. This step does not
    # exist in the recon-all workflow, because that workflow adds the talairach
    # to the orig.mgz file header before the talairach actually exists.
    add_xform_to_orig_nu = pe.Node(
        AddXFormToHeader(), name="Add_Transform_to_Orig_Nu")
    add_xform_to_orig_nu.inputs.copy_name = True
    add_xform_to_orig_nu.inputs.out_file = bias_correction.inputs.out_file

    ar1_wf.connect(
        [(bias_correction, add_xform_to_orig_nu, [('out_file', 'in_file')]),
         (copy_transform, add_xform_to_orig_nu, [('out_file', 'transform')])])
    
    # check the alignment of the talairach
    # TODO: Figure out how to read output from this node.
    check_alignment = pe.Node(
        CheckTalairachAlignment(), name="Check_Talairach_Alignment")
    check_alignment.inputs.threshold = 0.005
    ar1_wf.connect([
        (copy_transform, check_alignment, [('out_file', 'in_file')]),
    ])



     # intensity correction is performed before normalization
    intensity_correction = pe.Node(
        MNIBiasCorrection(), name="Intensity_Correction")
    intensity_correction.inputs.out_file = 'nu.mgz'
    intensity_correction.inputs.iterations = 2
    ar1_wf.connect([(add_xform_to_orig, intensity_correction,
                     [('out_file', 'in_file')]),
                      (copy_transform, intensity_correction,
                     [('out_file', 'transform')])])

    add_to_header_nu = pe.Node(AddXFormToHeader(), name="Add_XForm_to_NU")
    add_to_header_nu.inputs.copy_name = True
    add_to_header_nu.inputs.out_file = 'nu.mgz'
    ar1_wf.connect([(intensity_correction, add_to_header_nu, [
            ('out_file', 'in_file'),
        ]), (copy_transform, add_to_header_nu, [('out_file', 'transform')])])

    # Intensity Normalization
    # Performs intensity normalization of the orig volume and places the result in mri/T1.mgz.
    # Attempts to correct for fluctuations in intensity that would otherwise make intensity-based
    # segmentation much more difficult. Intensities for all voxels are scaled so that the mean
    # intensity of the white matter is 110.

    mri_normalize = pe.Node(Normalize(), name="Normalize_T1")
    mri_normalize.inputs.gradient = 1
    mri_normalize.inputs.out_file = 'T1.mgz'

    ar1_wf.connect([(add_to_header_nu, mri_normalize, [('out_file',
                                                            'in_file')])])


    ar1_wf.connect([(copy_transform, mri_normalize, [('out_file',
                                                      'transform')])])


    ar1_outputs = [
        'origvols', 't2_raw', 'flair', 'rawavg', 'orig_nu', 'orig',
        'talairach_auto', 'talairach', 't1', 'nu'
    ]



    outputspec = pe.Node(
            IdentityInterface(fields=ar1_outputs + ['nu']), name="outputspec")
    ar1_wf.connect([(add_to_header_nu, outputspec, [('out_file', 'nu')])])


    ar1_wf.connect([
        (T1_image_preparation, outputspec, [('out_file', 'origvols')]),
        (T2_convert, outputspec, [('out_file', 't2_raw')]),
        (FLAIR_convert, outputspec, [('out_file', 'flair')]),
        (create_template, outputspec, [('out_file', 'rawavg')]),
        (add_xform_to_orig, outputspec, [('out_file', 'orig')]),
        (add_xform_to_orig_nu, outputspec, [('out_file', 'orig_nu')]),
        (talairach_avi, outputspec, [('out_file', 'talairach_auto')]),
        (copy_transform, outputspec, [('out_file', 'talairach')]),
        (mri_normalize, outputspec, [('out_file', 't1')]),
    ])
                                            
    return ar1_wf, ar1_outputs



def create_AutoRecon1_workflow(name = 'AutoRecon1'):

    wf = pe.Workflow(name=name)

    inputspec = pe.Node(
        interface=IdentityInterface(fields=[
            'T1_files', 'T2_file', 'FLAIR_file', 'cw256', 'num_threads',
            'subjects_dir', 'subject_id', 'plugin_args'
        ]),
        run_without_submitting=True,
        name='inputspec')


    # ================     
    logger = logging.getLogger('nipype.workflow')

    fs_version_full = Info.version()
    if fs_version_full and ('v6.0' in fs_version_full
                                or 'dev' in fs_version_full):
        # assuming that dev is 6.0
        fsvernum = 6.0
        fs_version = 'v6.0'
        th3 = True
        shrink = 2
        distance = 200  # 3T should be 50
        stop = 0.0001
        exvivo = True
        entorhinal = True
        rb_date = "2014-08-21"
    else:
        # 5.3 is default
        fsvernum = 5.3
        if fs_version_full:
            if 'v5.3' in fs_version_full:
                fs_version = 'v5.3'
            else:
                fs_version = fs_version_full.split('-')[-1]
                logger.info(("Warning: Workflow may not work properly if "
                                "FREESURFER_HOME environmental variable is not "
                                "set or if you are using an older version of "
                                "FreeSurfer"))
        else:
            fs_version = 5.3  # assume version 5.3
        th3 = False
        shrink = None
        distance = 50
        stop = None
        exvivo = False
        entorhinal = False
        rb_date = "2008-03-26"

    logger.info("FreeSurfer Version: {0}".format(fs_version))

    # =================
    ar1_wf, ar1_outputs = create_AutoRecon1_without_skullstripping(
        name = 'Do_AutoRecon1',
        plugin_args=inputspec.plugin_args,
        stop=stop,
        distance=distance,
        shrink=shrink,
        fsvernum=fsvernum)


    wf.connect([(inputspec, ar1_wf, 
            [('T1_files', 'inputspec.T1_files'), ('T2_file', 'inputspec.T2_file'),
            ('FLAIR_file', 'inputspec.FLAIR_file'),
            ('num_threads', 'inputspec.num_threads'), ('cw256', 'inputspec.cw256')]
        )]
    )
    
    # Add ar1_outputs to outputspec
    outputspec = pe.Node(
        niu.IdentityInterface(fields=ar1_outputs, mandatory_inputs=True),
        name="outputspec")


    for field in ar1_outputs:  
        wf.connect([(ar1_wf,outputspec, [('outputspec.' + field, field)])])

    
    # PreDataSink: Switch Transforms to datasinked transfrom
    # The transforms in the header files of orig.mgz, orig_nu.mgz, and nu.mgz
    # are all reference a transform in the cache directory. We need to rewrite the
    # headers to reference the datasinked transform

    # get the filepath to where the transform will be datasinked
    def getDSTransformPath(subjects_dir, subject_id):
        import os
        transform = os.path.join(subjects_dir, subject_id, 'mri', 'transforms',
                                 'talairach.xfm')
        return transform

    dstransform = pe.Node(
        niu.Function(['subjects_dir', 'subject_id'], ['transform'],
                     getDSTransformPath),
        name="PreDataSink_GetTransformPath")

    wf.connect([(inputspec, dstransform,
                       [('subjects_dir', 'subjects_dir'), ('subject_id',
                                                           'subject_id')])])
                                                           
    
    
    # add the data sink transfrom location to the headers
    predatasink_orig = pe.Node(AddXFormToHeader(), name="PreDataSink_Orig")
    predatasink_orig.inputs.copy_name = True
    predatasink_orig.inputs.out_file = 'orig.mgz'
    wf.connect([(outputspec, predatasink_orig, [('orig', 'in_file')]),
                      (dstransform, predatasink_orig, [('transform',
                                                        'transform')])])
                                                      
    predatasink_orig_nu = pe.Node(
        AddXFormToHeader(), name="PreDataSink_Orig_Nu")
    predatasink_orig_nu.inputs.copy_name = True
    predatasink_orig_nu.inputs.out_file = 'orig_nu.mgz'
    wf.connect(
        [(outputspec, predatasink_orig_nu, [('orig_nu', 'in_file')]),
         (dstransform, predatasink_orig_nu, [('transform', 'transform')])])

    predatasink_nu = pe.Node(AddXFormToHeader(), name="PreDataSink_Nu")
    predatasink_nu.inputs.copy_name = True
    predatasink_nu.inputs.out_file = 'nu.mgz'
    
    wf.connect([(outputspec, predatasink_nu, [('nu', 'in_file')]),
                      (dstransform, predatasink_nu, [('transform',
                                                      'transform')])])
    
    
    # Datasink outputs
    datasink = pe.Node(DataSink(), name="DataSink")
    datasink.inputs.parameterization = False

    wf.connect([(inputspec, datasink,
                       [('subjects_dir', 'base_directory'), ('subject_id',
                                                             'container')])])


    # assign datasink inputs
    wf.connect([
        (predatasink_orig, datasink, [('out_file', 'mri.@orig')]),
        (predatasink_orig_nu, datasink, [('out_file', 'mri.@orig_nu')]),
        (predatasink_nu, datasink, [('out_file', 'mri.@nu')]),
        (outputspec, datasink, [
            ('origvols', 'mri.orig'),
            ('t2_raw', 'mri.orig.@t2raw'),
            ('flair', 'mri.orig.@flair'),
            ('rawavg', 'mri.@rawavg'),
            ('talairach_auto', 'mri.transforms.@tal_auto'),
            ('talairach', 'mri.transforms.@tal'),
            ('t1', 'mri.@t1')
     ]),
    ])
    
    # compeltion node
    # since recon-all outputs so many files a completion node is added
    # that will output the subject_id once the workflow has completed
    def completemethod(datasinked_files, subject_id):
        print("recon-all has finished executing for subject: {0}".format(
            subject_id))
        return subject_id

    completion = pe.Node(
        niu.Function(['datasinked_files', 'subject_id'], ['subject_id'],
                     completemethod),
        name="Completion")

    # create a special identity interface for outputing the subject_id

    postds_outputspec = pe.Node(
        niu.IdentityInterface(['subject_id']), name="postdatasink_outputspec")

    wf.connect(
        [(datasink, completion, [('out_file', 'datasinked_files')]),
         (inputspec, completion, [('subject_id', 'subject_id')]),
         (completion, postds_outputspec, [('subject_id', 'subject_id')])])
    

    return wf



"""
# ================
print(os.environ['FREESURFER_HOME'])
os.chdir('/scratch/connectome/dhkdgmlghks')
save_dir = os.path.abspath('lesion_tract_pipeline/lesion_tract')

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)



# ================
wf = pe.Workflow(name="autorecon1_without_skullstripping")
wf.base_dir = os.path.join(save_dir, 'workdir')


# ================
data_dir = os.path.abspath('lesion_tract_pipeline/BIDS/Data_Training_655')
os.chdir(data_dir)
subject_list = glob.glob('*')
subjects_dir = os.path.join(save_dir, 'Data_Training_655')
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
datasource.inputs.template = '%s/ses-1/anat/%s_%s_%s.nii.gz'
datasource.inputs.template_args = dict(struct=[['subject_id', 'subject_id','ses-1','T1w']])
datasource.inputs.subject_id = subject_list
datasource.inputs.sort_filelist = True

wf.connect(inputspec, 'subject_id', datasource, 'subject_id')

# ================
autorecon1 = create_AutoRecon1_workflow()
autorecon1.inputs.inputspec.subjects_dir = subjects_dir
wf.connect(inputspec, 'plugin_args', autorecon1, 'inputspec.plugin_args')
wf.connect(inputspec, 'subject_id', autorecon1, 'inputspec.subject_id')
wf.connect(datasource, 'struct', autorecon1, 'inputspec.T1_files')

# ================
#wf.run("MultiProc", plugin_args={'n_procs': 1})
wf.run()

# ================
import shutil
shutil.rmtree(wf.base_dir)
"""