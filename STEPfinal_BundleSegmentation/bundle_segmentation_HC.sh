#!/bin/bash
source ~/.bash_profile

root_dir=${PWD}

subj_list=( "sub-HC1" "sub-HC2" "sub-HC3" )
group_name="HC"
group_dir="/bundle_segmentation/${group_name}"
mkdir "${group_dir}/bundle_segmentation"

qsiprep_dir="${group_dir}/qsiprep"
qsirecon_dir="${group_dir}/qsirecon"
work_dir="${group_dir}/bundle_segmentation"

#####run bundle segmentation and summarize results for each subject
for sub in ${subj_list[@]}
do  
    #####making directories
    mkdir "${work_dir}/${sub}"
    mkdir "${work_dir}/${sub}/DTI"
    mkdir "${work_dir}/${sub}/tract_bundle"
    
    #####making DTI images 
    #making DTI tensor image
    cd "${work_dir}/${sub}/DTI" 
    echo `dwi2tensor -force ${qsiprep_dir}/${sub}/dwi/${sub}_space-T1w_desc-preproc_dwi.nii.gz -fslgrad ${qsiprep_dir}/${sub}/dwi/${sub}_space-T1w_desc-preproc_dwi.bvec ${qsiprep_dir}/${sub}/dwi/${sub}_space-T1w_desc-preproc_dwi.bval -mask  ${qsiprep_dir}/${sub}/dwi/${sub}_space-T1w_desc-brain_mask.nii.gz ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_tensor.nii.gz`

    #making FA image 
    echo `tensor2metric -force ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_tensor.nii.gz -fa ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_DTI_FA.nii.gz`

    #making MD image
    echo `tensor2metric -force ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_tensor.nii.gz -adc ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_DTI_MD.nii.gz`

    #removing DTI tensor image (because it is unnecessary)
    rm "${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_tensor.nii.gz"

    #####run bundle segmentation
    cd "${work_dir}/${sub}/tract_bundle"
    mkdir "${work_dir}/${sub}/tract_bundle/${sub}" #It is weird, but -outbase should be the existing directory.
    echo `03_auto_tracts.sh -tck ${qsirecon_dir}/${sub}/dwi/${sub}_space-T1w_desc-preproc_desc-tracks_ifod2.tck -outbase ${work_dir}/${sub}/tract_bundle/${sub}/ -mask ${qsiprep_dir}/${sub}/dwi/${sub}_space-T1w_desc-brain_mask.nii.gz -fa ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_DTI_FA.nii.gz -weights ${qsirecon_dir}/${sub}/dwi/${sub}_space-T1w_desc-preproc_desc-siftweights_ifod2.csv > ${work_dir}/${sub}/tract_bundle/log.txt`
    rm -r "${work_dir}/${sub}/tract_bundle/${sub}" #It is weird, but -outbase should be the existing directory.

    #####tck sample
    #tcksample for each tract bundle 
    cd "${work_dir}/${sub}/tract_bundle"
    #create directoy for the result files from tcksample with FA image 
    mkdir "${work_dir}/${sub}/tract_bundle/FA"
    #create directoy for the result files from tcksample with MD image 
    mkdir "${work_dir}/${sub}/tract_bundle/MD"

    for bundle_track in *.tck
    do 
        file_name_len=`echo $i | awk '{print length}'`
        file_name_len=$(($file_name_len - 4))
        subj_name_len=`echo $sub | awk '{print length}'`
        bundle_track_name=${bundle_track:subj_name_len:$file_name_len} #removing subject prefix and .tck from file names 
        #tcksample with FA image for each tract_bundle
        echo `tcksample -force ${work_dir}/${sub}/tract_bundle/${bundle_track} ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_DTI_FA.nii.gz ${work_dir}/${sub}/tract_bundle/FA/${bundle_track_name}.txt`
        #tcksample with MD image for each tract_bundle
        echo `tcksample -force ${work_dir}/${sub}/tract_bundle/${bundle_track} ${work_dir}/${sub}/DTI/${sub}_space-T1w_desc-mr_DTI_MD.nii.gz ${work_dir}/${sub}/tract_bundle/MD/${bundle_track_name}.txt`
    done
    
    #####summaize subject's result
    #summarize subject results of FA 
    echo `python3 ${root_dir}/summarize_subject_results.py --file_dir ${work_dir}/${sub}/tract_bundle --metric FA --subject ${sub}` 
    #summarize subject results of MD 
    echo `python3 ${root_dir}/summarize_subject_results.py --file_dir ${work_dir}/${sub}/tract_bundle --metric MD --subject ${sub}` 
done

#####summarize results of all subjects 
cd ${root_dir}
metric_list=( "count" "FA" "MD" )
echo `python3 ${root_dir}/summarize_all_results.py --work_dir ${work_dir} --group_name ${group_name} --subject_list ${subj_list[@]} --metric ${metric_list[@]}`
