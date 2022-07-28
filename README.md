# RadiomicsTexture_AlzheimerAndMCI
## Description
The use of 3D radiomics features used to study the differences among Healthy Normal (HN), Mild Cognitive Impairment (MCI), and Alzheimer's (AD).

Four types of texture descriptors are used in this work: Gray-Level Co-Occurrence Matrix (GLCM), Gray-level Run Length Matrix (GLRLM), Gray-level Size Zone Matrix (GLSZM), Neighbouring Gray Tone Difference Matrix (NGTDM).

We perform pairwise comparisons between stages, thus generating the following scenarios: HN *vs* MCI, HN *vs* AD, and MCI *vs* AD.

## Code information 
For this preliminary repository:
The code is disclosed to the scientific community to promote reproducibility.
* We are not able to share the T1w Magnetic Resonance Images from the patients, or the texture descriptor matrices, due to an agreement signed with the ADNI association (https://adni.loni.usc.edu/).
* On the other hand, we can share the p-values from each case. Only the HN *vs* AD comparison is share for now, since the paper has been submitted. Once the work is evaluated by the peers, the other two cases will be shared.
* Each case has a Stage1vStage2_p_value.csv and Stage1vStage2_mean_values.csv, corresponding to the uncorrected p-values and the average comparison between the two stages, respectively.
* The following files will be generated after running the code: 

File | Description
| :---: | :---: 
Stage1vStage2_pvalues_part.pdf | this file corresponds to the uncorrected pvalue values 
Stage1vStage2_pvalues_uncorrected.pdf | this image shows the uncorrected thresholded p-values<0.05
Stage1vStage2_pvalues_corrected.pdf | this image shows the corrected thresholded p-values<0.05 (white, blue and yellow colors corresponds to bilateral, left hemisphere, and right hemisphere regions, respectively).
Stage1vStage2_descriptor_pval_corrected.pdf |  this image illustrates the most reliable descriptors in the comparison
Stage1vStage2_descriptor.csv | this csv represents the values from each descriptor.

The requirements.txt file contains the versions of the packages, but to facilitate, follow the step-by-step provided below:

conda create -n py_36 python=3.6.10

conda activate py_36

Package      | code
|:---:        | :---:
Stats Models | conda install -c anaconda statsmodels
ResearchPy   | pip install researchpy
Numpy        | conda install numpy
Pandas       | conda install -c anaconda pandas
Matplotlib   | conda install -c conda-forge matplotlib
Seaborn      | conda install -c anaconda seaborn

In case the above codes don't, try pip install __name__ instead.
Probably you won't have issues with version, but in any case, check the file requirements.txt

As soon as this work is published, we will share the remaining csv files.

## How to run

Please open the file calculate_p_values_and_plots.py and change the line base_dir to your own repository, in case all the files are in the same folder (e.g., if you clone this rep) put '' in the field.
