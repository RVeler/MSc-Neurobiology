# MSc-Neurobiology
Pyramidal Neurons Characterization by their Anatomy and Synaptic Plasticity - Thesis 2016

# Scripts – 
  in this folder you can find Matlab and Python scripts for extracting parameters and creating figures from the swc files and matrices.
•	Python scripts - Here is .py files with explanation of the code (it is a little bit messy).

1. "hncUtility.py" – this is a scripts with many functions that has been called by the other scripts. when you see 'hnc.____' it means that we called for a fucntion from this script.
2. "rina_neurom_features_3.py" – In this script you put a folder with relevant swc files and create a matrix of neuroanatomy parameters from neuroM. (the matrix is csv file, but you need to arrange it by hand after creation (to delete irrelevant parameters, to change the order of parameters that will fit the parameters list and to remove blank rows. 
3. "rina_merge_1.py – This script is used to merge the matrix of SSP data with the matrix of neuroanatomy data.
4. "parameters distribution.py" – Script to create the figure of parameters distribution
5. "sp correlation" – Script to create color maps for r and p values between depth and ssp parameters.
6. "ttest for depth" – a file we used to test if there siginificant difference between features by their depth (we used the matrix file "depth_groups" in the folder "our dataset").
7. "remove outliers" – In this script you take a matrix with neuron's data and mark each parameter's outliers with the sign '^'. So you can recognize them in the matrix.
8. "p_value mse map" – In this script you create to colormap of significant\not significant MSE difference between our dataset to another dataset. You should use the matrices that Nitza send us with the p-values differences between the MSE's. 
9. "mse map" – In this script you can create a color map of mse ratio between our dataset and another dataset or a matrix with the values of MSEs and number of neuron in each parameter group (this matrix we send to Nitza in order to calculate the p values between MSEs).
10. "correlation_final2" – This script was designed for many targets.. You can "comments" the unnecessary lines. The script takes a matrix of neuroanatomy and SSP (if there is) dataset and making matrices of the correlations values (p, q, r, r_square etc.). It cans save them and/or use them for creating correlation figures or color maps figures.
