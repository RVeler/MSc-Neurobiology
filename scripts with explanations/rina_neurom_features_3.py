import neurom as nm
import numpy
import numpy as np
import os
import os.path
import csv
import sys

# parameters that you could extract from the neuroM, there are only one number and not a list
basic_features = ['number_of_bifurcations', 'number_of_forking_points', 'number_of_neurites', 'number_of_sections',
                   'number_of_terminations', 'total_length']
# the following features are list of numbers and you can take the mean\median\max\min ect...
featureList = ['neurite_lengths',
               'number_of_sections_per_neurite',
               'principal_direction_extents',
               'remote_bifurcation_angles',
               'section_branch_orders',
               'section_lengths',
               'section_path_distances',
               'section_radial_distances']


# the following features are not being handled right now
neuronFeatureList = [
    'trunk_origin_azimuths',
    'trunk_origin_elevations',
    'trunk_section_lengths']


"""
Since not every neuron has both apical dendrite, basel dendrite or an axon, we first verify what does this neuron has.
The result parameter is a simple list of neurite types that we will refer to later on.
"""
def get_neurite_types(nrn):
    ntypes = [];

    try:
        apical_dendrites = nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.apical_dendrite)[0]
        basal_dendrites = nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.basal_dendrite)[0]
        axons = nm.get('number_of_neurites', nrn, neurite_type=nm.NeuriteType.axon)[0]

    except:
        print('error! unable to get neurite data.')

    if apical_dendrites > 0:
        ntypes.append(nm.NeuriteType.apical_dendrite)
    if basal_dendrites > 0:
        ntypes.append(nm.NeuriteType.basal_dendrite)
    if axons > 0:
        ntypes.append(nm.NeuriteType.axon)
    if len(ntypes) == 3:
        ntypes.append(nm.NeuriteType.all)

    return ntypes


"""
This function receives a list of neurite types and returns all the extracted data.
"""

def get_neuron_features(nrn, ntypes):
    
    featuremap = get_default_map()

    for ttype in ntypes:

        for feature in basic_features:
            fdata = nm.get(feature, nrn, neurite_type=ttype)[0]
            ptype = str(ttype).replace(".", "_")
            fieldname = feature + '_' + str(ptype)
            featuremap[fieldname + '_value'] = fdata

        for feature in featureList:
            ptype = str(ttype).replace(".", "_")
            fieldname = feature + '_' + str(ptype)
            # print 'field : ' , fieldname
            try:
                fdata = nm.get(feature, nrn, neurite_type=ttype)


                if isinstance(fdata, numpy.ndarray):
                    fdata = fdata.tolist()

                # print(fdata)

                #featuremap[fieldname + '_length'] = len(fdata)
                featuremap[fieldname + '_mean'] = np.mean(fdata)
                #featuremap[fieldname + '_sum'] = np.sum(fdata)
                #featuremap[fieldname + '_std'] = np.std(fdata)
                #featuremap[fieldname + '_min'] = np.min(fdata)
                #featuremap[fieldname + '_max'] = np.max(fdata)

            except:
                print('unable to get feature.', feature, ' ttype: ', ttype)
                continue
    return featuremap


def get_default_map():

    featuremap  = {}
    all_types = []
    all_types.append(nm.NeuriteType.apical_dendrite)
    all_types.append(nm.NeuriteType.basal_dendrite)
    all_types.append(nm.NeuriteType.axon)
    all_types.append(nm.NeuriteType.all)

    for ttype in all_types:
        for feature in basic_features:
            ptype = str(ttype).replace(".", "_")
            fieldname = feature + '_' + str(ptype)
            featuremap[fieldname + '_value'] = np.nan

        for feature in featureList:
                ptype = str(ttype).replace(".", "_")
                fieldname = feature + '_' + str(ptype)
                #featuremap[fieldname + '_length'] = np.nan
                featuremap[fieldname + '_mean'] = np.nan
                #featuremap[fieldname + '_sum'] = np.nan
                #featuremap[fieldname + '_std'] = np.nan
                #featuremap[fieldname + '_min'] = np.nan
                #featuremap[fieldname + '_max'] = np.nan

    return featuremap


"""
extract data just about a single neuron SWC file
"""
def extract_single_file(filename):
    print('parsing single swc file: ', filename)
    nrn = nm.load_neuron(filename)
    print('succesfully loaded neuron file from: ', filename)

    ntypes = get_neurite_types(nrn)
    featuremap = get_neuron_features(nrn, ntypes)

    return featuremap



""" 
This function receives a directory path and analyzes all the swc files it has.
The output of the analysis is saved according to the file name argument. 
"""
def analyze_swc_files(data_path):
    print ('parsing all swc files from: ', data_path)
    data_path += '/'
    processed_data = []
    # iterate over all the files in the given directory
    for file in os.listdir(data_path):
        # filter just the files of swc type
        if file.endswith(".swc"):
            full_file_name = data_path + file
            neuron_name = file[0:-4]
            print('processing neuron: ', neuron_name)
            try:
                print ('analyzing file:', full_file_name)
                featuremap = extract_single_file(full_file_name)
                # add the neuron name to the feature map
                featuremap['neuron_name'] = neuron_name
                processed_data.append(featuremap)

            except:
                print('unable to load neuron into Neurom. Skipping file: ', full_file_name)
                continue

    return processed_data

def save_processing_to_csv(processed_features,filename):
    with open(filename, 'w') as f:  # Just use 'w' mode in 3.x
        for index, featuremap in enumerate(processed_features):
            w = csv.DictWriter(f, featuremap.keys())
            if index == 0:
                w.writeheader()
            w.writerow(featuremap)

    print('saved data to: ', filename)



def savetoCSV(featuremap,filename):

    with open(filename, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, featuremap.keys())
        w.writeheader()
        w.writerow(featuremap)

    print('saved data to: ', filename)


if __name__ == '__main__':
    # data_path - the adress of the folders which has the swc files
    data_path = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat In\trial'
    print('SWC feature analysis starting. data path:', data_path)

    # check if the file exists alerady
    # output_file - the adress and the name.csv you want the matrix will be saved.
    output_file = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat In\trial\ratIn.csv'
    if os.path.isfile(output_file):
        print('Error!!! file already exists!')
        sys.exit()

    processed_data = analyze_swc_files(data_path)
    save_processing_to_csv(processed_data, output_file)

    # singl_file = '/Users/user/brain/python1/data/rina1/010615B.swc'
    # featuremap = extract_single_file(singl_file)
    # csv_file = 'csv/' + 'test_1'
    # savetoCSV(featuremap, csv_file)
    print('done.')
