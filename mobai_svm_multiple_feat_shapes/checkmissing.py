import numpy as np 
import glob
import os 


# Check missing from morphed (i.e. missing ID for both morphs, meaning that we cannot get a probe image...)
def check_missing(): 
    morph_path = '../Data_Morphed/'
    bonafide_path = '../Data_Bonafide/'
    data_sets = ['AGE', 'FERET', 'FRGC', 'TUF']
    
    missing = {ds_i: [] for ds_i in data_sets}
    for ds_i in data_sets: 
        for morphed_im in glob.glob(os.path.join(morph_path, ds_i, '1_training_set') + '/*/*'):
            # Get IDs: 
            if ds_i == 'AGE': 
                _, id1, _, id2 = os.path.split(morphed_im)[-1].split('_')[:4]
                male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                check_id1 = os.path.exists(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1))
                check_id2 = os.path.exists(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2))
                if not check_id1 and not check_id2:
                    missing[ds_i].append(morphed_im)

            elif ds_i == 'FERET':
                _, id1, _, id2 = os.path.split(morphed_im)[-1].split('_')[:4]
                male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1) + '_*'))
                check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2) + '_*'))
                if check_id1 == 0 and check_id2 == 0:
                    missing[ds_i].append(morphed_im)

            elif ds_i == 'FRGC':
                _, id1, _, id2 = os.path.split(morphed_im)[-1].replace('d', '_').split('_')[:4]
                male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1) + '_*'))
                check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2) + '_*'))
                if check_id1 == 0 and check_id2 == 0:
                    missing[ds_i].append(morphed_im)                

            elif ds_i == 'TUF':
                _, id1, id2, _ = os.path.split(morphed_im)[-1].split('_')[:4]
                male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '2_for_bonafide', '1_training_set', male_female, id1) + '_*'))
                check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '2_for_bonafide', '1_training_set', male_female, id2) + '_*'))
                if check_id1 == 0 and check_id2 == 0:
                    missing[ds_i].append(morphed_im)       


    return missing

def check_numbers(): 
    morph_path = '../Data_Morphed/'
    bonafide_path = '../Data_Bonafide/'
    data_sets = ['AGE', 'FERET', 'FRGC', 'TUF']
    sub_ds = ['1_training_set', '3_test_set']
    subsub_ds = ['1_male_source', '2_female_source']
    
    lengths = {d: {dd: {ddd : [] for ddd in subsub_ds} for dd in sub_ds} for d in data_sets}  # Store number of files in folders
    not_morph = {d: {dd: {ddd : [] for ddd in subsub_ds} for dd in sub_ds} for d in data_sets}  # Store the paths for bonafide images that don't have morphs...
    
    for ds_i in data_sets: 
        for ds_type in sub_ds: 
            for gen_type in subsub_ds:
                for morphed_im in glob.glob(os.path.join(morph_path, ds_i, '1_training_set') + '/*/*'):
                    # Get IDs: 
                    if ds_i == 'AGE': 
                        _, id1, _, id2 = os.path.split(morphed_im)[-1].split('_')[:4]
                        male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                        check_id1 = os.path.exists(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1))
                        check_id2 = os.path.exists(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2))
                        if not check_id1 and not check_id2:
                            missing[ds_i].append(morphed_im)

                    elif ds_i == 'FERET':
                        _, id1, _, id2 = os.path.split(morphed_im)[-1].split('_')[:4]
                        male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                        check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1) + '_*'))
                        check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2) + '_*'))
                        if check_id1 == 0 and check_id2 == 0:
                            missing[ds_i].append(morphed_im)

                    elif ds_i == 'FRGC':
                        _, id1, _, id2 = os.path.split(morphed_im)[-1].replace('d', '_').split('_')[:4]
                        male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                        check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id1) + '_*'))
                        check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '1_training_set', male_female, id2) + '_*'))
                        if check_id1 == 0 and check_id2 == 0:
                            missing[ds_i].append(morphed_im)                

                    elif ds_i == 'TUF':
                        _, id1, id2, _ = os.path.split(morphed_im)[-1].split('_')[:4]
                        male_female = os.path.split(os.path.split(morphed_im)[0])[-1]
                        check_id1 = len(glob.glob(os.path.join(bonafide_path, ds_i, '2_for_bonafide', '1_training_set', male_female, id1) + '_*'))
                        check_id2 = len(glob.glob(os.path.join(bonafide_path, ds_i, '2_for_bonafide', '1_training_set', male_female, id2) + '_*'))
                        if check_id1 == 0 and check_id2 == 0:
                            missing[ds_i].append(morphed_im)

        

def check_num():
    morph_path = '../Feature_Morphed/'
    bonafide_path = '../Feature_Bonafide/'

    datasets = ['AGE', 'FERET', 'FRGC', 'TUF']
    subsets = ['1_training_set', '3_test_set']
    num_samples_bona = {d: {s: 0 for s in subsets} for d in datasets}
    num_samples_morph = {d: {s: 0 for s in subsets} for d in datasets}

    for ds_i in datasets: 
        for sub_i in subsets: 
            samples_m = len(glob.glob(
                os.path.join(morph_path, ds_i, sub_i) + '/*/ref_*'
            ))
            samples_b = len(glob.glob(
                os.path.join(bonafide_path, ds_i, sub_i) + '/*/ref_*'
            ))
            num_samples_morph[ds_i][sub_i] = samples_m
            num_samples_bona[ds_i][sub_i] = samples_b

    return num_samples_bona, num_samples_morph


def get_num_bona(): 
    bonafide_path = '../Data_Bonafide'
    datasets = ['AGE', 'FERET', 'FRGC', 'TUF']

if __name__ == '__main__':
    check_missing()
    # check_numbers()