import numpy as np 
import os 
import glob
from utils import readValuesFromFile
import pickle


class data_loader():
    def __init__(self, morphed_path, bonafide_path, flag='1_training_set', sub_folders=None, feat_shape=(49, 512), construct=True, save_paths=True):
        self.flag = flag
        self.morphed_path = morphed_path
        self.bonafide_path = bonafide_path
        self.sub_folders = sub_folders if sub_folders is not None else ['1_male_source', '2_female_source']
        self.feat_shape = feat_shape
        self.save_ds_path = os.path.join(os.path.dirname(os.path.dirname(self.morphed_path)), 'data_full')
        self.save_path_flag = save_paths

        # Construct the paths to construct dataset
        if construct:
            self.construct_pipeline()

    def find_match(self,this,other):
        print(f'DB1: {len(this)}')
        print(f'DB2: {len(other)}')
        ##remove uneccesary info for the comparison
        thisunp = [[x[0].split("/")[-4:], x[1].split("/")[-4:]] for x in this]
        otherunp = [[x[0].split("/")[-4:], x[1].split("/")[-4:]] for x in other]
        #lists of matching indices of the elements that match
        matchesthis = [1 if x in otherunp else 0 for x in thisunp]
        matchesother = [1 if x in thisunp else 0 for x in otherunp]
        
        #new lists with only the elements that match but containing the full path
        newthis = [this[i] for i in range(len(this)) if matchesthis[i] == 1]
        newother = [other[i] for i in range(len(other)) if matchesother[i] == 1]
        print(len(newthis))
        print(len(newother))
        print(newthis[50])
        print(newother[50])

        return newthis, newother

    def match_cross_dataset(self,other) -> (str, str, str, str):
        # print(self.paths_bonafide)
        bonafied_self, bonafied_other = self.find_match(self.paths_bonafide, other.paths_bonafide)
        morphed_self, morphed_other = self.find_match(self.paths_morphed, other.paths_morphed)
        return morphed_self, bonafied_self, morphed_other, bonafied_other

    def construct_pipeline(self):
        self.paths_morphed = []
        self.paths_bonafide = []

        data_sets = os.listdir(self.morphed_path)
        assert data_sets == os.listdir(self.bonafide_path)
        data_sets = [d for d in data_sets if d != ".DS_Store"]

        # Get morphed paths
        for ds_i in data_sets: 
            for folder_i in self.sub_folders:
                morph_paths = os.path.join(self.morphed_path, ds_i, self.flag, folder_i)
                for i, morph_path in enumerate(glob.glob(morph_paths + '/ref_*.txt')):
                    # Find the two probe images that can be combined with current morphed image: 
                    # print(morph_path)
                    morph_name = os.path.split(morph_path)[-1]
                    probe_id1, probe_id2 = morph_name.replace('.', '_').split('_')[1:3]
                    probe_path1 = os.path.join(os.path.dirname(morph_path), "probe_" + probe_id1 + ".txt")
                    probe_path2 = os.path.join(os.path.dirname(morph_path), "probe_" + probe_id2 + ".txt")

                    # print(probe_path1)
                    # print(probe_path2)

                    if os.path.exists(probe_path1):
                        self.paths_morphed.append([probe_path1, morph_path])
                    # else: 
                        # print("missed")

                    if os.path.exists(probe_path2):    
                        # print("hit")
                        self.paths_morphed.append([probe_path2, morph_path])
                    # else:
                        # print("missed2")

        # print(self.paths_morphed)
        # Get bonafide paths
        for ds_i in data_sets: 
            for folder_i in self.sub_folders:
                ref_paths = os.path.join(self.bonafide_path, ds_i, self.flag, folder_i)
                for i, ref_path in enumerate(glob.glob(ref_paths +  '/ref_*.txt')):
                    # Find the two probe images that can be combined with current morphed image: 
                    ref_name = os.path.split(ref_path)[-1]
                    probe_id = ref_name.replace('.', '_').split('_')[1]
                    
                    probe_path = os.path.join(os.path.dirname(ref_path), "probe_" + probe_id + ".txt")

                    self.paths_bonafide.append([probe_path, ref_path])
        
        if self.save_path_flag: 
            full_paths = self.paths_bonafide + self.paths_morphed
            full_labels = [0]*len(self.paths_bonafide) + [1]*len(self.paths_morphed)
            with open(os.path.join(self.save_ds_path, self.flag + '_full_paths.txt'), 'w') as file:
                for row in full_paths:
                    file.write(' '.join([str(item) for item in row]))
                    file.write('\n')
            with open(os.path.join(self.save_ds_path, self.flag + '_full_labels.txt'), 'w') as file:
                for item in full_labels:
                    file.write(str(item))
                    file.write('\n')                
    
    def load_full_data(self):
        return pickle.load(open(os.path.join(self.save_ds_path, self.flag + ".pkl"), 'rb'))

    def save_full_data(self, ds):
        if not os.path.exists(self.save_ds_path): 
            os.makedirs(self.save_ds_path)
        pickle.dump(ds, open(os.path.join(self.save_ds_path, self.flag + ".pkl"), 'wb'))
    
    def get_full_data(self, save_flag=False, load_from_file=True):
        if load_from_file and os.path.exists(os.path.join(self.save_ds_path, self.flag + ".pkl")): 
            x_full, y_full = self.load_full_data()
            if "test" in self.flag: 
                meta_data = pickle.load(open(os.path.join(self.save_ds_path, "meta_" + self.flag + ".pkl"), 'rb'))
            else: 
                meta_data = {}
        else:
            x_full = []
            y_full = []
            meta_data = {"gender": [], "dataset": []}
            num_skipped = 0
            which_skipped = []
            for lab_i, ds_i in enumerate([self.paths_bonafide, self.paths_morphed]):
                # print(lab_i)
                for i, pair_path in enumerate(ds_i):
                    probe_i = readValuesFromFile(pair_path[0], print_flag=False)
                    ref_i = readValuesFromFile(pair_path[1], print_flag=False)
                    
                    if len(probe_i) == 0: 
                        num_skipped += 1
                        which_skipped.append(pair_path[0])
                        continue

                    if len(ref_i) == 0: 
                        num_skipped += 1
                        which_skipped.append(pair_path[1])
                        continue

                    # print(self.feat_shape)
                    probe_i = np.reshape(probe_i, self.feat_shape)

                    ref_i = np.reshape(ref_i, self.feat_shape)
                    # x_i = np.concatenate([probe_i, ref_i], 1)
                    x_i = probe_i - ref_i

                    x_full.append(x_i)
                    y_full.append(lab_i)
                    if "test" in self.flag:
                        if 'female_source' in pair_path[0]:
                            meta_data["gender"].append("female")
                        else: 
                            meta_data["gender"].append("male")
                        meta_data["dataset"].append(pair_path[0].split("Feature_")[-1].split('/')[1])
            x_full = np.array(x_full).transpose(1, 0, 2).tolist()        # 49xNx1024
            for sk in which_skipped: 
                print(sk)
            print(num_skipped)

            if save_flag:
                self.save_full_data([x_full, y_full])
                if "test" in self.flag: 
                    pickle.dump(meta_data, open(os.path.join(self.save_ds_path, "meta_" + self.flag + ".pkl"), 'wb'))

        print(f"Finished get data. Num samples: {len(y_full)}")
        # print(np.shape(x_full))
        # print(y_full)
        return x_full, y_full, meta_data




