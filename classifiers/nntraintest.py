import numpy as np
import tensorflow as tf
import os
from libsvm import svmutil
from tensorflow.keras.callbacks import TensorBoard
from data_loader import data_loader
from matplotlib import pyplot as plt
from libsvm_train_test import plot_APCER_BPCER

#this code is heavly based on the libsvm_train_test.py code made by Mobai

def train_nn(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, param_str, oversampled, feat_shapes):
    loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='1_training_set', feat_shape=feat_shapes)
    train_nn_from_loader(loader, strSavingModelFilePath,feat_shapes, param_str = param_str, oversampled=oversampled)

def train_nn_from_loader(data_loader, strSavingModelFilePath, feat_shape, param_str=None, oversampled=False ):
    if not os.path.exists(strSavingModelFilePath):
        os.makedirs(strSavingModelFilePath)

    train_loader = data_loader
    train_x, train_y, _ = train_loader.get_full_data(save_flag=False, load_from_file=False)      # x=[49xNx512], y=[N,]
    # print(train_y)

    # Check if the training should oversample the bonafide samples to ensure class balance.
    #   Note: Currently only works for num BF < num Morph and num Morph < 2 * num BF. This would
    #
    #   be easy to change if the data is different...
    if oversampled: 
        np.random.seed(666)
        bf_indxs = np.where(np.array(train_y) != 1)[0]
        mp_indxs = np.where(np.array(train_y) == 1)[0]
        print(bf_indxs)
        print(mp_indxs)
        print(f"len mp {len(mp_indxs)}")
        print(f"len bf {len(bf_indxs)}")


        # print(len(bf_indxs), len(mp_indxs))
        bf_indxs = np.sort(np.concatenate([bf_indxs, np.random.choice(bf_indxs, len(mp_indxs) - len(bf_indxs), replace=False)]))
        os_indxs = np.concatenate([bf_indxs, mp_indxs])
        train_y = [train_y[ind] for ind in os_indxs]
        train_x = [[train_x[i][j] for j in os_indxs] for i in range(len(train_x))]
        assert np.sum(np.array(train_y) == 1) / len(train_y) == 0.5
        print(f"Oversampled Size {len(train_y)}, num_morph {np.sum(np.array(train_y) == 1)}")
    else: 
        print(f"Data Size {len(train_y)}, num_morph {np.sum(np.array(train_y) == 2)}")

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(feat_shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])
    print(len(train_x))
    print(len(train_x[0]))
    print(len(train_x[0][0]))
    print(len(train_y))
    train_x = np.array(train_x[0])
    # train_x = np.expand_dims(train_x, axis=0)
    print(train_x.shape)
    train_y = np.array(train_y)
    # return
    # Summary of the model
    print("starting fitting of model")
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=2)
    print("finished fitting of model")
    model.save(strSavingModelFilePath + 'model.keras')
    # Train the SVM model


def test_nn(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, feat_shapes,
                  plotname=""):
    test_loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='3_test_set', feat_shape=feat_shapes)

    test_x, test_y, meta_data = test_loader.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    print(len(test_y))
    test_x = np.array(test_x[0])
    preds = []
    p_vals = []
    # for i, x_i in enumerate(test_x):
    #     print("Starting Loop " + str(i))
    #     model_i = svmutil.svm_load_model(strSavingModelFilePath + 'model_' + str(i) + '.model')
    #     _, _, p_val_i = svmutil.svm_predict(test_y, x_i, model_i, '-b 1 -q')
    #     # preds.append(pred_i)
    #     p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())

    model = tf.keras.models.load_model(strSavingModelFilePath + 'model.keras')
    p_vals = model.predict(test_x)
    
    # print(p_vals.shape)
    # print(p_vals[0])
    # print(p_vals[1:10])
    # print(p_vals[-1])
    p_vals = [p_vals,p_vals]
    
    p_vals = np.array(p_vals)
    print(p_vals.mean(0))
    #np.savetxt(os.path.join(strSavingModelFilePath, 'pred_test.txt'), preds,fmt='%i')
    #np.savetxt(os.path.join(strSavingModelFilePath, 'p_vals_test.txt'), p_vals)
    #np.savetxt(os.path.join(strSavingModelFilePath, 'pred_mean_test.txt'), pred_mean, delimiter="\n")
    #np.savetxt(os.path.join(strSavingModelFilePath, 'pred_std_test.txt'), pred_std, delimiter="\n")
    #np.savetxt(os.path.join(strSavingModelFilePath, 'pred_mean_class_test.txt'), pred_mean_class.astype(int), fmt='%i', delimiter="\n")
    #with open(os.path.join(strSavingModelFilePath, 'acc.txt'), "w") as f:
    #    f.write(f"Accuracy Test: {acc}")


    # Get evaluation metrics and plot: 
    fig, axs = plt.subplots(2, 4, figsize=(30, 10))

    # Plot the apcr/bpcr results for entire test dataset: 
    apcr, bpcr, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
    indxs_10 = np.where(np.abs(np.array(bpcr) - 0.1).min() == np.abs(np.array(bpcr) - 0.1))[0]
    indxs_1 = np.where(np.abs(np.array(bpcr) - 0.01).min() == np.abs(np.array(bpcr) - 0.01))[0]
    axs[0, 0].loglog(apcr, bpcr, label='__nolegend__')

    ten_perc = [np.array(bpcr)[indxs_10], np.array(apcr)[indxs_10]]
    one_perc = [np.array(bpcr)[indxs_1], np.array(apcr)[indxs_1]]
    axs[0, 0].loglog([0, ten_perc[-1][0]], [ten_perc[0][0], ten_perc[0][0]], c="black", 
                    label=f'BPCR: {np.round(ten_perc[0][0]*100, 2)}% APCR: {np.round(ten_perc[-1][0]*100, 2)}%  eer: {np.round(eer*100, 2)}% ')
    axs[0, 0].loglog([ten_perc[-1][0], ten_perc[-1][0]], [0, ten_perc[0][0]], c="black", label='__nolegend__')
    axs[0, 0].loglog([0, one_perc[-1][0]], [one_perc[0][0], one_perc[0][0]], c="black", ls='--', 
                    label=f'BPCR: {np.round(one_perc[0][0]*100, 2)}% APCR: {np.round(one_perc[-1][0]*100, 2)}%')
    axs[0, 0].loglog([one_perc[-1][0], one_perc[-1][0]], [0, one_perc[0][0]], c="black", ls='--', label='__nolegend__')
    axs[0, 0].legend()
    axs[0, 0].set_title('plot_apcr_bpcr.png')

    # Accuracies for different thresholds
    accs_thres = []
    thres = np.arange(0.01, 1., 0.01)
    for t_i in thres: 

        preds_i = (p_vals.mean(0) > t_i).astype(int)
        test_y = np.array(test_y).reshape(-1,1)
        # print(preds_i.shape)
        # print(test_y.shape)
        # print(np.array(test_y) - preds_i)
        acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
        # print(np.abs(np.array(test_y) - preds_i).sum())
        # print(len(test_y))
        # print(np.abs(np.array(test_y) - preds_i).sum() / len(test_y))
        accs_thres.append(acc_i)
        if t_i == 0.5: 
            acc_mid = acc_i
            axs[0, 1].plot([0, 0.5], [acc_i, acc_i], c='black', ls='--')
            axs[0, 1].plot([0.5, 0.5], [0, acc_i], c='black', ls='--')
    axs[0, 1].plot(thres, accs_thres)
    axs[0, 1].set_ylim([min(accs_thres) - 0.05, 1.])
    axs[0, 1].set_xlim([0, 1.])
    axs[0, 1].set_title(f'Acc: {np.round(acc_mid * 100, 2)}%, Acc Max: {np.round(max(accs_thres) * 100, 2)}%')

    # Accuracies of individual predictors: 
    accs_ind = []
    acc_all = 1 - np.abs(np.array(test_y) - p_vals.mean(0).round()).sum() / len(test_y)
    for p_val_i in p_vals: 
        preds_i = p_val_i.round()
        acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
        accs_ind.append(acc_i)
        print(accs_ind)
    axs[0, 2].plot(np.arange(len(accs_ind)), accs_ind)
    axs[0, 2].plot([0, len(accs_ind) - 1], [acc_all, acc_all])
    axs[0, 2].set_title(f"Acc all {np.round(acc_all*100, 2)}")
    
    # Accuracies for different probability bins:
    p_vals_mean = p_vals.mean(0) 
    accs_bins = []
    prob_bins = []
    for i in range(10):
        min_i = i / 10
        max_i = (i + 1) / 10
        indxs = np.where(np.stack([p_vals_mean >= min_i, p_vals_mean < max_i]).all(0))[0]
        if len(indxs) != 0:
            p_vals_i = p_vals_mean[indxs].round()
            test_i = np.array(test_y)[indxs]
            acc_i = 1 - np.abs(np.array(test_i) - p_vals_i).sum() / len(test_i)
            accs_bins.append(acc_i)
            prob_bins.append(max_i)
    axs[0, 3].plot(prob_bins, accs_bins)
    axs[0, 3].set_xlim(0.1, 1.)

    # APCR/BPCR/Acc for male v female
    male_indxs = np.where(np.array(meta_data["gender"]) == "male")[0]
    female_indxs = np.where(np.array(meta_data["gender"]) == "female")[0]
    p_vals_male = p_vals[:, male_indxs]
    y_test_male = np.array(test_y)[male_indxs]
    p_vals_female = p_vals[:, female_indxs]
    y_test_female = np.array(test_y)[female_indxs]
    acc_male = 1 - np.abs(np.array(y_test_male) - p_vals_male.mean(0).round()).sum() / len(y_test_male)
    acc_female = 1 - np.abs(np.array(y_test_female) - p_vals_female.mean(0).round()).sum() / len(y_test_female)
    apcr_male, bpcr_male, _ ,_= plot_APCER_BPCER(p_vals_male, np.array(y_test_male), step=0.01, plot_flag=False)
    apcr_female, bpcr_female, _,_ = plot_APCER_BPCER(p_vals_female, np.array(y_test_female), step=0.01, plot_flag=False)
    axs[1, 0].loglog(apcr_male, bpcr_male, label='male')
    axs[1, 0].loglog(apcr_female, bpcr_female, label='female')
    axs[1, 0].set_title(f"Accuracy Male {np.round(acc_male * 100, 2)}% and Female {np.round(acc_female * 100, 2)}%")
    axs[1, 0].legend()

    # APCR/BPCR/Acc for different datasets:
    thres = np.arange(0.01, 1., 0.01)
    for ds in np.unique(meta_data["dataset"]):
        indxs = np.where(np.array(meta_data["dataset"]) == ds)[0]
        p_vals_i = p_vals[:, indxs]
        test_i = np.array(test_y)[indxs]
        acc_i = 1 - np.abs(np.array(test_i) - p_vals_i.mean(0).round()).sum() / len(test_i)
        apcr_i, bpcr_i, _,_ = plot_APCER_BPCER(p_vals_i, np.array(test_i), step=0.01, label=ds, plot_flag=False)
        axs[1, 1].loglog(apcr_i, bpcr_i, label=f"{ds} - {np.round(acc_i*100, 2)}%")
        accs_thres = []
        for t_i in thres: 
            preds_i = (p_vals_i.mean(0) > t_i).astype(int)
            acc_i = 1 - np.abs(np.array(test_i) - preds_i).sum() / len(test_i)
            accs_thres.append(acc_i)
            if t_i == 0.5: 
                acc_mid = acc_i
                axs[1, 2].plot([0, 0.5], [acc_i, acc_i], c='black', ls='--')
        axs[1, 2].plot(thres, accs_thres, label=f"{ds} - {np.round(acc_mid*100, 2)}% - {np.round(max(accs_thres)*100, 2)}%")
    axs[1, 1].legend()
    axs[1, 2].legend()

    fig.savefig(os.path.join(strSavingModelFilePath, 'plot_metrics'+ plotname + '.png'))

