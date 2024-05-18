import numpy as np
import pickle
import os
from libsvm import svmutil
from tensorflow.python.framework.tensor_util import FastAppendFloat8e5m2ArrayToTensorProto
from data_loader import data_loader
from matplotlib import pyplot as plt
from libsvm_train_test import plot_APCER_BPCER
import tensorflow as tf
from dual_test import train_svm_from_loader 
from nntraintest import train_nn_from_loader, test_nn
#this code is heavly based on the libsvm_train_test.py code made by Mobai

def dual_train_nn(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, param_str, oversampled, feat_shapes,
                   strInputBonafideFeaturesFolders2, strInputAttacksFeaturesFolders2, strSavingModelFilePath2,param_str2, oversampled2, feat_shapes2):
    loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='1_training_set', feat_shape=feat_shapes)
    loader2 = data_loader(strInputAttacksFeaturesFolders2, strInputBonafideFeaturesFolders2, flag='1_training_set', feat_shape=feat_shapes2)
    match_morphed1, match_bonafide1, match_morphed2, match_bonafide2 = loader.match_cross_dataset(loader2)
    loader.paths_morphed = match_morphed1
    loader.paths_bonafide = match_bonafide1
    loader2.paths_morphed = match_morphed2
    loader2.paths_bonafide = match_bonafide2
    train_svm_from_loader(loader, strSavingModelFilePath, param_str=param_str, oversampled=oversampled)
    print("done training mobai svm")
    train_nn_from_loader(loader2, strSavingModelFilePath2,feat_shape=feat_shapes2, param_str = param_str2, oversampled=oversampled2)
    print("done training face svm")

def dual_test_nn(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, feat_shapes,
                  strInputBonafideFeaturesFolders2, strInputAttacksFeaturesFolders2, strSavingModelFilePath2,feat_shapes2, ratio=0.5, plotname=""):


    print(ratio)
    test_loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='3_test_set', feat_shape=feat_shapes)
    test_loader2 = data_loader(strInputAttacksFeaturesFolders2, strInputBonafideFeaturesFolders2, flag='3_test_set', feat_shape=feat_shapes2)
    match_morphed1, match_bonafide1, match_morphed2, match_bonafide2 = test_loader.match_cross_dataset(test_loader2)

    test_loader.paths_morphed = match_morphed1
    test_loader.paths_bonafide = match_bonafide1
    test_loader2.paths_morphed = match_morphed2
    test_loader2.paths_bonafide = match_bonafide2

    test_x, test_y, meta_data = test_loader.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    test_x2, test_y2, meta_data2 = test_loader2.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    print(len(test_y))
    print(len(test_y2))
    preds = []
    p_vals = []
    preds2 = []
    p_vals2 = []
    print("nn testing")
    model = tf.keras.models.load_model(strSavingModelFilePath2 + 'model.keras')

    test_x2 = np.array(test_x2[0])
    p_vals2 = model.predict(test_x2)
    print("svm testing")
    for i, x_i in enumerate(test_x):
        print("Starting Loop " + str(i))
        model_i = svmutil.svm_load_model(strSavingModelFilePath + 'model_' + str(i) + '.model')
        pred_i, acc_i, p_val_i = svmutil.svm_predict(test_y, x_i, model_i, '-b 1 -q')
        preds.append(pred_i)
        p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())

    preds = np.array(preds)
    p_vals = np.array(p_vals)
    p_vals2 = np.array(p_vals2)
    n_vals = p_vals.mean(0)
    n_vals2 = np.squeeze(p_vals2)
    print(p_vals.shape)
    print(p_vals2.shape)
    p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
    #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
    p_vals = np.array([p_vals,p_vals])
    pred_mean = preds.mean(0)
    pred_std  = preds.std(0)
    pred_mean_class = pred_mean.round()
    acc = 1 - np.abs(np.array(test_y) - pred_mean_class).sum() / len(test_y)

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
        acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
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

    det_plot_vars = {
        "one_perc": one_perc,
        "ten_perc": ten_perc,
        "apcr": apcr,
        "bpcr": bpcr,
        "eer": eer,
    }

    with open(os.path.join(strSavingModelFilePath, 'plot_metrics'+ plotname +"_"+ str(ratio)+  "det_plot_vars.pkl"), "wb") as file:
        pickle.dump(det_plot_vars, file)


    fig.savefig(os.path.join(strSavingModelFilePath, 'plot_metrics'+ plotname +"_"+ str(ratio)+ '.png'))





def dualnn_tune_ratio(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, feat_shapes,
                    strInputBonafideFeaturesFolders2, strInputAttacksFeaturesFolders2, strSavingModelFilePath2,feat_shapes2, plotname=""):


    test_loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='2_dev_set', feat_shape=feat_shapes)
    test_loader2 = data_loader(strInputAttacksFeaturesFolders2, strInputBonafideFeaturesFolders2, flag='2_dev_set', feat_shape=feat_shapes2)
    match_morphed1, match_bonafide1, match_morphed2, match_bonafide2 = test_loader.match_cross_dataset(test_loader2)

    test_loader.paths_morphed = match_morphed1
    test_loader.paths_bonafide = match_bonafide1
    test_loader2.paths_morphed = match_morphed2
    test_loader2.paths_bonafide = match_bonafide2

    test_x, test_y, meta_data = test_loader.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    test_x2, test_y2, meta_data2 = test_loader2.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    print(len(test_y))
    print(len(test_y2))
    preds = []
    p_vals = []
    preds2 = []
    p_vals2 = []
    print("nn testing")
    model = tf.keras.models.load_model(strSavingModelFilePath2 + 'model.keras')

    test_x2 = np.array(test_x2[0])
    p_vals2 = model.predict(test_x2)
    print("svm testing")
    for i, x_i in enumerate(test_x):
        print("Starting Loop " + str(i))
        model_i = svmutil.svm_load_model(strSavingModelFilePath + 'model_' + str(i) + '.model')
        pred_i, acc_i, p_val_i = svmutil.svm_predict(test_y, x_i, model_i, '-b 1 -q')
        preds.append(pred_i)
        p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())

    preds = np.array(preds)
    p_vals = np.array(p_vals)
    p_vals2 = np.array(p_vals2)
    n_vals = p_vals.mean(0)
    n_vals2 = np.squeeze(p_vals2)
    print(p_vals.shape)
    print(p_vals2.shape)
    # p_vals = np.expand_dims(p_vals, axis=0)
    # p_vals2 = np.expand_dims(p_vals2, axis=0)

    optimalapcr = [0,0]
    ##HACK TO CIRCUMVENT THE MEANS
    p_vals2 = np.array([n_vals2,n_vals2])
    print(p_vals.shape)
    print(p_vals2.shape)

    apcer, bpcer, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
    apcer2, bpcer2, _, eer2 = plot_APCER_BPCER(p_vals2, np.array(test_y2), step=0.01, plot_flag=False)
    optimal_static_weight = (1/eer)/(1/eer + 1/eer2)

    print(f"Optimal static weight: {optimal_static_weight}")
    bpcer10 = []
    bpcer1 = []

    eers = []
    for ratio in np.linspace(0,1,101):
        p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
        #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
        p_vals = np.array([p_vals,p_vals])

        apcer, bpcer, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
        eers.append(eer)

    mineer = min(eers)
    optimal_brute = eers.index(mineer)/100

    print(f"Optimal brute force weight: {optimal_brute}")
    return optimal_brute
    eers = []
    for ratio in np.linspace(0,1,101):
        p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
        #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
        p_vals = np.array([p_vals,p_vals])

        apcer, bpcer, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
        eers.append(eer)

    mineer = min(eers)
    optimal_brute = eers.index(mineer)/100
    print(f"Optimal brute weight {optimal_brute} eer: {mineer}")
    return optimal_brute
    
    
    """
    for ratio in np.linspace(0,1,11):
        p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
        # p_vals = np.array(n_vals) + np.array(n_vals2)

        #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
        p_vals = np.array([p_vals,p_vals])

        pred_mean = preds.mean(0)
        pred_std  = preds.std(0)
        pred_mean_class = pred_mean.round()
        acc = 1 - np.abs(np.array(test_y) - pred_mean_class).sum() / len(test_y)

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
        bpcer10.append([round(ratio,2),ten_perc[1]])
        bpcer1.append([round(ratio,2),one_perc[1]])
        axs[0, 0].loglog([0, ten_perc[-1][0]], [ten_perc[0][0], ten_perc[0][0]], c="black", 
                        label=f'BPCR: {np.round(ten_perc[0][0]*100, 2)}% APCR: {np.round(ten_perc[-1][0]*100, 2)}% eer: {np.round(eer*100, 2)}%')
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
            acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
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
        # print("I <3 print debugging")
        # print(p_vals)
        for p_val_i in p_vals: 
            preds_i = p_val_i.round()
            acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
            # print(acc_i)
            accs_ind.append(acc_i)
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
            apcr_i, bpcr_i, _ ,_= plot_APCER_BPCER(p_vals_i, np.array(test_i), step=0.01, label=ds, plot_flag=False)
            axs[1, 1].loglog(apcr_i, bpcr_i, label=f"{ds} - {np.round(acc_i*100, 2)}%")
            accs_thres 

def dualnn_tune_ratio2(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, feat_shapes,
                    strInputBonafideFeaturesFolders2, strInputAttacksFeaturesFolders2, strSavingModelFilePath2,feat_shapes2, plotname=""):


    test_loader = data_loader(strInputAttacksFeaturesFolders, strInputBonafideFeaturesFolders, flag='3_test_set', feat_shape=feat_shapes)
    test_loader2 = data_loader(strInputAttacksFeaturesFolders2, strInputBonafideFeaturesFolders2, flag='3_test_set', feat_shape=feat_shapes2)
    match_morphed1, match_bonafide1, match_morphed2, match_bonafide2 = test_loader.match_cross_dataset(test_loader2)

    test_loader.paths_morphed = match_morphed1
    test_loader.paths_bonafide = match_bonafide1
    test_loader2.paths_morphed = match_morphed2
    test_loader2.paths_bonafide = match_bonafide2

    test_x, test_y, meta_data = test_loader.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    test_x2, test_y2, meta_data2 = test_loader2.get_full_data(save_flag=False, load_from_file=False,)      # x=[49xNx512], y=[N,]
    print(len(test_y))
    print(len(test_y2))
    preds = []
    p_vals = []
    preds2 = []
    p_vals2 = []
    print("nn testing")
    model = tf.keras.models.load_model(strSavingModelFilePath2 + 'model.keras')

    test_x2 = np.array(test_x2[0])
    p_vals2 = model.predict(test_x2)
    model = tf.keras.models.load_model(strSavingModelFilePath + 'model.keras')

    test_x = np.array(test_x[0])
    p_vals = model.predict(test_x)
    # print("svm testing")
    # for i, x_i in enumerate(test_x):
    #     print("Starting Loop " + str(i))
    #     model_i = svmutil.svm_load_model(strSavingModelFilePath + 'model_' + str(i) + '.model')
    #     pred_i, acc_i, p_val_i = svmutil.svm_predict(test_y, x_i, model_i, '-b 1 -q')
    #     preds.append(pred_i)
    #     p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())

    preds = np.array(preds)
    p_vals = np.array(p_vals)
    p_vals2 = np.array(p_vals2)
    n_vals = np.squeeze(p_vals)
    n_vals2 = np.squeeze(p_vals2)
    print(p_vals.shape)
    print(p_vals2.shape)
    # p_vals = np.expand_dims(p_vals, axis=0)
    # p_vals2 = np.expand_dims(p_vals2, axis=0)

    optimalapcr = [0,0]
    ##HACK TO CIRCUMVENT THE MEANS
    p_vals = np.array([n_vals,n_vals])
    p_vals2 = np.array([n_vals2,n_vals2])
    print(p_vals.shape)
    print(p_vals2.shape)

    apcer, bpcer, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
    apcer2, bpcer2, _, eer2 = plot_APCER_BPCER(p_vals2, np.array(test_y2), step=0.01, plot_flag=False)
    optimal_static_weight = (1/eer)/(1/eer + 1/eer2)

    print(f"Optimal static weight: {optimal_static_weight}")
    bpcer10 = []
    bpcer1 = []

    eers = []
    for ratio in np.linspace(0,1,11):
        p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
        #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
        p_vals = np.array([p_vals,p_vals])

        apcer, bpcer, _, eer = plot_APCER_BPCER(p_vals, np.array(test_y), step=0.01, plot_flag=False)
        eers.append(eer)

    mineer = min(eers)
    optimal_brute = eers.index(mineer)/100
    print(f"Optimal brute weight {optimal_brute} eer: {mineer}")
    return optimal_static_weight
    for ratio in np.linspace(0,1,11):
        p_vals = np.array(n_vals) *(ratio) + np.array(n_vals2) *(1-ratio)
        # p_vals = np.array(n_vals) + np.array(n_vals2)

        #HACK TO CIRCUMVENT THE MEANS ALL OVER THE PLOTTING CODE.
        p_vals = np.array([p_vals,p_vals])
        print(p_vals)

        print(p_vals.shape)
        # pred_mean = preds.mean(0)
        # pred_std  = preds.std(0)
        # pred_mean_class = pred_mean.round()
        # acc = 1 - np.abs(np.array(test_y) - pred_mean_class).sum() / len(test_y)

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
        bpcer10.append([round(ratio,2),ten_perc[1]])
        bpcer1.append([round(ratio,2),one_perc[1]])
        axs[0, 0].loglog([0, ten_perc[-1][0]], [ten_perc[0][0], ten_perc[0][0]], c="black", 
                        label=f'BPCR: {np.round(ten_perc[0][0]*100, 2)}% APCR: {np.round(ten_perc[-1][0]*100, 2)}% eer: {np.round(eer*100, 2)}%')
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
            acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
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
        # print("I <3 print debugging")
        # print(p_vals)
        for p_val_i in p_vals: 
            preds_i = p_val_i.round()
            acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
            # print(acc_i)
            accs_ind.append(acc_i)
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
            apcr_i, bpcr_i, _ ,_= plot_APCER_BPCER(p_vals_i, np.array(test_i), step=0.01, label=ds, plot_flag=False)
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
        print(os.path.join(strSavingModelFilePath, 'plot_metrics_tuning'+ plotname + "_" + str(round(ratio,2))+ '.png'))

        fig.savefig(os.path.join(strSavingModelFilePath, 'plot_metrics_tuning'+ plotname + "_" + str(round(ratio,2))+ '.png'))
    print("done")
    print(np.linspace(0,1,11))
    print(bpcer10)
    print(bpcer1)

"""