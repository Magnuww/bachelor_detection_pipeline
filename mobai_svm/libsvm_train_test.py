from libsvm import svmutil
from utils import readAllFeatureVectorsFromFolder
import numpy as np
from data_loader import data_loader
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy


def calc_APCER_BPCER(p_vals, labels, thres=0.5):
    # print(p_vals.shape)
    # print(labels.shape)
    preds = p_vals.mean(0)
    preds = (preds >= thres).astype(int)

    morph_inxs = np.where(labels == 1)[0]
    bona_indxs = np.where(labels != 1)[0]

    # APCER - False Negative (i.e. morph classified as BF)
    # BPCER - False Positive (i.e. BF classified as morph)
    apcer = np.sum([preds[morph_inxs] != 1]) / len(morph_inxs)
    bpcer = np.sum([preds[bona_indxs] == 1]) / len(bona_indxs)

    return apcer, bpcer


def plot_APCER_BPCER(
    p_vals, labels, step=0.01, logscale=True, label="", plot_flag=True
):
    apcer = []
    bpcer = []
    thres = []
    for thres_i in np.arange(0.05, 0.95 + step, step):
        apcer_i, bpcer_i = calc_APCER_BPCER(p_vals, labels, thres_i)
        apcer.append(apcer_i)
        bpcer.append(bpcer_i)
        thres.append(thres_i)
    if plot_flag:
        fig, axs = plt.subplots()
        if logscale:
            axs.loglog(apcer, bpcer, label=label)
        else:
            axs.plot(apcer, bpcer, label=label)

        axs.set_xlabel("APCER")
        axs.set_ylabel("BPCER")
        return apcer, bpcer, thres, [fig, axs]

    return apcer, bpcer, thres


def tune_svm(
    strInputBonafideFeaturesFolders,
    strInputAttacksFeaturesFolders,
    strSaveResultsFolder,
):
    run_name = "second_diffC_"
    train_val_split = 0.8
    # Set tuning parameters:
    tune_params = {
        "Kernel": ["Linear"],  # , "Poly", "RBF"],
        "weight_imbalance": [-2, -1, 0, 1],
        "C": [10.0, 0.1],
    }
    setups = np.array(np.meshgrid(*tune_params.values())).T.reshape(
        -1, len(tune_params)
    )
    params = list(tune_params.keys())

    np.random.seed(666)
    train_loader = data_loader(
        strInputAttacksFeaturesFolders,
        strInputBonafideFeaturesFolders,
        flag="1_training_set",
    )
    train_x, train_y, _ = train_loader.get_full_data(
        save_flag=True, load_from_file=True
    )  # x=[49xNx512], y=[N,]

    # Split train set into train_val:
    train_indxs = np.random.choice(
        len(train_y), int(len(train_y) * train_val_split), replace=False
    )
    val_indxs = np.arange(len(train_y))[~np.isin(np.arange(len(train_y)), train_indxs)]
    train_indxs = train_indxs[np.argsort(train_indxs)]
    val_indxs = val_indxs[np.argsort(val_indxs)]
    val_x = [[train_x[i][j] for j in val_indxs] for i in range(len(train_x))]
    val_y = [train_y[j] for j in val_indxs]
    train_x_orig = [[train_x[i][j] for j in train_indxs] for i in range(len(train_x))]
    train_y_orig = [train_y[j] for j in train_indxs]

    # Class weights:
    weight_0 = 1
    weight_1 = np.round(
        (len(train_y) - sum(np.array(train_y) == 1)) / sum(np.array(train_y) == 1), 5
    )

    # Dict to store the results:
    results = {
        "APCR": [],
        "BPCR": [],
        "Acc": [],
        "Acc_thres": [],
        "thres": [],
        "p_vals": [],
    }

    if os.path.exists(os.path.join(strSaveResultsFolder, run_name + "results.pkl")):
        results = pickle.load(
            open(os.path.join(strSaveResultsFolder, run_name + "results.pkl"), "rb")
        )
        setup_df = pickle.load(
            open(os.path.join(strSaveResultsFolder, run_name + "params.pkl"), "rb")
        )
        train_indxs = pickle.load(
            open(os.path.join(strSaveResultsFolder, run_name + "train_indxs.pkl"), "rb")
        )

    for i, setup_array_i in enumerate(setups):
        if len(results["Acc"]) > i:
            continue
        setup_i = dict(zip(params, setup_array_i))
        param_str = "-s 0"
        if setup_i["Kernel"] == "Linear":
            param_str += " -t 0"
        elif setup_i["Kernel"] == "Poly":
            param_str += " -t 1 -d 3"
        elif setup_i["Kernel"] == "RBF":
            param_str += " -t 2"
        else:
            raise NotImplementedError

        if "C" in setup_i:
            param_str += " -c {}".format(setup_i["C"])

        param_str += " -b 1"
        train_x = train_x_orig
        train_y = train_y_orig
        if "weight_imbalance" in setup_i:
            if setup_i["weight_imbalance"] == "1":
                param_str += (
                    f" -w0 {weight_0} -w1 {weight_1}"  # .format(weight_0, weight_1)
                )
            elif setup_i["weight_imbalance"] == "-1":
                # Undersampled versions of the dataset:
                # i.e. remove n of the morphed samples so that we have the same proportion of samples
                train_x_us = copy.copy(train_x_orig)
                train_y_us = copy.copy(train_y_orig)
                bf_indxs = np.where(np.array(train_y_us) != 1)[0]
                mp_indxs = np.where(np.array(train_y_us) == 1)[0]
                mp_indxs = np.sort(
                    np.random.choice(mp_indxs, len(bf_indxs), replace=False)
                )
                us_indxs = np.concatenate([bf_indxs, mp_indxs])
                train_y_us = [train_y_us[ind] for ind in us_indxs]
                train_x_us = [
                    [train_x_us[i][j] for j in us_indxs] for i in range(len(train_x))
                ]
                assert np.sum(np.array(train_y_us) == 1) / len(train_y_us) == 0.5

                train_x = train_x_us
                train_y = train_y_us
            elif setup_i["weight_imbalance"] == "-2":
                # Oversampled versions of the dataset:
                train_x_us = copy.copy(train_x_orig)
                train_y_us = copy.copy(train_y_orig)
                bf_indxs = np.where(np.array(train_y_us) != 1)[0]
                mp_indxs = np.where(np.array(train_y_us) == 1)[0]
                bf_indxs = np.sort(
                    np.concatenate(
                        [
                            bf_indxs,
                            np.random.choice(
                                bf_indxs, len(mp_indxs) - len(bf_indxs), replace=False
                            ),
                        ]
                    )
                )
                us_indxs = np.concatenate([bf_indxs, mp_indxs])
                train_y_us = [train_y_us[ind] for ind in us_indxs]
                train_x_us = [
                    [train_x_us[i][j] for j in us_indxs] for i in range(len(train_x))
                ]
                assert np.sum(np.array(train_y_us) == 1) / len(train_y_us) == 0.5
                print(
                    f"Ds Size {len(train_y_us)}, num_morph {np.sum(np.array(train_y_us) == 1)}"
                )
                train_x = train_x_us
                train_y = train_y_us

        param_str += " -q"

        # Loop over the N features and fit N different models:
        print(f"Starting loop {i}: {param_str}")
        svm_models_i = []
        for j, x_j in enumerate(train_x):
            problem_i = svmutil.svm_problem(train_y, x_j)
            param_i = svmutil.svm_parameter(param_str)
            model_i = svmutil.svm_train(problem_i, param_i)
            svm_models_i.append(model_i)
            # Save the model to a file
            if j % 5 == 0:
                print(f"\tFinished Model {j} for loop {i}")

        # Validation
        print("\n\tStarting validation")
        p_vals = []
        for j, x_j in enumerate(val_x):
            model_i = svm_models_i[j]
            p_val_i = svmutil.svm_predict(val_y, x_j, model_i, "-b 1 -q")[-1]
            p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())
        p_vals = np.array(p_vals)
        results["p_vals"].append(p_vals)
        print("\tFinished validation")

        # Compute accuracy:
        val_y = [0 if v != 1 else v for v in val_y]
        acc_i = 1 - np.abs(np.array(val_y) - p_vals.mean(0).round()).sum() / len(val_y)
        results["Acc"].append(acc_i)
        print("\tAccuracy: ", acc_i)

        # Compute APCR and BPCR values:
        apcr, bpcr, _ = plot_APCER_BPCER(
            p_vals, np.array(val_y), step=0.01, plot_flag=False
        )
        results["APCR"].append(apcr)
        results["BPCR"].append(bpcr)

        # Compute accuracies for different thresholds:
        thresholds = np.arange(0.05, 1, 0.05)
        acc_t_i = [
            1
            - np.abs(np.array(val_y) - (p_vals.mean(0) >= t_i).astype(int)).sum()
            / len(val_y)
            for t_i in thresholds
        ]
        results["Acc_thres"].append(acc_t_i)
        results["thres"] = thresholds
        print(f"Finished loop {i} \n\n")

        # Save results:
        if not os.path.exists(strSaveResultsFolder):
            os.makedirs(strSaveResultsFolder)
        pickle.dump(
            results,
            open(os.path.join(strSaveResultsFolder, run_name + "results.pkl"), "wb"),
        )
        setup_df = pd.DataFrame(setups, columns=params)
        pickle.dump(
            setup_df,
            open(os.path.join(strSaveResultsFolder, run_name + "params.pkl"), "wb"),
        )
        pickle.dump(
            train_indxs,
            open(
                os.path.join(strSaveResultsFolder, run_name + "train_indxs.pkl"), "wb"
            ),
        )


def train_svm(
    strInputBonafideFeaturesFolders,
    strInputAttacksFeaturesFolders,
    strSavingModelFilePath,
    param_str=None,
    oversampled=False,
    feat_shapes=(49, 512),
):
    if not os.path.exists(strSavingModelFilePath):
        os.makedirs(strSavingModelFilePath)

    train_loader = data_loader(
        strInputAttacksFeaturesFolders,
        strInputBonafideFeaturesFolders,
        flag="1_training_set",
        feat_shape=feat_shapes,
    )
    train_x, train_y, _ = train_loader.get_full_data(
        save_flag=False, load_from_file=False
    )  # x=[49xNx512], y=[N,]
    # print(train_y)

    # Check if the training should oversample the bonafide samples to ensure class balance.
    #   Note: Currently only works for num BF < num Morph and num Morph < 2 * num BF. This would
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
        bf_indxs = np.sort(
            np.concatenate(
                [
                    bf_indxs,
                    np.random.choice(
                        bf_indxs, len(mp_indxs) - len(bf_indxs), replace=False
                    ),
                ]
            )
        )
        os_indxs = np.concatenate([bf_indxs, mp_indxs])
        train_y = [train_y[ind] for ind in os_indxs]
        train_x = [[train_x[i][j] for j in os_indxs] for i in range(len(train_x))]
        assert np.sum(np.array(train_y) == 1) / len(train_y) == 0.5
        print(
            f"Oversampled Size {len(train_y)}, num_morph {np.sum(np.array(train_y) == 1)}"
        )
    else:
        print(f"Data Size {len(train_y)}, num_morph {np.sum(np.array(train_y) == 1)}")

    # Train the SVM model
    for i, x_i in enumerate(train_x):
        if not os.path.exists(strSavingModelFilePath + "model_" + str(i) + ".model"):
            problem_i = svmutil.svm_problem(train_y, x_i)
            if param_str is None:
                param_i = svmutil.svm_parameter("-s 0 -t 0 -b 1")
            else:
                param_i = svmutil.svm_parameter(param_str)
            model_i = svmutil.svm_train(problem_i, param_i)

            # Save the model to a file
            print(f"Finished Loop {i}")
            svmutil.svm_save_model(
                strSavingModelFilePath + "model_" + str(i) + ".model", model_i
            )


def test_svm(
    strInputBonafideFeaturesFolders,
    strInputAttacksFeaturesFolders,
    strSavingModelFilePath,
    load_preds=False,
    feat_shapes=(49, 512),
    resultOutput=None,
):
    if resultOutput is None:
        resultOutput = strSavingModelFilePath

    test_loader = data_loader(
        strInputAttacksFeaturesFolders,
        strInputBonafideFeaturesFolders,
        flag="3_test_set",
        feat_shape=feat_shapes,
    )
    test_x, test_y, meta_data = test_loader.get_full_data(
        save_flag=False,
        load_from_file=False,
    )  # x=[49xNx512], y=[N,]

    if not load_preds:
        preds = []
        p_vals = []
        for i, x_i in enumerate(test_x):
            print("Starting Loop " + str(i))
            model_i = svmutil.svm_load_model(
                strSavingModelFilePath + "model_" + str(i) + ".model"
            )
            pred_i, acc_i, p_val_i = svmutil.svm_predict(
                test_y, x_i, model_i, "-b 1 -q"
            )
            preds.append(pred_i)
            p_vals.append(np.array(p_val_i)[:, 1].flatten().tolist())

        preds = np.array(preds)
        p_vals = np.array(p_vals)
        pred_mean = preds.mean(0)
        pred_std = preds.std(0)
        pred_mean_class = pred_mean.round()
        acc = 1 - np.abs(np.array(test_y) - pred_mean_class).sum() / len(test_y)

        np.savetxt(
            os.path.join(strSavingModelFilePath, "pred_test.txt"), preds, fmt="%i"
        )
        np.savetxt(os.path.join(strSavingModelFilePath, "p_vals_test.txt"), p_vals)
        np.savetxt(
            os.path.join(strSavingModelFilePath, "pred_mean_test.txt"),
            pred_mean,
            delimiter="\n",
        )
        np.savetxt(
            os.path.join(strSavingModelFilePath, "pred_std_test.txt"),
            pred_std,
            delimiter="\n",
        )
        np.savetxt(
            os.path.join(strSavingModelFilePath, "pred_mean_class_test.txt"),
            pred_mean_class.astype(int),
            fmt="%i",
            delimiter="\n",
        )
        with open(os.path.join(strSavingModelFilePath, "acc.txt"), "w") as f:
            f.write(f"Accuracy Test: {acc}")
    else:
        preds = np.loadtxt(os.path.join(strSavingModelFilePath, "pred_test.txt"))
        p_vals = np.loadtxt(os.path.join(strSavingModelFilePath, "p_vals_test.txt"))

    # Get evaluation metrics and plot:
    fig, axs = plt.subplots(2, 4, figsize=(30, 10))

    # Plot the apcr/bpcr results for entire test dataset:
    apcr, bpcr, _ = plot_APCER_BPCER(
        p_vals, np.array(test_y), step=0.01, plot_flag=False
    )
    indxs_10 = np.where(
        np.abs(np.array(bpcr) - 0.1).min() == np.abs(np.array(bpcr) - 0.1)
    )[0]
    indxs_1 = np.where(
        np.abs(np.array(bpcr) - 0.01).min() == np.abs(np.array(bpcr) - 0.01)
    )[0]
    axs[0, 0].loglog(apcr, bpcr, label="__nolegend__")

    ten_perc = [np.array(bpcr)[indxs_10], np.array(apcr)[indxs_10]]
    one_perc = [np.array(bpcr)[indxs_1], np.array(apcr)[indxs_1]]
    axs[0, 0].loglog(
        [0, ten_perc[-1][0]],
        [ten_perc[0][0], ten_perc[0][0]],
        c="black",
        label=f"BPCR: {np.round(ten_perc[0][0]*100, 2)}% APCR: {np.round(ten_perc[-1][0]*100, 2)}%",
    )
    axs[0, 0].loglog(
        [ten_perc[-1][0], ten_perc[-1][0]],
        [0, ten_perc[0][0]],
        c="black",
        label="__nolegend__",
    )
    axs[0, 0].loglog(
        [0, one_perc[-1][0]],
        [one_perc[0][0], one_perc[0][0]],
        c="black",
        ls="--",
        label=f"BPCR: {np.round(one_perc[0][0]*100, 2)}% APCR: {np.round(one_perc[-1][0]*100, 2)}%",
    )
    axs[0, 0].loglog(
        [one_perc[-1][0], one_perc[-1][0]],
        [0, one_perc[0][0]],
        c="black",
        ls="--",
        label="__nolegend__",
    )
    axs[0, 0].legend()
    axs[0, 0].set_title("plot_apcr_bpcr.png")

    # Accuracies for different thresholds
    accs_thres = []
    thres = np.arange(0.01, 1.0, 0.01)
    for t_i in thres:
        preds_i = (p_vals.mean(0) > t_i).astype(int)
        acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
        accs_thres.append(acc_i)
        if t_i == 0.5:
            acc_mid = acc_i
            axs[0, 1].plot([0, 0.5], [acc_i, acc_i], c="black", ls="--")
            axs[0, 1].plot([0.5, 0.5], [0, acc_i], c="black", ls="--")
    axs[0, 1].plot(thres, accs_thres)
    axs[0, 1].set_ylim([min(accs_thres) - 0.05, 1.0])
    axs[0, 1].set_xlim([0, 1.0])
    axs[0, 1].set_title(
        f"Acc: {np.round(acc_mid * 100, 2)}%, Acc Max: {np.round(max(accs_thres) * 100, 2)}%"
    )

    # Accuracies of individual predictors:
    accs_ind = []
    acc_all = 1 - np.abs(np.array(test_y) - p_vals.mean(0).round()).sum() / len(test_y)
    for p_val_i in p_vals:
        preds_i = p_val_i.round()
        acc_i = 1 - np.abs(np.array(test_y) - preds_i).sum() / len(test_y)
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
        indxs = np.where(np.stack([p_vals_mean >= min_i, p_vals_mean < max_i]).all(0))[
            0
        ]
        if len(indxs) != 0:
            p_vals_i = p_vals_mean[indxs].round()
            test_i = np.array(test_y)[indxs]
            acc_i = 1 - np.abs(np.array(test_i) - p_vals_i).sum() / len(test_i)
            accs_bins.append(acc_i)
            prob_bins.append(max_i)
    axs[0, 3].plot(prob_bins, accs_bins)
    axs[0, 3].set_xlim(0.1, 1.0)

    # APCR/BPCR/Acc for male v female
    male_indxs = np.where(np.array(meta_data["gender"]) == "male")[0]
    female_indxs = np.where(np.array(meta_data["gender"]) == "female")[0]
    p_vals_male = p_vals[:, male_indxs]
    y_test_male = np.array(test_y)[male_indxs]
    p_vals_female = p_vals[:, female_indxs]
    y_test_female = np.array(test_y)[female_indxs]
    acc_male = 1 - np.abs(
        np.array(y_test_male) - p_vals_male.mean(0).round()
    ).sum() / len(y_test_male)
    acc_female = 1 - np.abs(
        np.array(y_test_female) - p_vals_female.mean(0).round()
    ).sum() / len(y_test_female)
    apcr_male, bpcr_male, _ = plot_APCER_BPCER(
        p_vals_male, np.array(y_test_male), step=0.01, plot_flag=False
    )
    apcr_female, bpcr_female, _ = plot_APCER_BPCER(
        p_vals_female, np.array(y_test_female), step=0.01, plot_flag=False
    )
    axs[1, 0].loglog(apcr_male, bpcr_male, label="male")
    axs[1, 0].loglog(apcr_female, bpcr_female, label="female")
    axs[1, 0].set_title(
        f"Accuracy Male {np.round(acc_male * 100, 2)}% and Female {np.round(acc_female * 100, 2)}%"
    )
    axs[1, 0].legend()

    # APCR/BPCR/Acc for different datasets:
    thres = np.arange(0.01, 1.0, 0.01)
    for ds in np.unique(meta_data["dataset"]):
        indxs = np.where(np.array(meta_data["dataset"]) == ds)[0]
        p_vals_i = p_vals[:, indxs]
        test_i = np.array(test_y)[indxs]
        acc_i = 1 - np.abs(np.array(test_i) - p_vals_i.mean(0).round()).sum() / len(
            test_i
        )
        apcr_i, bpcr_i, _ = plot_APCER_BPCER(
            p_vals_i, np.array(test_i), step=0.01, label=ds, plot_flag=False
        )
        axs[1, 1].loglog(apcr_i, bpcr_i, label=f"{ds} - {np.round(acc_i*100, 2)}%")
        accs_thres = []
        for t_i in thres:
            preds_i = (p_vals_i.mean(0) > t_i).astype(int)
            acc_i = 1 - np.abs(np.array(test_i) - preds_i).sum() / len(test_i)
            accs_thres.append(acc_i)
            if t_i == 0.5:
                acc_mid = acc_i
                axs[1, 2].plot([0, 0.5], [acc_i, acc_i], c="black", ls="--")
        axs[1, 2].plot(
            thres,
            accs_thres,
            label=f"{ds} - {np.round(acc_mid*100, 2)}% - {np.round(max(accs_thres)*100, 2)}%",
        )
    axs[1, 1].legend()
    axs[1, 2].legend()

    figName = "plot_metrics"
    figSavePath = os.path.join(strSavingModelFilePath, figName + ".png")
    fig.savefig(figSavePath)

    os.symlink(figSavePath, os.path.join(resultOutput, figName + ".png"))

    # save fig to pkl file
    # with open("plot_metrics.pkl", "wb") as f:
    #     pickle.dump(fig, f)
