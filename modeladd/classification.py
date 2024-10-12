import numpy as np
from matplotlib import pyplot as plt
from netcal.scaling import TemperatureScaling, LogisticCalibration, BetaCalibration
import pandas as pd
from netcal.metrics import ECE
# from netcal.presentation import ReliabilityDiagram

from sklearn.metrics import auc, roc_curve, precision_score
from sklearn.preprocessing import MinMaxScaler
from netcal.binning import HistogramBinning, IsotonicRegression, BBQ
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
from lightgbm import LGBMClassifier

input=".\\2Padd\\ico_two_model_add.csv"

output_path_ted=".\\2Padd\inter\calprob_ted.csv"
output_path_eth=".\\2Padd\inter\calprob_eth.csv"
output_path_two="..\\2Padd\inter\calprob_twocsv"

output_path_ted_norm=".\\2Padd\inter\LDG_result_norm.csv"
output_path_eth_norm=".\\2Padd\inter\GSG_result_norm.csv"


def my_lightgbm(x_train, x_test, y_train, y_test):
    # Train the model (fixed number of rounds and early stop)
    clf = LGBMClassifier(n_estimators=200, early_stopping_rounds=20)  # Set the early stop mechanism, where 10 can be adjusted as needed
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])

    # Predicting the test set probability
    test_predict_proba = clf.predict_proba(x_test)

    # Get the best round
    best_round = clf.best_iteration_

    # Predict
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    # # Feature importance visualization
    # data = {'y': data_features_part.columns,
    # 'x': clf.feature_importances_}
    # data = pd.DataFrame(data)
    # data = data.sort_values(by="x", ascending=False)

    # Evaluation index calculation
    train_accuracy = accuracy_score(y_train, train_predict)
    test_accuracy = accuracy_score(y_test, test_predict)

    roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    recall = recall_score(y_test, test_predict)
    f1 = f1_score(y_test, test_predict)
    precision = precision_score(y_test, test_predict)  # 计算精确率

    conf_matrix = confusion_matrix(y_test, test_predict)
    print('Best Round:', best_round)
    print('Train Accuracy:', train_accuracy)
    print('Precision:', precision)  # 输出精确率
    print('Test Accuracy:', test_accuracy)
    print('AUC:', roc_auc)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('Confusion Matrix:\n', conf_matrix)
    # return test_predict
    return roc_auc, test_predict_proba[:, 1]

if __name__ == '__main__':
    data = pd.read_csv(input)
    confidence_columns1 = ["ted0", "ted1"]
    confidence_columns2 = ["eth0", "eth1"]

    # Get the confidence column and convert to a numpy array
    confidences1 = data[confidence_columns1].values
    confidences2 = data[confidence_columns2].values

    ground_truth = data["label"].values  # 获取标签列的值并转换为 numpy 数组

    # The mean and standard deviation of positive and negative examples are calculated
    mean_positive1 = np.mean(confidences1[ground_truth == 1])
    std_positive1 = np.std(confidences1[ground_truth == 1])
    mean_negative1 = np.mean(confidences1[ground_truth == 0])
    std_negative1 = np.std(confidences1[ground_truth == 0])

    mean_positive2 = np.mean(confidences2[ground_truth == 1])
    std_positive2 = np.std(confidences2[ground_truth == 1])
    mean_negative2 = np.mean(confidences2[ground_truth == 0])
    std_negative2 = np.std(confidences2[ground_truth == 0])

    # The positive and negative examples are scaled separately
    confidences_scaled1 = np.empty_like(confidences1)  # Initialize an array of the same size as confidences
    confidences_scaled1[ground_truth == 1] = (confidences1[ground_truth == 1] - mean_positive1) / std_positive1
    confidences_scaled1[ground_truth == 0] = (confidences1[ground_truth == 0] - mean_negative1) / std_negative1

    confidences_scaled2 = np.empty_like(confidences2)
    confidences_scaled2[ground_truth == 1] = (confidences2[ground_truth == 1] - mean_positive2) / std_positive2
    confidences_scaled2[ground_truth == 0] = (confidences2[ground_truth == 0] - mean_negative2) / std_negative2

    # Normalized confidence column
    scaler = MinMaxScaler()
    confidences1 = scaler.fit_transform(confidences_scaled1)

    confidences2 = scaler.fit_transform(confidences_scaled2)

    calibration_models = [BetaCalibration(),LogisticCalibration(), TemperatureScaling(), BBQ(), IsotonicRegression(), HistogramBinning()]

    calibrations1 = []
    calibrations2= []

    print("Minimum value of ground_truth array:", np.min(ground_truth))
    print("Maximum value of ground_truth array:", np.max(ground_truth))
    print("Shape of ground_truth array:", ground_truth.shape)

    for model in calibration_models:
        model.fit(confidences1, ground_truth)
        calibration1 = model.transform(confidences1)
        calibrations1.append(calibration1)

        model.fit(confidences2, ground_truth)
        calibration2 = model.transform(confidences2)
        calibrations2.append(calibration2)

    #Saving the calibrated confidence value to a CSV file
    calibration_data1 = np.column_stack(calibrations1)
    calibration_df1 = pd.DataFrame(calibration_data1, columns=[f"ted_calibration_{i+1}" for i in range(calibration_data1.shape[1])])
    calibration_df1.to_csv(output_path_ted, index=False)

    calibration_data2 = np.column_stack(calibrations2)
    calibration_df2 = pd.DataFrame(calibration_data2, columns=[f"eth_calibration_{i+1}" for i in range(calibration_data1.shape[1])])
    calibration_df1.to_csv(output_path_eth, index=False)

    # Uncalibrated and calibrated ECE were calculated
    n_bins = 10  # Set the number of bins used by ECE
    ece = ECE(n_bins)
    uncalibrated_score1 = ece.measure(confidences1, ground_truth)
    print("Ted Uncalibrated ECE:", uncalibrated_score1)

    uncalibrated_score2 = ece.measure(confidences2, ground_truth)
    print("Eth Uncalibrated ECE:", uncalibrated_score2)

    calibration_scores1 = []
    calibration_scores2 = []
    for calibration1 in calibrations1:
        calibration_score1 = ece.measure(calibration1, ground_truth)
        calibration_scores1.append(calibration_score1)
        print("Ted Calibrated ECE:", calibration_score1)

    for calibration in calibrations2:
        calibration_score2 = ece.measure(calibration2, ground_truth)
        calibration_scores2.append(calibration_score2)
        print("Eth Calibrated ECE:", calibration_score2)

    # The difference between uncalibrated and calibrated ECE is output
    diff1=[]
    diff2=[]
    for i, score in enumerate(calibration_scores1):
        diff1.append(uncalibrated_score1 - score)
        print(f"Difference in ECE for Calibration {i+1}:", uncalibrated_score1 - score)
    print(diff1)
    print("The upper and lower records are input into weighting：")
    for i, score in enumerate(calibration_scores2):
        diff2.append(uncalibrated_score2 - score)
        print(f"Difference in ECE for Calibration {i+1}:", uncalibrated_score2 - score)
    print(diff2)

    df2 = pd.read_csv(output_path_eth)
    df1 = pd.read_csv(output_path_ted)

    # 归一化
    sum1 = 0
    for i in range(len(diff1)):
        sum1 = sum1 + diff1[i]

    sum2 = 0
    for i in range(len(diff2)):
        sum2 = sum2 + diff2[i]
    coefficients_norm1 = []
    coefficients_norm2 = []

    for i in range(len(diff1)):
        coefficients_norm1.append(diff1[i] / sum1)
    print(coefficients_norm1)
    print("Record the normalized ratio above and below：")
    for i in range(len(diff2)):
        coefficients_norm2.append(diff2[i] / sum2)
    print(coefficients_norm2)

    drop_cols=[]
    x1 = df1.drop(drop_cols, axis=1)
    x2 = df2.drop(drop_cols, axis=1)

    result1 = (x1 * diff1).sum(axis=1)
    result2 = (x2 * diff2).sum(axis=1)

    result_df1 = pd.DataFrame(result1, columns=['ted_result'])
    result_df2 = pd.DataFrame(result2, columns=['eth_result'])

    result_df1.to_csv(output_path_ted_norm, index=False)
    result_df2.to_csv(output_path_eth_norm, index=False)
    print("Result has been written to result.csv file.")

    # Combine the two CSVs and add the columns of the second file, starting from the last column of the first file
    result = pd.concat([result_df2, result_df1], axis=1)

    last_column = data.iloc[:, -1]

    result['label'] = last_column

    result.to_csv(output_path_two, index=False)
    # lightgbm
    df = pd.read_csv(output_path_two)
    y = df.label
    drop_cols = ['label']

    x = df.drop(drop_cols, axis=1)

    data_target_part = y
    data_features_part = x

    # Split the training and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size=0.2
                                                        , random_state=42)
    my_lightgbm(x_train, x_test, y_train, y_test)
