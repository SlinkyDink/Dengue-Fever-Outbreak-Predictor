import numpy as np
import pandas as pd

TRAIN_X = 'data/dengue_features_train.csv'
TRAIN_Y = 'data/dengue_labels_train.csv'
TEST_X = 'data/dengue_features_test.csv'
SUB_FORMAT = 'data/submission_format.csv'
SUBMISSION = 'data/submit.csv'

def write_submission_file(sj_predictions,iq_predictions, write_to=SUBMISSION):
    sub = pd.read_csv(SUB_FORMAT, index_col=[0])
    sub.total_cases = np.concatenate([sj_predictions.astype(int), 
                                      iq_predictions.astype(int)])
    sub.to_csv(write_to)
    print("Should enter 416 predictions")
    print("Entered {0} predictions".format(len(sub.total_cases)))
    return None

def preprocess_data(isTest, isSJ):
    if isTest == True:
        data_path = TEST_X
    else:
        data_path = TRAIN_X

    if isSJ == True:
       features = [
            'station_avg_temp_c_shifted3',
            'station_min_temp_c_shifted3',
            'reanalysis_specific_humidity_g_per_kg_shifted3', 
            'reanalysis_dew_point_temp_k_shifted3', 
            'reanalysis_min_air_temp_k_shifted3',
            ]
    else: 
       features = [
            'reanalysis_specific_humidity_g_per_kg', 
            'reanalysis_dew_point_temp_k', 
            'reanalysis_min_air_temp_k',
            'station_min_temp_c_shifted1',
            ]
    
    df = pd.read_csv(data_path, index_col=[0])
    df = df[features]
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    if isTest == False:
        # Remove the first three columns of the Training set
        sj = sj.iloc[3:]    
        iq = iq.iloc[3:]
    
    # Fill empty slots with previous values
    sj.fillna(method='ffill', inplace=True)     
    iq.fillna(method='ffill', inplace=True)     
    
    if isSJ == True:
        return sj.astype('float32')
    else:
        return iq.astype('float32')

def get_train_label(isSJ):
    data_path = TRAIN_Y
    df = pd.read_csv(data_path, index_col=[0])

    if isSJ == True:
        sj = df.loc['sj']
        return sj.iloc[3:].astype('float32')
    else:
        iq = df.loc['iq']
        return iq.iloc[3:].astype('float32') 

def get_train_feature_label():
    sjLabel = get_train_label(isSJ=True)
    iqLabel = get_train_label(isSJ=False)
    sjFeature = preprocess_data(isTest=False, isSJ=True)
    iqFeature = preprocess_data(isTest=False, isSJ=False)
    sj = sjFeature
    sj["total_cases"] = sjLabel["total_cases"]
    iq = iqFeature
    iq["total_cases"] = iqLabel["total_cases"]
    return sj, iq
