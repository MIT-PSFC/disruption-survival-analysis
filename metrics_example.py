import numpy as np
import pandas as pd
import pickle
from functools import partial

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, fbeta_score, \
    confusion_matrix, balanced_accuracy_score, auc

# Set seed
np.random.seed(42)

INPUT_FILE = 'tests/data_example/train_dataloader_example.pickle'

def fpr_score(y_true, y_pred):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return fp / (fp + tn)

def tpr_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)


metrics_dict = {
    'f1_score': f1_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
    'balanced_accuracy_score': balanced_accuracy_score,
    'accuracy_score': accuracy_score,
    'roc_auc_score': roc_auc_score,
    'f2_score': partial(fbeta_score, beta=2),
    'f0.5_score': partial(fbeta_score, beta=0.5),
    'confusion_matrix': confusion_matrix,
    'tpr': tpr_score,
    'fpr': fpr_score
}


def main():
     # Load data
    with open(INPUT_FILE,'rb') as fp:
         dataloader_example = pickle.load(fp)



    # y_pred_record_shot: timeseries of the disruptivity of a single shot [shape 1, number of time instance for the selected shot]
    # If you have logits take the logit that refers to close to disruption probability

    # Predict using your model
    # y_pred_record_shot = model.predict(dataloader_example['x'])    
    
    # Order dataloader by shot and time
    shots_info = pd.DataFrame(np.hstack([dataloader_example['groups'].reshape(-1,1), dataloader_example['time'][:,:, -1],
                  dataloader_example['time_until_disrupt'][:,:, -1],dataloader_example['y']]
                  ),
                  columns=['uid_shot', 'time', 'time_until_disrupt', 'y_true'])
    shots_info = shots_info.sort_values(by=['uid_shot', 'time'])
    


    
    y_true_shot, y_pred_shot = get_alarm_prediction(
        dataloader_custom=dataloader_example,
        shots_info_df=shots_info,
        mitigation_time = 0.030, # Mitigation time in seconds,
        t_alarm = 0.001, #[s]
        high_thr = 0.9,
        low_thr = 0.3
         
    )

    # Metrics of shot
    for m_name, metrics_fun in metrics_dict.items():
        val = metrics_fun(y_true_shot, y_pred_shot)
        print(f'{m_name} - {np.round(val, 5)}')


    ### The following steps evaluate the correct labelling and alarm raise for the shots of the dataset
    ### According to the method used by Zhu in his paper
    # ROC curve is evaluated on shot metrics (1 label for each shot) changing the alarm detection threshold.
    tpr_list = []
    fpr_list = []
    threshold_list = []
    for thr in np.linspace(0,1, 101):
        # Get prediction on shot for the selected threshold
        y_true_shot_t, y_pred_shot_t = get_alarm_prediction(
            dataloader_custom=dataloader_example,
            shots_info_df=shots_info,
            mitigation_time = 0.030, # Mitigation time in seconds,
            t_alarm = 0.001, #[s]
            high_thr = thr,
            low_thr = thr
        )
        # Eval TPR and FPR for these prediction
        tpr_list.append(tpr_score(y_true_shot_t, y_pred_shot_t))
        fpr_list.append(fpr_score(y_true_shot_t, y_pred_shot_t))
        threshold_list.append(thr)
        # Log
        print(f'Threshold {thr:.2f} has tpr {tpr_list[-1]:.2f} and fpr {fpr_list[-1]:.2f}')

    fpr_s = np.array(fpr_list)[np.argsort(fpr_list)]
    tpr_s = np.array(tpr_list)[np.argsort(fpr_list)]
    auc_score = auc(fpr_s, tpr_s)
    best_idx = np.argsort(-fpr_s + tpr_s +1)[-1]
    best_threshold = np.array(threshold_list)[np.argsort(fpr_list)][best_idx]
    print(f'Zhu AUC score {auc_score} with best threshold of {best_threshold}')


    



def get_alarm_prediction(
        dataloader_custom: dict,
        shots_info_df:pd.DataFrame,
        mitigation_time = 0.030, # Mitigation time in seconds,
        t_alarm: float = 0.001, #[s]
        high_thr: float = 0.9,
        low_thr: float = 0.3
    ):
    # Eval rolling prediction for each shot
    # Note: shot should always be ordered by time
    shot_list = shots_info_df['uid_shot'].unique()
    X_unordered = dataloader_custom['x']

    y_pred_alarm_lst = []
    for shot in shot_list:
        # Random just for demo!!!!!!!!!!
        sel_shot_info = shots_info_df[shots_info_df['uid_shot'] == shot]
        # Take reorder X by shot and time
        filter_dl = dataloader_custom['groups']==shot
        ts_x = dataloader_custom['time'][:,:,-1][filter_dl]
        idx_sort_time = np.argsort(ts_x.flatten())

        sel_shot_X = X_unordered[dataloader_custom['groups']==shot][idx_sort_time]
        sel_shot_y = sel_shot_info['y_true'].values

        # Your model prediction goes here->>>>
        y_pred_record_shot = np.random.uniform(low=0.0, high=1.0, size=sel_shot_X.shape[0])
        # <<<<<------------




        # Eval the alarm state during the shot
        y_pred_alarm = predict_rolling(
                        shot_time=sel_shot_info['time'].values, # Timestamps of the shot in seconds
                        y_pred_record = y_pred_record_shot, # Disruptivity array of the records of the shot
                        t_alarm=t_alarm, # Time after which if plasma is continuosly unstable, raise an alarm [hysteresis window size]
                        high_thr=high_thr,
                        low_thr=low_thr
                        )
        df_s = pd.DataFrame(y_pred_alarm, columns=['predicted_alarm'])
        df_s['uid_shot'] = shot
        df_s['time'] = sel_shot_info['time'].values
        df_s['time_until_disrupt'] = sel_shot_info['time_until_disrupt'].values
        df_s['y_true'] = sel_shot_y
        y_pred_alarm_lst.append(df_s)


    y_pred_alarm_all = pd.concat(y_pred_alarm_lst, axis=0, ignore_index=True)
    
    

    ### THE FOLLOWING STEPS APPLY TO EACH SHOT AND METRICS SHOULD BE EVALUATED ON ALL OF THEM
    
    # Get the prediction of the alarm for the shot
    # y_true_shot is 1 if shot is disruptive, 0 if is not disruptive
    # y_pred_shot is:
    #   1: if alarm is raised in before mitigation time for a disruptive shot/not disruptive shot
    #   1: if alarm is raised in after mitigation time for a not disruptive shot (late alarm for a not disruptive)
    #   0: if the alarm is not raised or raised after mitigation time for a disruptive shot
    y_pred_alarm_all.set_index('uid_shot', inplace=True)
    y_true_shot, y_pred_shot = build_rolling_array(
         y_pred_alarm=y_pred_alarm_all['predicted_alarm'].values,
         time=y_pred_alarm_all[['time']],
         y_true=y_pred_alarm_all['y_true'],
         time_to_disr=y_pred_alarm_all['time_until_disrupt'],
         mitigation_time=mitigation_time
    )
    return y_true_shot, y_pred_shot
     


def predict_rolling(shot_time:np.ndarray, y_pred_record:np.ndarray, t_alarm: float, high_thr: float,
                        low_thr: float):
        """
        The function predict the alarm state of a time series using the predicted 
        failure/anomaly probability

        """
        # Init control system state
        normal = True  # Inital plasma condition
        t0_reset = np.inf  # Never reset alarm if raised
        alarm_state = 0 # Alarm if off
        t0 = t0_reset
        # List of alarm throught the timeseries
        close_to_disruption_pred = []
        # Predict and take only disruption probability
        for t, disr_prob in zip(shot_time, y_pred_record):
            if alarm_state != 1:
                ## Hysteresis
                if (normal and (disr_prob > high_thr)):
                    t0 = t  # Keep trace of the first timestamp when the plasma behaviour is unstable
                    normal = False  # Raise the unstable plasma state (not normal)
                if disr_prob < low_thr:
                    # Reset the unstable plasma state
                    normal = True
                    t0 = t0_reset
                if t - t0 > t_alarm:
                    # If plasma is unstable for more than t_alarm duration, then raise and alarm
                    alarm_state = 1
            # Append the alarma state
            close_to_disruption_pred.append(alarm_state)
        return np.array(close_to_disruption_pred)



def build_rolling_array(y_pred_alarm:np.ndarray, time:pd.DataFrame, y_true:pd.DataFrame, time_to_disr: pd.DataFrame,
                        mitigation_time: float):
        # TRUE -> 0: non disruptive 1: disruptive  (true values are all supposed to be in time)
        gp_true_alm_intime = y_true.groupby('uid_shot').any()
        y_true_ = gp_true_alm_intime.astype(int).values  #
        # PRED -> 0: non disruptive or late alarm 1: disrutpive and alarm in time
        pred_df = time.copy()
        # Save prediciton for each timestamp
        pred_df['disrupt_alarm'] = y_pred_alarm
        # Import time to disruption
        pred_df['time_until_disrupt'] = time_to_disr.values
        # Filter where disrupt alarm is one for each shot and take the first alar in the 
        # any alarm is raised
        if 1 in pred_df.groupby(['disrupt_alarm', 'uid_shot']).first().index.get_level_values(0).unique():
            gp_alm = pred_df.groupby(['disrupt_alarm', 'uid_shot']).first().loc[1]
            # Check if on this selection, time to disrupt is far enough from disruption
            gp_true_nd = gp_alm[gp_alm['time_until_disrupt'].isna()] # Alarm was raised for these shots but time until disruption if NaN [False Positive]
            gp_alm_intime = (gp_alm['time_until_disrupt'] >= mitigation_time)
            gp_alm_late = (gp_alm['time_until_disrupt'] < mitigation_time)
            # All other records are classified as 0, non disruptive becuase shot was not disruptive or alarm raising was late
            gp_not_alm = [x for x in pred_df.reset_index()['uid_shot'].unique() if x not in gp_alm.index]
            # Late alarm are considered as FP is discharge is not disruptive, FN if disruptive
            late_alm_df = gp_alm_late[gp_alm_late].rename('late_alarm_pred').to_frame()
            late_alm_df = late_alm_df.merge(gp_true_alm_intime.rename('is_disruptive'), left_index=True, right_index=True, how='left')
          
            # Then late alarm for Not Disruptive shot are treated as FP
            if late_alm_df.empty:
                pass
            else:
                # Late alarm
                # If late alarm but shot is disruptive, then we keep it as a not raised alarm [@ t_useful/mitigation time]
                late_alm_df.loc[(late_alm_df['late_alarm_pred']) & (late_alm_df['is_disruptive']), 'pred_lbl'] = False
                # If late alarm but shot is not disruptive, then we keep it as a raised alarm [@ t_useful/mitigation time]
                late_alm_df.loc[(late_alm_df['late_alarm_pred']) & (~late_alm_df['is_disruptive']), 'pred_lbl'] = True

            # Concat results and reorder dataset
            if late_alm_df.empty:
                concat_ser = [pd.Series(False, index=gp_not_alm), gp_alm_intime[gp_alm_intime], pd.Series(True, index=gp_true_nd.index)]
            else:
                # We concat not alarmed shot, alarm in time, late alarm, not disruptive alarmed shot (FP)
                concat_ser = [pd.Series(False, index=gp_not_alm), gp_alm_intime[gp_alm_intime], late_alm_df['pred_lbl'], pd.Series(True, index=gp_true_nd.index)]
        else:
            # Not raising any alarm
            gp_not_alm = [x for x in pred_df.reset_index()['uid_shot'].unique()]
            concat_ser = [pd.Series(False, index=gp_not_alm)]
        
        tot_alm = pd.concat(concat_ser)
        y_pred_ = tot_alm.loc[gp_true_alm_intime.index].astype(int).values
        return y_true_, y_pred_

if __name__ =='__main__':
     main()