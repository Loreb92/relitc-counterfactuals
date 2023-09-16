import numpy as np
from scipy import stats
from datetime import datetime


def print_msg(msg, with_time=True):
    
    if with_time:
        now = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        msg = now + " : " + msg
    print(msg, flush=True)


def shuffle_array(arr):
    '''
    Returns an array in which the values are shuffled
    
    Paramters:
    arr : np.array, the array we want to shuffle
    
    Returns:
    arr_copy : np.array, array with same values of arr but in different order
    '''
    arr_copy = arr.copy()
    np.random.shuffle(arr_copy)
    return arr_copy


def bootstrap_CI(arr, alpha=0.05, n_iters=999):
    
    btst_result = stats.bootstrap((arr, ), statistic=np.mean, vectorized=False)
    
    avg = np.mean(arr)
    lb = btst_result.confidence_interval.low
    ub = btst_result.confidence_interval.high
    
    return {'avg':avg, 'lb':lb, 'ub':ub}
            
    
def get_counterfactuals_scores(results_df, metrics=['fluency', 'sem_similarity', 'minimality'], 
                                      criterions=['mask_frac', 'minimality']):
    """
    Aggregate the scores for each instance. It computes both aggregate scores for counterfactuals with minimum
    mask_frac and with minumum minimality. 
    Note thah for experiments using the BERT CMLM, the two results are the same.
    
    Returns:
    results_df : pandas DataFrame, the input df with additional columns corresponding to the metrics.
    
    """
    
    counterfactuals_df = results_df[['id', 'counterfactuals']].copy()
    
    for criterion in criterions: 
                
        if criterion == 'minimality':
            def _get_ctfs_with_min_minimality(ctfs):
                if ctfs is not None:
                    min_minimality = min(ctfs, key=lambda item: item['minimality'])['minimality']
                    return [ctf for ctf in ctfs if ctf['minimality']==min_minimality ]
                else:
                    return None

            counterfactuals_df.counterfactuals = counterfactuals_df.counterfactuals.apply(lambda ctfs: 
                                                          _get_ctfs_with_min_minimality(ctfs) )
        elif criterion == 'mask_frac':
            pass
    
        for metric in metrics:  
            if metric == 'mask_frac':
                continue
            
            # get aggregate metric
            counterfactuals_df.loc[:, f'{metric}-{criterion}'] = \
                            counterfactuals_df.counterfactuals.apply(lambda ctfs: 
                                           np.mean([ctf[metric] for ctf in ctfs]) 
                                                                     if ctfs is not None else ctfs)
            
            
    results_df = results_df.merge(counterfactuals_df.drop(columns=['counterfactuals']), on='id', how='left')
    return results_df
            

def aggregate_scores(results_df, metrics=['fluency', 'sem_similarity', 'minimality'], 
                     criterions=['mask_frac', 'minimality'],
                    wrt_class='original_label',
                    exception_for_mask_frac=True):
    
    results_df_class_0 = results_df[results_df[wrt_class] == 0 ]
    results_df_class_1 = results_df[results_df[wrt_class] == 1 ]
    
    scores = {}
    for criterion in criterions:
        for metric in metrics:
            
            if exception_for_mask_frac and metric=='mask_frac':
                # get average and confint
                scores[f'{metric}-{criterion}-{wrt_class}'] = bootstrap_CI(results_df['mask_frac'].values)
                # get scores class-wise
                scores[f'{metric}-{criterion}-{wrt_class}-class_0'] = \
                                        bootstrap_CI(results_df_class_0['mask_frac'].values)
                scores[f'{metric}-{criterion}-{wrt_class}-class_1'] = \
                                        bootstrap_CI(results_df_class_1['mask_frac'].values)
                continue
                

            # get average and confint
            scores[f'{metric}-{criterion}-{wrt_class}'] = bootstrap_CI(results_df[f'{metric}-{criterion}'].values)
            
            # get scores class-wise
            scores[f'{metric}-{criterion}-{wrt_class}-class_0'] = \
                                    bootstrap_CI(results_df_class_0[f'{metric}-{criterion}'].values)
            scores[f'{metric}-{criterion}-{wrt_class}-class_1'] = \
                                    bootstrap_CI(results_df_class_1[f'{metric}-{criterion}'].values)
            
    return scores