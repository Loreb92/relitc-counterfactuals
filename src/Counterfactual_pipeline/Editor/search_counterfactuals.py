import re
import numpy as np


def run_binary_search(Editor, 
                      Blackbox,
                      dataset, 
                      idx, 
                      min_mask_frac, 
                      max_mask_frac, 
                      max_search_levels=4, 
                      n_edited_texts=15, 
                      results=None):
    
    """
    Searches for the optimal mask fraction to generate counterfactuals through a binary search.
    
    Parameters:
    Editor : 
    Blackbox : 
    dataset : 
    idx : 
    min_mask_frac : 
    max_mask_frac : 
    max_search_levels : 
    n_edited_texts : 
    results :
    
    Returns:
    results : list of dict
    """
    
    # set mask fraction
    mask_frac_tokens = (max_mask_frac - min_mask_frac) / 2 + min_mask_frac
    dataset.frac_mask_words = mask_frac_tokens
    
    # get instance (it is already masked)
    encoded_masked_text = dataset[idx]
    original_text = dataset.df.loc[idx].text
    original_label = int(dataset.df.loc[idx].label)
    original_pred_proba = dataset.df.loc[idx].pred_proba
    counterf_label_name = dataset.id2label[original_label]
    
    # make input for Editor
    # we create this list of same elements for convenience, in case in a later stage we want to use input texts masked differently
    editor_encoded_inputs = [encoded_masked_text] * n_edited_texts
    masked_text = Editor.tokenizer.decode(encoded_masked_text['input_ids'])
    masked_texts = [masked_text] * n_edited_texts
        
    # generate candidate counterfactuals by infilling mask tokens
    edited_outputs = Editor(editor_encoded_inputs)
    
    # decode
    edited_outputs_text = [Editor.tokenizer.decode(out, skip_special_tokens=True) for out in edited_outputs]
    
    if dataset.use_ctrl_code:
        # remove ctrl code
        ctrl_code = Editor.tokenizer.decode(Editor.tokenizer(counterf_label_name.title())['input_ids'], skip_special_tokens=True)
        #edited_outputs_text = [re.sub(f"(?i){counterf_label_name.title()}\s:\s", "", out).strip() for out in edited_outputs_text]
        edited_outputs_text = [re.sub(f"(?i){ctrl_code.title()}\s:\s", "", out).strip() for out in edited_outputs_text]
            
    # run classifier and check if success
    # check if at least one successfull counterfactual
    preds = Blackbox(edited_outputs_text)
    
    # decrement search steps of binary search
    max_search_levels -= 1
    
    # store results
    txts_unique = set()
    results_run = []
    is_successful_run = False
    for edited_output_text, masked_text, pred_proba in zip(edited_outputs_text, masked_texts, preds[:,1].tolist()):
        
        # if the edited text is duplicated, skip
        if edited_output_text in txts_unique:
            continue
        txts_unique.add(edited_output_text)
        
        pred_class = int(pred_proba > 0.5)
        is_success = bool(original_label != pred_class)
        if is_success:
            is_successful_run = True
        
        result_ = {
            'masked_text':masked_text,
            'edited_output_text':edited_output_text,
            'pred_proba':pred_proba,
            'pred_class':pred_class,
            'success':is_success,
            'mask_frac':mask_frac_tokens,
        }
        
        results_run.append(result_)
        
        
    if results is None:
        results = []
        
    # TODO: add counterfactual_label
    results.append({'original_text':original_text, 'mask_frac':mask_frac_tokens, 'original_label':original_label, 'original_pred_proba':original_pred_proba, 'counterfactual_label_name':counterf_label_name, 'success':is_successful_run, 'results':results_run})
        
    if max_search_levels > 0:
        if is_successful_run:
            results = run_binary_search(Editor=Editor, 
                                          Blackbox=Blackbox,
                                          dataset=dataset, 
                                          idx=idx, 
                                          min_mask_frac=min_mask_frac, 
                                          max_mask_frac=mask_frac_tokens, 
                                          max_search_levels=max_search_levels, 
                                          n_edited_texts=n_edited_texts, 
                                          results=results)
        else:
            results = run_binary_search(Editor=Editor, 
                                          Blackbox=Blackbox,
                                          dataset=dataset, 
                                          idx=idx, 
                                          min_mask_frac=mask_frac_tokens, 
                                          max_mask_frac=max_mask_frac, 
                                          max_search_levels=max_search_levels, 
                                          n_edited_texts=n_edited_texts, 
                                          results=results)
            
    return results
    
    
    
    
    
                                          
    
    
    
    
    
    