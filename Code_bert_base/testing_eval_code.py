import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from training_code import *
from load_data import initialize_eval_test
from reading_datasets import read_task
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(model_load_location):
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../Datasets/Subtask_1a/training/'
    test_data = read_task(dataset_location , split = 'dev')

    labels_to_ids = task7_labels_to_ids
    input_data = (test_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time

    tokenizer = AutoTokenizer.from_pretrained(model_load_location)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

    # unshuffled testing data
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    # Getting testing dataloaders
    test_loader= initialize_eval_test(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = False)

    start = time.time()

    # Run the model with unshuffled testing data
    test_result, overall_f1, overall_precision, overall_recall, overall_accuracy, overall_cr_df, overall_cm_df, eval_logits  = val_testing(model, test_loader, labels_to_ids, device)

    print('EVAL TEST ACC:', overall_accuracy)
    print('EVAL TEST F1:', overall_f1)
    print('EVAL TEST PRECISION:', overall_precision)
    print('EVAL TEST RECALL:', overall_recall)

    now = time.time()

    print('TIME TO COMPLETE:', (now-start)/60 )
    print()

    return test_result, overall_f1, overall_precision, overall_recall, overall_accuracy, overall_cr_df, overall_cm_df, eval_logits



if __name__ == '__main__':
    train_val_start_time = time.time()
    n_rounds = 3
    models = ['bert-large-uncased', 'roberta-large']

    # setting up the arrays to save data for all loops, models, and epochs
    # accuracy
    all_best_acc = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_f1_score = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_precision = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_recall = pd.DataFrame(index=range(n_rounds), columns=models)

    for loop_index in range(n_rounds):
        for model_name in models:
            test_print_statement = 'Testing ' + model_name + ' from loop ' + str(loop_index)
            print(test_print_statement)

            model_load_location = '../2nd_try_15_epochs_large_model/saved_models_1a/' + model_name + '/' + str(loop_index) + '/' 
            
            result_save_location = '../2nd_try_15_epochs_large_model/eval_testing/saved_eval_test_result_1a/' + model_name + '/' + str(loop_index) + '/'
            report_result_save_location = '../2nd_try_15_epochs_large_model/eval_testing/saved_eval_report_1a/' + model_name + '/' + str(loop_index) + '/'

            unformatted_result_save_location = result_save_location + 'unformatted_eval_test_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_eval_test_result.tsv'

            test_result, best_f1, best_precision, best_recall, best_accuracy, best_cr_df, best_cm_df, eval_logits = main(model_load_location)

            # Getting best f1, precision, and recall, accuracy
            all_best_acc.at[loop_index, model_name] = best_accuracy
            all_best_f1_score.at[loop_index, model_name] = best_f1
            all_best_precision.at[loop_index, model_name] = best_precision
            all_best_recall.at[loop_index, model_name] = best_recall

            os.makedirs(report_result_save_location, exist_ok=True)
            cr_df_location = report_result_save_location + 'classification_report.tsv'
            cm_df_location = report_result_save_location + 'confusion_matrix.tsv'
            eval_logits_location = report_result_save_location + 'eval_logits.tsv'
            
            format_eval_logits = pd.DataFrame(eval_logits, columns=['0', '1', '2'])
            format_eval_logits.to_csv(eval_logits_location, sep='\t')
        
            best_cr_df.to_csv(cr_df_location, sep='\t')
            best_cm_df.to_csv(cm_df_location, sep='\t')

            print("\n Testing results")
            print(test_result)
            formatted_test_result = test_result.drop(columns=['text'])

            os.makedirs(result_save_location, exist_ok=True)
            test_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_test_result.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Result files saved")

    # printing results for analysis

    print("\n All best dev acc")
    print(all_best_acc)

    print("\n All best f1 score")
    print(all_best_f1_score)

    print("\n All best precision")
    print(all_best_precision)

    print("\n All best recall")
    print(all_best_recall)

    #saving all results into tsv

    os.makedirs('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/', exist_ok=True)
    all_best_acc.to_csv('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/all_best_dev_acc.tsv', sep='\t')
    all_best_f1_score.to_csv('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/all_best_f1_score.tsv', sep='\t')
    all_best_precision.to_csv('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/all_best_precision.tsv', sep='\t')
    all_best_recall.to_csv('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/all_best_recall.tsv', sep='\t')

    train_val_end_time = time.time()

    total_time = (train_val_end_time - train_val_start_time) / 60
    print("Everything successfully completed")
    print("Time to complete:", total_time) 

    with open('../2nd_try_15_epochs_large_model/eval_testing/eval_validation_statistics/time.txt', 'w') as file:
        file.write("Time to complete: ")
        file.write(str(total_time))
        file.write(" mins")










    
        