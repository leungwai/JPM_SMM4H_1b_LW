import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from training_code import *
from load_data import initialize_test
from reading_datasets import read_test
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(model_load_location, report_result_save_location):
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
    test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = False)

    test_ind_f1 = 0
    test_ind_precision = 0
    test_ind_recall = 0

    start = time.time()

    # Run the model with unshuffled testing data
    test_result, test_labels, test_predictions, test_accuracy, test_f1, test_precision, test_recall, test_overall_cr_df, test_overall_cm_df = validate(model, test_loader, labels_to_ids, device)
    
    print('DEV ACC:', test_accuracy)

    print(' ')
    print('test_f1:', test_f1)
    print('test_precision:', test_precision)
    print('test_recall:', test_recall)   

    os.makedirs(report_result_save_location, exist_ok=True)
    cr_df_location = report_result_save_location + 'classification_report.tsv'
    cm_df_location = report_result_save_location + 'confusion_matrix.tsv'

    test_overall_cr_df.to_csv(cr_df_location, sep='\t')
    test_overall_cm_df.to_csv(cm_df_location, sep='\t')

    now = time.time()

    print('TIME TO COMPLETE:', (now-start)/60 )
    print()

    return test_result, test_accuracy, test_f1, test_precision, test_recall

if __name__ == '__main__':
    n_epochs = 1
    models = ['bert-base-uncased']

    # setting up the arrays to save data for all loops, models,

    # dev and test acc
    all_test_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # factors to calculate final f1 performance metric
    all_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_precision = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_recall = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    for loop_index in range(5):
        for model_name in models:
            test_print_statement = 'Testing ' + model_name + ' from loop ' + str(loop_index)
            print(test_print_statement)

            model_load_location = '../saved_models_1b/' + model_name + '/' + str(loop_index) + '/' 
            
            result_save_location = '../saved_test_result_1b/' + model_name + '/' + str(loop_index) + '/'
            
            report_result_save_location = '../saved_test_report_1b/' + model_name + '/' + str(loop_index)

            unformatted_result_save_location = result_save_location + 'unformatted_test_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_test_result.tsv'

            test_result, test_acc, test_f1_score, test_precision, test_recall = main(model_load_location, report_result_save_location)

            # Getting accuracy
            all_test_acc.at[loop_index, model_name] = test_acc

            # Getting best individual data (by category)
            all_f1_score.at[loop_index, model_name] = test_f1_score
            all_precision.at[loop_index, model_name] = test_precision
            all_recall.at[loop_index, model_name] = test_recall

            print("\n Testing results")
            print(test_result)
            formatted_test_result = test_result.drop(columns=['text', 'Orig'])

            os.makedirs(result_save_location, exist_ok=True)
            test_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_test_result.to_csv(formatted_result_save_location, sep='\t', index=False, header=False)

            print("Result files saved")

    print("\n All best overall f1 score")
    print(all_f1_score)

    print("\n All best f1 score")
    print(all_f1_score)

    print("\n All best precision")
    print(all_precision)

    print("\n All best recall")
    print(all_recall)   

    #saving all results into tsv

    os.makedirs('../testing_statistics/', exist_ok=True)
    all_f1_score.to_csv('../testing_statistics/all_f1_score.tsv', sep='\t')
    all_test_acc.to_csv('../testing_statistics/all_test_acc.tsv', sep='\t')
    all_f1_score.to_csv('../testing_statistics/all_f1_score.tsv', sep='\t')
    all_precision.to_csv('../testing_statistics/all_precision.tsv', sep='\t')
    all_recall.to_csv('../testing_statistics/all_recall.tsv', sep='\t')     

    print("Everything successfully completed")













    
        