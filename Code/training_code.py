import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from load_data import initialize_data
from reading_datasets import read_task
from labels_to_ids import task7_labels_to_ids
import time
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def train(epoch, training_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    optimizer.zero_grad()
    
    for idx, batch in enumerate(training_loader):
        if batch:
            ids = batch['new_input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            if (idx + 1) % 20 == 0:
                print('FINSIHED BATCH:', idx, 'of', len(training_loader))

            #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, token_type_ids=None, attention_mask=mask, labels=labels)
    
            tr_loss += output[0]

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)
            
            # compute training accuracy
            predictions = torch.argmax(output.logits, axis=2) # shape (batch_size * seq_len,)
            
            labels_formatted = labels.detach().cpu().numpy().tolist()
            predictions_formatted = predictions.detach().cpu().numpy().tolist()

            tmp_batch_accuracy = 0

            for sentence in range(len(labels_formatted)):
                tmp_batch_accuracy += accuracy_score(labels_formatted[sentence], predictions_formatted[sentence])
                
            tmp_tr_accuracy = tmp_batch_accuracy / len(labels_formatted)
            
            tr_accuracy += tmp_tr_accuracy
            # only compute accuracy at active labels
            # active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
            
            
            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            # tmp_tr_accuracy = accuracy_score(labels.detach().cpu().numpy().tolist(), predictions.detach().cpu().numpy().tolist())
            # tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )
            
            # backward pass
            output['loss'].backward()
            if (idx + 1) % grad_step == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = tr_loss / nb_tr_steps
        # tr_accuracy = tr_accuracy / nb_tr_steps
        #print(f"Training loss epoch: {epoch_loss}")
        #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def validate(model, testing_loader, labels_to_ids, device, tokenizer):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    
    eval_f1, eval_precision, eval_recall = 0, 0, 0

    eval_preds, eval_labels = [], []
    eval_tweet_ids, eval_orig_sentences, eval_class_labels = [], [], []
    eval_begin_values, eval_end_values, eval_orig_spans = [], [], []
    eval_original_converted_spans, eval_predicted_converted_spans = [], []
    eval_predicted_begin, eval_predicted_end = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['new_input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            # to attach back to prediction data later 
            tweet_ids = batch['tweet_id']
            orig_sentences = batch['orig_sentence']
            begin_values = batch['begin']
            end_values = batch['end']
            orig_spans = batch['orig_span']

            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, token_type_ids=None, attention_mask=mask, labels=labels)
      
            eval_loss += output['loss'].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
            
            predictions = torch.argmax(output.logits, axis=2) # shape (batch_size * seq_len,)
            
            
            # extending the original labels
            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            # Getting the formatted labels
            labels_formatted = labels.detach().cpu().numpy().tolist()
            predictions_formatted = predictions.detach().cpu().numpy().tolist()



            # gettng the span values
            batch_prediction_data = pd.DataFrame(zip(ids.cpu().numpy(), labels_formatted, predictions_formatted, orig_sentences), columns=['id', 'label', 'prediction', 'orig_sentence'])

            # finding the matching token
            original_label_span, prediction_label_span, prediction_begin, prediction_end = find_matching_token(batch_prediction_data, tokenizer)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            eval_original_converted_spans.extend(original_label_span)
            eval_predicted_converted_spans.extend(prediction_label_span)
            eval_predicted_begin.extend(prediction_begin)
            eval_predicted_end.extend(prediction_end)
            
            # eval_class_labels.extend(class_labels)
            eval_begin_values.extend(begin_values)
            eval_end_values.extend(end_values)
            eval_orig_spans.extend(orig_spans)
            eval_tweet_ids.extend(tweet_ids)
            eval_orig_sentences.extend(orig_sentences)

            tmp_batch_accuracy = 0

            for sentence in range(len(labels_formatted)):
                tmp_batch_accuracy += accuracy_score(labels_formatted[sentence], predictions_formatted[sentence])
                
            tmp_eval_accuracy = tmp_batch_accuracy / len(labels_formatted)
            
            eval_accuracy += tmp_eval_accuracy

    print("Eval labels")
    print(eval_labels)

    print("Eval preds")
    print(eval_preds)
  
    # labels = [id.item() for id in eval_labels]
    # predictions = [ids_to_labels[id.item()] for id in eval_preds]

    # Concatenating all data together into a single table
    overall_prediction_data = pd.DataFrame(zip(eval_tweet_ids, eval_orig_sentences, eval_class_labels, eval_begin_values, eval_end_values, eval_orig_spans, eval_original_converted_spans, eval_predicted_converted_spans, eval_predicted_begin, eval_predicted_end), columns=['id', 'text', 'class', 'begin', 'end', 'Orig', 'Label_convert', 'Span_convert', 'begin', 'end'])
    
    overall_prediction_data.to_csv("test_result.tsv", sep='\t')
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return overall_prediction_data, labels, predictions, eval_accuracy, eval_f1, eval_precision, eval_recall


def find_matching_token(batch_prediction_df, tokenizer):
    token_original_output = []
    token_prediction_output = []
    token_prediction_begin = []
    token_prediction_end = []
    for index, row in batch_prediction_df.iterrows():
        tmp_token_original = []
        tmp_token_prediction = []
    
        for j in range(len(row['id'])):
            if row['label'][j] == 1 and row['id'][j] != 2: 
                tmp_token_original.append(row['id'][j])

        for j in range(len(row['id'])):
            if row['prediction'][j] == 1 and row['id'][j] != 2: 
                tmp_token_prediction.append(row['id'][j])

        # converting token to string 
        original_span_token = tokenizer.convert_ids_to_tokens(tmp_token_original)
        predicted_span_token = tokenizer.convert_ids_to_tokens(tmp_token_prediction)

        original_span_string = tokenizer.convert_tokens_to_string(original_span_token)
        predicted_span_string = tokenizer.convert_tokens_to_string(predicted_span_token)

        print("Original span token")
        print(original_span_token)
        print("Predicted span token")
        print(predicted_span_token)

        print("Original span string")
        print(original_span_string)
        print("Predicted span string")
        print(predicted_span_string)
        
        # if prediction came out to no result
        if predicted_span_token == []:
            token_prediction_begin.append(0)
            token_prediction_end.append(0)
            token_original_output.append(original_span_string)
            token_prediction_output.append("")
        else: # if prediction came out to some word
            # append the original sentence to the list
            token_original_output.append(original_span_string)
            token_prediction_output.append(predicted_span_string)
            all_words_in_span = predicted_span_string.split()
            first_word_in_predicted_span_token = all_words_in_span[0]
            last_word_in_predicted_span_token = all_words_in_span[-1]

            original_sentence = row['orig_sentence']
            span_begin = re.search(first_word_in_predicted_span_token, original_sentence)
            token_prediction_begin.append(span_begin.start())

            rest_of_sentence = original_sentence[span_begin.start():]
            span_end = re.search(last_word_in_predicted_span_token, rest_of_sentence)
            token_prediction_end.append(span_end.end())
            

    print("Finished parsing everything")
        
    return token_original_output, token_prediction_output, token_prediction_begin, token_prediction_end

            

def main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location, report_result_save_location):
    #Initialization training parameters
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../Datasets/Subtask_1b/training/'

    train_data = read_task(dataset_location , split = 'train')
    dev_data = read_task(dataset_location , split = 'dev')

    labels_to_ids = task7_labels_to_ids
    input_data = (train_data, dev_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    if model_load_flag:
        tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        model = BertForTokenClassification.from_pretrained(model_load_location)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader = initialize_data(tokenizer, initialization_input, train_data, labels_to_ids, shuffle = True)
    dev_loader = initialize_data(tokenizer, initialization_input, dev_data, labels_to_ids, shuffle = True)

    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1

    best_f1 = 0
    best_precision = 0
    best_recall = 0

    all_epoch_data = pd.DataFrame(index=[0,1,2,3,4,5,6,7,8,9], columns=['dev_accuracy', 'dev_f1', 'dev_precision', 'dev_recall'])

    best_overall_prediction_data = []

    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model = train(epoch, train_loader, model, optimizer, device, grad_step)
        print("Finished training")
        
        #testing and logging
        dev_overall_prediction, labels_dev, predictions_dev, dev_accuracy, dev_f1, dev_precision, dev_recall = validate(model, dev_loader, labels_to_ids, device, tokenizer)
        print("Finished evaluating")

        print('DEV ACC:', dev_accuracy)
        print('DEV F1:', dev_f1)
        print('DEV PRECISION:', dev_precision)
        print('DEV RECALL:', dev_recall)
        
        all_epoch_data.at[epoch, 'dev_accuracy'] = dev_accuracy

        all_epoch_data.at[epoch, 'dev_f1'] = dev_f1
        all_epoch_data.at[epoch, 'dev_precision'] = dev_precision
        all_epoch_data.at[epoch, 'dev_recall'] = dev_recall

        report_result_save_location = report_result_save_location + '/epoch_' + str(epoch) + '/'

        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            best_f1 = dev_f1
            best_precision = dev_precision
            best_recall = best_recall

            #best_test_acc = test_accuracy
            best_epoch = epoch
            
            best_overall_prediction_data = dev_overall_prediction
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)

        '''if best_tb_acc < test_accuracy_tb:
            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch'''

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5))
        print('BEST F1 --> ', 'DEV:', best_f1)
        print('BEST PRECISION --> ', 'DEV:', best_precision)
        print('BEST RECALL --> ', 'DEV:',  best_recall)
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_overall_prediction_data, best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_f1, best_precision, best_recall, all_epoch_data





if __name__ == '__main__':
    train_val_start_time = time.time()
    n_epochs = 2
    models = ['bert-base-uncased']
    
    #model saving parameters
    model_save_flag = True
    model_load_flag = False

    # setting up the arrays to save data for all loops, models, and epochs
    # accuracy
    all_best_dev_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_test_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_tb_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    
    # epoch
    all_best_epoch = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_tb_epoch = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # factors to calculate final f1 performance metric
    all_best_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_precision = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_recall = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    for loop_index in range(1):
        for model_name in models:
            print('Running loop', loop_index)
            print()

            model_save_location = '../saved_models_1b/' + model_name + '/' + str(loop_index) + '/' 
            model_load_location = None

            epoch_save_location = '../saved_epoch_1b/' + model_name + '/' + str(loop_index) + '/' 
            epoch_save_name = epoch_save_location + '/epoch_info.tsv'

            result_save_location = '../saved_data_1b/' + model_name + '/' + str(loop_index) + '/'

            report_result_save_location = '../saved_report_1b/' + model_name + '/' + str(loop_index)

            unformatted_result_save_location = result_save_location + 'unformatted_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_result.tsv'

            best_prediction_result, best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_f1_score, best_precision, best_recall, epoch_data = main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location, report_result_save_location)

            # Getting accuracy
            all_best_dev_acc.at[loop_index, model_name] = best_dev_acc
            all_best_test_acc.at[loop_index, model_name] = best_test_acc
            all_best_tb_acc.at[loop_index, model_name] = best_tb_acc
            
            # Getting best epoch data
            all_best_epoch.at[loop_index, model_name] = best_epoch
            all_best_tb_epoch.at[loop_index, model_name] = best_tb_epoch

            # Getting best f1, precision, and recall
            all_best_f1_score.at[loop_index, model_name] = best_f1_score
            all_best_precision.at[loop_index, model_name] = best_precision
            all_best_recall.at[loop_index, model_name] = best_recall

            # Get all epoch info 
            os.makedirs(epoch_save_location, exist_ok=True)
            epoch_data.to_csv(epoch_save_name, sep='\t')

            print("\n Prediction results")
            print(best_prediction_result)

            # formatted_prediction_result = best_prediction_result.drop(columns=['Orig', 'text'])
            formatted_prediction_result = best_prediction_result

            # os.makedirs(result_save_location, exist_ok=True)
            # best_prediction_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            # formatted_prediction_result.to_csv(formatted_result_save_location, sep='\t', index=False, header=False)

            print("Result files saved")

    print("\n All best dev acc")
    print(all_best_dev_acc)

    print("\n All best f1 score")
    print(all_best_f1_score)

    print("\n All best precision")
    print(all_best_precision)

    print("\n All best recall")
    print(all_best_recall)

    #saving all results into tsv

    os.makedirs('../validating_statistics/', exist_ok=True)
    all_best_dev_acc.to_csv('../validating_statistics/all_best_dev_acc.tsv', sep='\t')
    all_best_f1_score.to_csv('../validating_statistics/all_best_f1_score.tsv', sep='\t')
    all_best_precision.to_csv('../validating_statistics/all_best_precision.tsv', sep='\t')
    all_best_recall.to_csv('../validating_statistics/all_best_recall.tsv', sep='\t')

    train_val_end_time = time.time()

    total_time = (train_val_end_time - train_val_start_time) / 60
    print("Everything successfully completed")
    print("Time to complete:", total_time)






