from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np
import torch


class dataset(Dataset):
    def __init__(self, all_data, tokenizer, labels_to_ids, max_len, for_training):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.for_training = for_training

    def __getitem__ (self, index):
        tweet_id = self.data[index][0]
        sentence = self.data[index][1]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        # step 4: if it is for training, get input labels as well
        if self.for_training:
            
            begin = self.data[index][2]
            end = self.data[index][3]
            span = self.data[index][4]
            
            # tokenizing the span itself
            span_encoding = self.tokenizer(span,
                                        return_offsets_mapping=True,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_len)

            # Creating an array for the combined label and adding it to dataset
            label_match = np.array([0] * len(encoding.input_ids))
            span_label = combine_labels(encoding.input_ids, span_encoding.input_ids, label_match)
            item['labels'] = torch.as_tensor(span_label)

            # Storing everything else to dataset
            item['begin'] = begin
            item['end'] = end
            item['orig_span'] = span
            item['orig_sentence'] = sentence

        return item

    def __len__(self):
        return self.len


def combine_labels(encoding, span_encoding, label_match):

    for i in range(len(span_encoding)):
        for j in range(len(encoding)):
            if span_encoding[i] == encoding[j] and encoding[j] != 0 and span_encoding[i] != 101 and span_encoding[i] != 102:
                k = i
                l = j

                # check if the span detection is one single phrase - if the word does not match before
                in_single_phrase = 1
                while span_encoding[k] != 102:
                    if encoding[l] != span_encoding[k]:
                        in_single_phrase = 0
                        break
                    k += 1
                    l += 1

                m = 0
                if in_single_phrase == 1:
                    while span_encoding[i + m] != 102:
                        label_match[j + m] = 1
                        m += 1

                    break

    return label_match


def initialize_data(tokenizer, initialization_input, input_data, labels_to_ids, shuffle=True):
    max_len, batch_size = initialization_input
    data_split = dataset(input_data, tokenizer, labels_to_ids, max_len, True)


    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': 4
              }

    loader = DataLoader(data_split, **params)
    
    return loader


def initialize_test(tokenizer, initialization_input, input_data, labels_to_ids, shuffle=False):
    max_len, batch_size = initialization_input
    data_split = dataset(input_data, tokenizer, labels_to_ids, max_len, False)


    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': 4
              }

    loader = DataLoader(data_split, **params)
    
    return loader


if __name__ == '__main__':
    pass
