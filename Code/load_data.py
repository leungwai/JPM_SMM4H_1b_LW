from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np
import torch


class dataset(Dataset):
  def __init__(self, all_data, tokenizer, labels_to_ids, max_len):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data[index][0]
        #joined_sentnece = ' '.join(sentence)
        input_label = self.data[index][1]
        begin = int(self.data[index][2])
        end = int(self.data[index][3])
        span = self.data[index][4]

        # add [SPAN] at the beginning and end of the sentence
        
        
        # span classification - like a normal classification
        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
      #   print("Regular Encoding \n")
      #   print(encoding)
        # tokenizing the span itself
        span_encoding = self.tokenizer(span,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
       
      #   print("Span Encoding \n")
      #   print(span_encoding)

        # Creating an array for the combined label 
        label_match = np.array(['O'] * len(encoding.input_ids))
        matched_keywords = 0

      #   print("Printing label match \n")
      #   print(label_match)

        combine_results = combine_labels(encoding.input_ids, span_encoding.input_ids, label_match, matched_keywords)
        
        label_match = combine_results[0]
        matched_keywords = combine_results[1]
      #   print("After label matching: \n")
      #   print(label_match)

      
      #   print("\n Matched Keywords: \n")
      #   print(matched_keywords)
               
        labels = self.labels_to_ids[input_label]

        final_match_label = convert_labels(label_match)

      #   print("\n Matched keywords converted: \n")
      #   print(final_match_label)
        
        encoding.input_ids = convert_initial_tokens(encoding.input_ids)
      #   print("Printing encoding input_ids: \n")
      #   print(encoding.input_ids)
        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        
        
        item['labels'] = torch.as_tensor(labels)
      #   item['begin'] = torch.as_tensor(begin)
      #   item['end'] = torch.as_tensor(end)
        item['target'] = torch.as_tensor(final_match_label)
        

        print("Encoding Input IDs: ")
        print(encoding.input_ids)

        print("\n Label Matched IDs:")
        print(final_match_label)
        quit()
        

      #   print("Final dataloader: \n")
      #   print(item)
      #   print("\n --------------------------------------------- \n")
      #   quit()
        return item

  def __len__(self):
        return self.len

def combine_labels(encoding, span_encoding, label_match, matched_keywords):
      end = 0
      for x in range(len(encoding)):
            if encoding[x] == 102:
                  end = 1
                  continue
            if end == 1:
                label_match[x] = "P"
                  

      for i in range(len(span_encoding)):
            for j in range(len(encoding)):
                  if span_encoding[i] == encoding[j] and encoding[j] != 0 and span_encoding[i] != 101 and span_encoding[i] != 102:
                        k = i
                        l = j
                        in_single_phrase = 1
                        while span_encoding[k] != 102:
                              if encoding[l] != span_encoding[k]:
                                    in_single_phrase = 0
                                    break
                              k+=1
                              l+=1

                        m = 0
                        if in_single_phrase == 1:
                              while span_encoding[i + m] != 102:
                                    label_match[j + m] = 'B'
                                    matched_keywords+=1
                                    m+=1
                              
                              return label_match, matched_keywords

def convert_initial_tokens(encoding):
      final_label_match = [] 
      end = 0
      for x in range(len(encoding)):
            if encoding[x] == 102:
                  final_label_match.append(encoding[x])
                  end = 1
            elif end == 1:
                  final_label_match.append(2)
            else:
                  final_label_match.append(encoding[x])
      return final_label_match



def convert_labels(label_match):
      final_label_match = []
      for i in range(len(label_match)):
            if label_match[i] == 'O':
                  final_label_match.append(0)
            elif label_match[i] == 'B':
                  final_label_match.append(1)
            elif label_match[i] == 'P':
                  final_label_match.append(2)
      
      return final_label_match
                  
                                               
def initialize_data(tokenizer, initialization_input, input_data, labels_to_ids, shuffle = True):
    max_len, batch_size = initialization_input
    data_split = dataset(input_data, tokenizer, labels_to_ids, max_len)


    params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 4
                }

    loader = DataLoader(data_split, **params)

    return loader




if __name__ == '__main__':
  pass