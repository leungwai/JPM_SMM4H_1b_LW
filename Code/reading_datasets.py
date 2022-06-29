import pickle
import json
import csv

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_task(location, split = 'train'):
    filename = location + split + '.tsv'

    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i > 0:
                tweet_id = row[2]
                sentence = row[7].strip()
                begin = row[4]
                end = row[5]
                span = row[6]
                label = row[3]
                data.append((sentence, label, begin, end, span))

    return data


if __name__ == '__main__':
    location = '../Datasets/Subtask_1b/training/'
    split = 'train'
    
    data = read_task(location, split)
    print(len(data))

    data = read_task(location, 'dev')
    print(len(data))

    




