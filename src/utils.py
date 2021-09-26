import re
import json
import os


# очистка текста
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r"[\"#/@;:<>{}`+=~|?,]", "", text)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text


# добавление json в csv
def read_append_return(filename, train_files_path, output='text', keep_list=False):
    json_path = os.path.join(train_files_path, (filename + '.json'))
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
            combined.append(data.get('section_title'))
            combined.append(data.get('text'))

    if not keep_list:
        headings = ' '.join(headings)
        contents = ' '.join(contents)
        combined = '\n\n '.join(combined)

    if output == 'text':
        return contents
    elif output == 'head':
        return headings
    else:
        return combined


# приведение таргета к необходимому виду
def prepare_labels(data):
    datasets_titles = [x.lower() for x in
                       set(data['dataset_title'].unique()).union(set(data['dataset_label'].unique()))]

    labels = []
    for index in data['Id']:
        publication_text = data[data['Id'] == index].text.str.cat(sep='\n').lower()
        label = []
        for dataset_title in datasets_titles:
            if dataset_title in publication_text:
                label.append(preprocess_text(dataset_title))
        labels.append(' | '.join(label))

    return labels
