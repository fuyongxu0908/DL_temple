import codecs
import json
datasets = ["dialogue_nli_train.jsonl", "dialogue_nli_test.jsonl", "dialogue_nli_dev.jsonl"]
for dataset in datasets:
    with codecs.open(dataset, "r", "utf8") as r:
        data = json.load(r)
    with codecs.open("processed/"+dataset, "w", "utf8") as fw:
        for dic in data:
            try:
                label = dic["label"]
                if label == "negative":
                    label = 0
                elif label == "neutral":
                    label = 1
                elif label == "positive":
                    label = 2
                sentence1 = dic["sentence1"]
                sentence2 = dic["sentence2"]
                corpus = sentence1 + sentence2
                triple1 = dic["triple1"]
                triple2 = dic["triple2"]
                dtype = dic["dtype"]
                id = dic["id"]
                df = {"label": label, "sentence1": sentence1, "sentence2": sentence2, "corpus": corpus, "triple1": triple1,
                      "triple2": triple2, "dtype": dtype, "id": id}
                encoded_json = json.dumps(df)
                print(encoded_json, file=fw)
            except:
                pass


