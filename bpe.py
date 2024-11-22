import json
import os
import numpy as np

class BPE:
    def __init__(self, pad_tok="<pad>", eos_tok="<eos>", unk_tok="<unk>"):
        self.pad_tok, self.eos_tok, self.unk_tok = pad_tok, eos_tok, unk_tok
        self.idx_voc, self.voc_idx = dict(), dict()

    def vocabuliary_init(self, batches_texts: list = None, epochs:int=100):
        vocabuliaries = list(set("".join(batches_texts)))
        i_vb = 0

        while i_vb < len(vocabuliaries) and i_vb < epochs:
            
            all_of_pairs = []
            texts = " ".join(batches_texts)
            for adj_w in texts.split(vocabuliaries[i_vb])[1:]:
                if len(adj_w) > 1:
                    all_of_pairs.append(vocabuliaries[i_vb]+adj_w[0])
            
            counter_appeering_times = {}
            for vc in all_of_pairs:
                if vc in counter_appeering_times:
                    counter_appeering_times[vc] += 1
                else:
                    counter_appeering_times[vc] = 0
            
            max_val = 0
            for value in counter_appeering_times.values():
                if value > max_val:
                    max_val = value

            most_popular_pair = []
            for key, value in counter_appeering_times.items():
                if value == max_val and max_val != 0 and key not in vocabuliaries:
                    most_popular_pair.append(key)
            
            vocabuliaries += most_popular_pair

            i_vb += 1
        
        return vocabuliaries
    
    def train(self, batches_texts:list=None, epochs:int=100, path_saved_model="./tok_model.json"):
        if os.path.exists(path_saved_model):
            with open(path_saved_model, mode="r", encoding="utf-8") as file:
                data = json.load(fp=file)
            self.idx_voc, self.voc_idx = data['idx_voc'], data['voc_idx']
            return "Loaded previous saved model"
        
        vocabularies = [self.pad_tok, self.eos_tok, self.unk_tok] + self.vocabuliary_init(batches_texts, epochs)
        for i in range(len(vocabularies)):
            self.idx_voc[i] = vocabularies[i]
            self.voc_idx[vocabularies[i]] = i

        with open(path_saved_model, mode="w", encoding="utf-8") as file:
            json.dump(
                obj={"idx_voc": self.idx_voc, "voc_idx": self.voc_idx},
                fp=file,
                indent=4,
                ensure_ascii=False
            )
        return "training complete!"
    
    def texts_to_sequences(self, texts: list, padding_dim=20):
        batches = []
        for text in texts:

            logit = []
            length = len(text)

            while len(text) >= 0 and length >= 0:

                if text[:length] in self.voc_idx:
                    logit.append(self.voc_idx[text[:length]])
                    text = text[length:]
                    length = len(text)

                elif length == 1 and text[:length] not in self.voc_idx:
                    logit.append(self.voc_idx[self.unk_tok])
                    text = text[length + 1 : ]
                    length = len(text)

                else:
                    length -= 1
            
            for _ in range(len(logit), padding_dim):
                logit.append(self.voc_idx[self.pad_tok])
            
            if len(logit) > padding_dim:
                logit = logit[:padding_dim]

            batches.append(logit)
        return np.array(batches)
    
    def sequences_to_texts(self, sequences):
        texts = []
        for batch in sequences:
            text = ""
            for logit in batch:
                try:
                    text += self.idx_voc[int(logit)]
                except:
                    text += self.idx_voc[str(int(logit))]
            texts.append(text)
        return texts

data = open("data.txt", "r", encoding="utf8").read().splitlines()[:5000]
tok = BPE()
out = tok.train(
    batches_texts=data,
    epochs=1000
)
seq = tok.texts_to_sequences(["Meaning of hello in English hello exclamation"], padding_dim=32)
txt = tok.sequences_to_texts(seq)

print(seq)
print(txt)