from models.bpe import BPE
from models.ar_model import ModelG1, trainer
import torch
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_contexts_data(data_path="data.txt"):
    with open(data_path, "r", encoding="utf8") as file:
        data = file.read().splitlines()
    i = 0
    c = 4
    contexts = []
    while c < len(data):
        contexts.append("<eos>".join(data[i:c]))
        i += 4
        c += 4
    return contexts

with open("data.txt", "r", encoding="utf8") as file:
    data = file.read().splitlines()[:25000] # train thuật toán BPE với 25000 câu

bpe_tokenizer = BPE()
bpe_tokenizer.train(
    batches_texts=data,
    epochs=1000,
    path_saved_model="bpe.json"
)

# khởi tạo mô hình transformer AR
probs_language_model = ModelG1(
    vocab_size=len(bpe_tokenizer.idx_voc),
    d_model=768,
    dropout=0.1,
    nhead=12,
    dim_feedforward=1536,
    num_layers=12,
    device=D
)
probs_language_model.load_state_dict(
    torch.load(
        "m.pth",
        weights_only=True,
        map_location=D
    )
)

data = get_contexts_data()[:500]
data_train_test = bpe_tokenizer.texts_to_sequences(
    texts=data,
    padding_dim=768,
    padding=True,
    end_token=True
)
data_train_test = torch.tensor(data_train_test).to(D)

trainer(
    epochs=10,
    inp=data_train_test,
    model=probs_language_model,
    lr=0.0001,
    batch_size=32
)

test_inference_input = bpe_tokenizer.texts_to_sequences(['I have a dog'], padding_dim=50, end_token=True)
test_inference_input = torch.tensor(test_inference_input).to(D)

all_of_seq, only_pred_seq = probs_language_model.generate(
    sequences=test_inference_input,
    beam_width=1,
    penalty=2,
    length_penalty=3,
    early_stoping=False,
    tokens_penalty=[],
    max_tokens=50,
    temperature=0.2
)


text_output = bpe_tokenizer.sequences_to_texts(all_of_seq)
print(text_output)