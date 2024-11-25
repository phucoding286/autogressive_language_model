from models.ar_model import ModelG1
from models.bpe import BPE
import torch
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

bpe_tokenizer = BPE()
bpe_tokenizer.train(
    batches_texts=[""],
    epochs=1000,
    path_saved_model="bpe.json"
)

probs_language_model = ModelG1(
    vocab_size=len(bpe_tokenizer.idx_voc),
    d_model=768,
    dropout=0.1,
    nhead=12,
    dim_feedforward=1536,
    num_layers=12,
    device=D
)

probs_language_model.load_state_dict(torch.load("generate_model.pth", weights_only=True, map_location=D))

while True:
    inp = input("Generate Start: ")
    inp = torch.tensor(bpe_tokenizer.texts_to_sequences([inp], padding=False)).to(D)
    
    all_of_seq, only_pred_seq = probs_language_model.generate(
        sequences=inp,
        beam_width=1,
        penalty=2,
        length_penalty=3,
        early_stoping=False,
        tokens_penalty=[],
        max_tokens=50,
        temperature=0.2
    )

    print(bpe_tokenizer.sequences_to_texts(all_of_seq, skip_token=False))