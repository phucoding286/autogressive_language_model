from autogressive_transformer import tM
from bpe import BPE
import torch
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

bpe_tokenizer = BPE()
bpe_tokenizer.train(
    batches_texts=[""],
    epochs=1000,
    path_saved_model="bpe.json"
)
probs_language_model = tM(
    vocab_size=len(bpe_tokenizer.idx_voc),
    d_model=256,
    dropout=0.1,
    nhead=8,
    ffn_dim=512,
    num_layers=6,
    device=D
)

probs_language_model.load_state_dict(torch.load("model.pth"))
output = probs_language_model.generate(
    torch.tensor(bpe_tokenizer.texts_to_sequences(["hello how are you?"], 20)).to(D)
)
print(bpe_tokenizer.sequences_to_texts(output))