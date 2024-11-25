from models.ar_model import ModelG1, expand_embedding
from models.bpe import BPE
import torch
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# khởi tạo mô hình tokenizer
bpe_tokenizer = BPE()
bpe_tokenizer.train(
    batches_texts=[""],
    epochs=1000,
    path_saved_model="./models_file/bpe.json",
    learning_more=False
)

# khởi tạo mô hình
probs_language_model = ModelG1(
    vocab_size=len(bpe_tokenizer.idx_voc),
    d_model=768,
    dropout=0.1,
    nhead=12,
    dim_feedforward=1536,
    num_layers=12,
    device=D
)

# # mở rộng embedding nếu có thêm tập dataset
# expand_embedding(
#     new_vocab_size=len(bpe_tokenizer.idx_voc),
#     old_model_path="./models_file/context_model.pth",
#     new_model_path="./models_file/context_model.pth"
# )

# tải trọng số mô hình
probs_language_model.load_state_dict(
    torch.load(
        f="./models_file/context_model.pth",
        weights_only=True,
        map_location=D
    )
)
probs_language_model.eval()

print(" ------------------------")
print("| CHATBOT DEEP LEARNING  |")
print("| language: english      |")
print("| type: context generate |")
print(" ------------------------")

contexts = [] # lưu trữ ngữ cảnh
# chat với LLM
while True:

    # nhận đầu vào và khởi tạo cùng với ngữ cảnh
    user_input = input(" You: ")
    inp = "".join(contexts) + user_input
    
    # chuyển văn bản đầu vào sang tokenize
    inp = torch.tensor(
        bpe_tokenizer.texts_to_sequences(
            [inp],
            padding=False,
            end_token=True
        )
    )
    inp = inp.to(D)
    
    # mô hình suy luận
    all_sequences, only_pred_seq = probs_language_model.generate(
        sequences=inp,
        beam_width=1,
        penalty=1.7,
        length_penalty=5,
        early_stoping=False,
        tokens_penalty=[],
        max_tokens=50,
        temperature=1
    )
    
    # chuyển mô hình sang chuổi
    output = bpe_tokenizer.sequences_to_texts(only_pred_seq, skip_token=False, end_tok_stop=True)

    # lưu trữ ngữ cảnh
    contexts.append(user_input + "<eos>")
    contexts.append(output[0])
    
    # cắt bớt bộ nhớ ngữ cảnh
    if len(contexts) >= 4:
        contexts = contexts[2:]

    print(" Bot:", output[0])