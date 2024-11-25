import torch
from torch import nn
from models.beam_search import beam_search, search
from torch.utils.data import DataLoader, TensorDataset
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# hàm train dành riêng cho mô hình sinh chuổi AR (Autogressive)
def trainer(epochs=20, inp=None, model: object = None, lr: int = 0.0001, batch_size=32, model_path="model.pth"):
    # đảm bảo input là long
    inp = inp.long()
    dataset = TensorDataset(inp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        for batch_idx, (batch_inp,) in enumerate(dataloader):
            """
            inputs và labels ở đây chính là cùng một văn bản, với mục đích train cho
            mô hình dự đoán từ tiếp theo từ câu đầu vào, việc che đi phần cuối cho đầu vào
            và che đi token đầu cho labels giúp mô hình học được cách suy diễn từ tiếp theo.
            nếu không có phần này, mô hình sẽ không thể học được cách suy diễn từ tiếp theo
            mà mô hình chỉ học thuộc lòng đầu vào và đầu ra sẽ không khác gì so với đầu vào
            """
            inputs = batch_inp[:, :-1]
            labels = batch_inp[:, 1:]

            pred_seq_prob = model(inputs)
            pred_seq_prob = pred_seq_prob.view(-1, pred_seq_prob.size(-1))
            labels = labels.reshape(-1)

            # Tính loss
            loss = criterion(pred_seq_prob, labels)

            # Cập nhật trọng số của mô hình
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Step {batch_idx+1}/{len(dataloader)}, Loss on step -> {loss.item()}")

        # Tính loss trung bình cho mỗi epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
    
    torch.save(model.state_dict(), model_path)
    print("Đã huấn luyện thành công mô hình xác suất tự chú ý")
    return 0


# mở rộng trọng số embedding nếu có thêm token mới
def expand_embedding(new_vocab_size:int, old_model_path="model.pth", new_model_path="m.pth"):
    checkpoint = torch.load(old_model_path, map_location=D)
    old_vocab_size = checkpoint['position_embedding.embedding.weight'].size(0)
    distance_vocab_size = (new_vocab_size - old_vocab_size)

    probs_language_model = ModelG1(
        vocab_size=old_vocab_size,
        d_model=768,
        dropout=0.1,
        nhead=12,
        dim_feedforward=1536,
        num_layers=12,
        device=D
    )

    probs_language_model.load_state_dict(
        torch.load(
            old_model_path,
            weights_only=True,
            map_location=D
        )
    )

    # mở rộng weight của embedding
    old_embed_weight = probs_language_model.position_embedding.embedding.weight
    new_embed_weight = torch.nn.Embedding(
        new_vocab_size,
        old_embed_weight.size(1)
    )
    new_embed_weight = new_embed_weight.weight
    probs_language_model.position_embedding.embedding.weight = torch.nn.Parameter(
        torch.concatenate(
            [old_embed_weight, new_embed_weight[-distance_vocab_size:, :]],
            dim=0
        )
    )

    # mở rộng vocab size của weight và bias của lớp đầu ra
    old_out_weight = probs_language_model.linear_out.weight
    old_out_bias = probs_language_model.linear_out.bias
    new_w = torch.nn.Linear(
        probs_language_model.linear_out.in_features,
        probs_language_model.linear_out.out_features + distance_vocab_size
    )
    new_out_weight, new_out_bias = new_w.weight, new_w.bias

    # update bias và weight mới
    probs_language_model.linear_out.bias = torch.nn.Parameter(
        torch.concatenate(
            [old_out_bias, new_out_bias[-distance_vocab_size:]],
            dim=0
        )
    )
    probs_language_model.linear_out.weight = torch.nn.Parameter(
        torch.concatenate(
            [old_out_weight, new_out_weight[-distance_vocab_size:, :]],
            dim=0
        )
    )
    
    # lưu lại mô hình đã mở rộng
    torch.save(probs_language_model.state_dict(), new_model_path)
    print("size mới trọng số của embedding:", probs_language_model.position_embedding.embedding.weight.size())
    print("size mới trọng số của linear out:", probs_language_model.linear_out.weight.size())
    print("size mới bias của linear out:", probs_language_model.linear_out.bias.size())

# embedding mã hóa vị trí  
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 dropout=0.1, padding_idx=0, device: bool = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.positional_encoding_dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            device=device
        )

    def pos(self, max_sequence_length):
        even_i = torch.arange(
            start=0,
            end=self.embedding_dim,
            step=2,
            device=self.device
        ).float()
        denominator = torch.pow(
            10000,
            even_i/self.embedding_dim
        )
        position = torch.arange(
            max_sequence_length,
            device=self.device
        ).reshape(max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack(
            [even_PE, odd_PE],
            dim=2
        )
        PE = torch.flatten(
            stacked,
            start_dim=1,
            end_dim=2
        )
        return PE.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        out = self.positional_encoding_dropout(x + self.pos(max_sequence_length=x.shape[1]))
        return out.to(self.device)
    
# embedding mã hóa vị trí  
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 dropout=0.1, padding_idx=0, device: bool = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.positional_encoding_dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            device=device
        )

    def pos(self, max_sequence_length):
        even_i = torch.arange(
            start=0,
            end=self.embedding_dim,
            step=2,
            device=self.device
        ).float()
        denominator = torch.pow(
            10000,
            even_i/self.embedding_dim
        )
        position = torch.arange(
            max_sequence_length,
            device=self.device
        ).reshape(max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack(
            [even_PE, odd_PE],
            dim=2
        )
        PE = torch.flatten(
            stacked,
            start_dim=1,
            end_dim=2
        )
        return PE.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        out = self.positional_encoding_dropout(x + self.pos(max_sequence_length=x.shape[1]))
        return out.to(self.device)
    

# khởi tạo mô hình
class ModelG1(nn.Module):
    "Mô hình tương tự GPT hoặc các mô hình tự hồi quy khác"
    """
    Khi training model này xin lưu ý, nên training nó theo kiểu tự hồi quy
    khác với các kiểu training truyền thống.

    Có nghĩa là mô hình sẽ dự đoán đầu ra dựa vào đầu vào, sau đó giảm loss
    dựa trên đầu vào và đầu ra, cách training này khiến mô hình học cách sinh
    văn bản tiếp theo.

    Vì vậy mô hình không cần training theo kiểu X hay Y truyền thống, với mô hình tự
    hồi quy, nó không thể training theo kiểu chuổi nguồn và chuổi đích.

    Nếu bạn muốn một mô hình dành cho dịch máy hãy quan tâm transformer.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1,
                activation=nn.functional.relu, batch_first=True, norm_first=True, device=None,
                vocab_size=None, pad_token_id=None):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.position_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            dropout=dropout,
            padding_idx=pad_token_id,
            device=device
        )
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device
            ),
            num_layers=num_layers
        )
        self.linear_out = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device
        )

    def causal_mask(self, sequence_length, device):
        causal = nn.Transformer.generate_square_subsequent_mask(sequence_length)
        causal_bool = (causal == -torch.inf)
        return causal_bool.to(device)

    def create_key_padding_mask(self, inputs, device=None):
        key_padding_mask = (inputs == 0)
        return key_padding_mask.to(device)

    def forward(self, x):
        mask = self.causal_mask(
            x.shape[1],
            device=self.device
        )
        src_key_padding_mask = self.create_key_padding_mask(
            x,
            device=self.device
        )
        x = self.position_embedding(x)
        out = self.model(
            src=x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True
        )
        out = self.linear_out(out)
        return out
    
    def generate(self, sequences, max_tokens=10, temperature=0.2, beam_width=1, penalty=2,
                length_penalty=3, early_stoping=False, tokens_penalty: list=[]):
        
        only_predict = torch.tensor([[] for _ in range(beam_width)]).to(self.device).long()
        sequences = torch.tensor([sequences[0].tolist() for _ in range(beam_width)]).to(self.device)
        weight = None

        for _ in range(max_tokens):
            # output model
            logit = (self.forward(sequences) / temperature)[0]

            # áp dụng beam_search lên đầu ra
            prediction, weight = search(
                logit = logit,
                beam_width=beam_width,
                penalty=penalty,
                length_penalty=length_penalty,
                early_stoping=early_stoping,
                tokens_penalty=tokens_penalty,
                device=self.device
            )
            # lấy token cuối của logit (vì nó là token tiếp theo trong nguyên lý AR)
            prediction = prediction[:, -1].reshape(beam_width, 1).to(self.device)

            """
            biến sequences có công dụng lưu tất cả chuổi bao gồm cả chuổi đầu vào (phù hợp cho sinh
            từ tiếp theo)
            biến only_predict chỉ lưu các token tiếp theo không bao gồm token cuối (phù hợp cho huấn
            luyện LLM chatbot)
            """
            sequences = torch.concatenate(
                [ sequences, prediction ],
                dim=-1
            )
            only_predict = torch.concatenate(
                [ only_predict, prediction ],
                dim=-1
            )
        
        # thuật toán cuối cùng sau beam search, lấy ra chuổi tốt nhất
        only_nextok_probs = {}
        full_sequences_probs = {}
        list_probs = []

        for i in range(beam_width):
            full_sequences_probs[float(weight[i])] = sequences[i]
            only_nextok_probs[float(weight[i])] = only_predict[i]
            list_probs.append(float(weight[i]))
        
        # lấy ra chuổi tốt nhất
        max_prob = max(list_probs)
        only_next_token_predict_sequence = only_nextok_probs[max_prob].unsqueeze(0).to(self.device)
        full_predict_sequences = full_sequences_probs[max_prob].unsqueeze(0).to(self.device)
        
        # thêm token <end> vào nếu token cuối không có (mục đích duy trì sự ổn định của ngữ cảnh)
        if int(full_predict_sequences[0][-1]) != 1:
            full_predict_sequences = torch.concatenate(
                [ full_predict_sequences, torch.tensor([[1]]).to(self.device) ],
                dim=-1
            )
            only_next_token_predict_sequence = torch.concatenate(
                [ only_next_token_predict_sequence, torch.tensor([[1]]).to(self.device) ],
                dim=-1
            )

        return full_predict_sequences, only_next_token_predict_sequence