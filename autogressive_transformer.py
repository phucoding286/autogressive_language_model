import torch
from torch import nn
from beam_search import beam_search
from torch.utils.data import DataLoader, TensorDataset
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# train mô hình
def trainer(epochs=20, inp=None, model: object = None, lr: int = 0.0001, batch_size=32, model_path="model.pth"):
    # Chuyển dữ liệu đầu vào thành TensorDataset và DataLoader
    inp = inp.long()
    dataset = TensorDataset(inp)  # Xây dựng dataset từ inp
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Chia dữ liệu thành các batch

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()  # Đặt mô hình ở chế độ huấn luyện

        epoch_loss = 0.0  # Tổng loss trong mỗi epoch
        for batch_idx, (batch_inp,) in enumerate(dataloader):  # Duyệt qua các batch
            # Dự đoán xác suất chuỗi
            pred_seq_prob = model.generate(batch_inp, train=True)
            pred_seq_prob = pred_seq_prob.view(-1, pred_seq_prob.size(-1))  # Thay đổi hình dạng để phù hợp với CrossEntropyLoss

            # Tính loss
            loss = criterion(pred_seq_prob, batch_inp.view(-1))  # Lưu ý: batch_inp cần phải reshape thành 1D

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
    return "Training complete!"


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
    
# mô hình tự hồi quy cơ bản
class tM(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model=512, dropout=0.1, nhead=8, ffn_dim=1024, num_layers=6,
                device=None):
        super().__init__()
        self.e = self.position_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            dropout=dropout,
            device=device
        )
        self.m = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                device=device,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.o = torch.nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device
        )
        self.device = device

    def causal_mask(self, sequence_length, device):
        causal = torch.nn.Transformer.generate_square_subsequent_mask(sequence_length)
        causal_bool = (causal == -torch.inf)
        return causal_bool.to(device)
    
    def create_key_padding_mask(self, inputs, device=None):
        key_padding_mask = (inputs == 0)
        return key_padding_mask.to(device)
    
    # cách dự đoán theo kiểu autogressive
    def generate(self, x, train=False, max_tokens=2, beam_width=3, penalty=1, length_penalty=3,
                temperature:float=0.2):
        x = self.e(x)
        max_tok_inp = x.size(1)

        if train:
            max_tokens = x.size(1)

        sequences = []
        for i in range(max_tokens):
            mask = self.causal_mask(
                x.size(1),
                device=self.device
            )

            prediction = self.m(
                x,
                mask=mask,
                is_causal=True
            )
            prediction = prediction[:, -1, :].unsqueeze(1)

            x = torch.concatenate([x, prediction], dim=1)

            # giảm bớt bộ nhớ ngữ cảnh phía sau nếu nó tràn
            if x.size(1) > max_tok_inp:
                x = x[:, max_tok_inp:, :]

            sequences.append(prediction)

        out = torch.cat(sequences, dim=1)
        out = self.o(out)
        
        if not train:
            sequences = out = beam_search(
                model_probs=out,
                beam_width=beam_width,
                penalty=penalty,
                length_penalty=length_penalty,
                temperature=temperature,
                early_stoping=True,
                device=self.device
            )
            return sequences

        return out
