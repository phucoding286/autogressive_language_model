import torch
from torch import nn
from beam_search import beam_search
D = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# train mô hình
def trainer(epochs=20, inp=None, model: object = None, lr: int = 0.0001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        pred_seq_prob = model.generate(inp, train=True)
        pred_seq_prob = pred_seq_prob.view(-1, pred_seq_prob.size(-1))
        loss = criterion(pred_seq_prob, inp.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            
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
            )
            return sequences

        return out
