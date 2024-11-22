import torch

def __steps(indices, weight, sequences, penalty=1.5, length_penalty=3):
    token_dict = dict()
    for i in range(weight.size(0)):
        for l in range(weight.size(1)):
            if indices[0][i] in sequences[l][-length_penalty:]:
                weight[i, l] /= penalty
            token_dict[float(weight[i, l])] = indices[0][i]

    probs, tokens = list(), list()
    for prob in torch.max(weight, -1)[0]:
        probs.append(prob)
        tokens.append(token_dict[float(prob)])
    
    return torch.stack(probs), torch.stack(tokens)

def mul_steps(values, weight, beam_width):
    return values.reshape((beam_width, 1)) * weight

def __search(logit, beam_width=3, penalty=1.5, length_penalty=3, early_stoping=True):
    def __est(indices, beam_width):
        appeering_count = 0
        for i in range(beam_width):
            if appeering_count / beam_width > 0.9:
                return True
            else:
                appeering_count = 1
            if int(indices[0, i]) in indices[0].tolist():
                appeering_count += 1
        else:
            return False
        
    w = torch.ones(size=(beam_width,)).float()
    sequences = torch.tensor([[] for _ in range(beam_width)]).int()
    for l in logit:
        l = l.unsqueeze(0)
        l = torch.nn.functional.log_softmax(l, dim=-1)
        l = torch.topk(l, k=beam_width)
        if early_stoping and __est(l[1], beam_width):
            return sequences, w
        sum_w = mul_steps(l[0], w, beam_width)
        w, tokens =  __steps(l[1], sum_w, sequences, penalty, length_penalty)
        sequences = torch.cat([sequences, tokens.reshape((beam_width, 1))], dim=-1)

    return sequences, w

def beam_search(model_probs, beam_width=3, penalty=1.5, length_penalty=3, temperature=0.2, early_stoping=True):
    model_probs /= temperature
    batch = []
    for logit in model_probs:

        seq, w = __search(logit, beam_width, penalty, length_penalty,  early_stoping)
        seq_probs = {}
        probs = []

        for i in range(seq.size(0)):
            seq_probs[float(w[i])] = seq[i]
            probs.append(float(w[i]))
        
        batch.append(seq_probs[max(probs)])
    return torch.stack(batch)