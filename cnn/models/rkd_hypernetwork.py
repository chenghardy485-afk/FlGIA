import torch
import torch.nn as nn
import math



class rkdLinear(nn.Module):
    def __init__(self, hidden_dim, d1, d2, rank=1):
        super(rkdLinear, self).__init__()
        self.lora_A = nn.Parameter(torch.empty(hidden_dim, d1))  
        self.lora_B = nn.Parameter(torch.empty(hidden_dim, d2))          
        self.bias_A = nn.Parameter(torch.empty(d1, 1))
        self.bias_B = nn.Parameter(torch.empty(1, d2))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5)) 
        nn.init.zeros_(self.bias_A)
        nn.init.zeros_(self.bias_B)

    def forward(self, x):
        out = row_wise_kronecker_product(self.lora_A, self.lora_B)
        bias = torch.mm(self.bias_A, self.bias_B).view(1, -1)
        return  x @ out + bias

def row_wise_kronecker_product(A, B):
    A_expanded = A.unsqueeze(2)  
    B_expanded = B.unsqueeze(1) 
    result = torch.bmm(A_expanded , B_expanded).reshape(A.shape[0], -1) 
    return result