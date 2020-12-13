import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder

import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_channel: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 pe: bool = False):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_input = d_input
        self._d_channel = d_channel
        self._d_model = d_model
        self._pe = pe

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout) for _ in range(N)])

        # self.layers_decoding = nn.ModuleList([Decoder(d_model,
        #                                               q,
        #                                               v,
        #                                               h,
        #                                               dropout=dropout) for _ in range(N)])

        self._embedding = nn.Linear(self._d_channel, d_model)

        self._linear = nn.Linear(d_model * d_input, d_output)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        encoding = self._embedding(x)

        # 位置编码
        if self._pe:
            pe = torch.ones_like(encoding[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000)/self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding = encoding + pe

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # if pe:
        #     decoding = encoding + pe
        # Decoding stack
        # for layer in self.layers_decoding:
        #     decoding = layer(decoding, encoding)

        # 三维变两维
        encoding = encoding.reshape(encoding.shape[0], -1)

        # d_model -> output
        output = self._linear(encoding)

        return output
