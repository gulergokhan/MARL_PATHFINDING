"""Ag mimarileri. Asama 3'te sadece MLP; QMIX hypernetwork'u Asama 7'de eklenir."""
import torch
import torch.nn as nn

from config import HIDDEN


class MLP(nn.Module):
    """Q agi: obs -> her aksiyon icin Q degeri.

    5x5 gridde ~35k parametre; CPU'da rahat egitilir.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


NEG_INF = -1e9      # -inf yerine: maskeli softmax/max'ta NaN uretmez


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gecersiz aksiyonlari cok buyuk negatif degere it.

    Gercek -inf kullanmiyoruz: terminal gecislerde tum aksiyonlar maskeliyse
    -inf * 0 = NaN cikar ve gradyan sessizce bozulur.
    """
    return q.masked_fill(mask <= 0, NEG_INF)
