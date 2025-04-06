from torch import nn


class AttentionPoolingModel(nn.Module):
    def __init__(self, nli_model_dim):
        super(AttentionPoolingModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(nli_model_dim, 1),
        )
        self.attention = nn.Linear(nli_model_dim, 1)

    def forward(self, x):
        attention_weights = nn.functional.softmax(self.attention(x), dim=1)
        x = (x * attention_weights).sum(1)
        x = self.classifier(x)
        return x
