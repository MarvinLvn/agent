from torch import nn


class SSLAgentNN(nn.Module):
    def __init__(self, inverse_model, feature_extractor,
                 synthesizer, vocoder, discriminator_model=None):
        super(SSLAgentNN, self).__init__()
        self.inverse_model = inverse_model
        self.discriminator_model = discriminator_model
        self.feature_extractor = feature_extractor
        self.synthesizer = synthesizer
        self.vocoder = vocoder

    def forward(self, feat_seqs):
        art_seqs_pred = self.inverse_model(feat_seqs)
        return art_seqs_pred
