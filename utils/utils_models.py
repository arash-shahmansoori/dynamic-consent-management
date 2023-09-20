import torch
import torch.nn as nn
import abc
import torch.nn.functional as F


# class RMSNorm(nn.Module):
#     def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
#         """
#             Root Mean Square Layer Normalization
#         :param d: model size
#         :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
#         :param eps:  epsilon value, default 1e-8
#         :param bias: whether use bias term for RMSNorm, disabled by
#             default because RMSNorm doesn't enforce re-centering invariance.
#         """
#         super(RMSNorm, self).__init__()

#         self.eps = eps
#         self.d = d
#         self.p = p
#         self.bias = bias

#         self.scale = nn.Parameter(torch.ones(d))
#         self.register_parameter("scale", self.scale)

#         if self.bias:
#             self.offset = nn.Parameter(torch.zeros(d))
#             self.register_parameter("offset", self.offset)

#     def forward(self, x):
#         # if self.p < 0.0 or self.p > 1.0:
#         norm_x = x.norm(2, dim=-1, keepdim=True)
#         d_x = self.d
#         # else:
#         #     partial_size = int(self.d * self.p)
#         #     partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

#         #     norm_x = partial_x.norm(2, dim=-1, keepdim=True)
#         #     d_x = partial_size

#         rms_x = norm_x * d_x ** (-1.0 / 2)
#         x_normed = x / (rms_x + self.eps)

#         # x_normed = x / norm_x

#         if self.bias:
#             return self.scale * x_normed + self.offset

#         return self.scale * x_normed


class DvectorInterface(nn.Module, metaclass=abc.ABCMeta):
    """d-vector interface."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            and hasattr(subclass, "seg_len")
            or NotImplemented
        )

    @abc.abstractmethod
    def forward(self, inputs):
        """Forward a batch through network.

        Args:
            inputs: (batch, seg_len, mel_dim)

        Returns:
            embeds: (batch, emb_dim)
        """
        raise NotImplementedError

    # @torch.jit.export
    def embed_utterance(self, utterance):
        """Embed an utterance by segmentation and averaging

        Args:
            utterance: (uttr_len, mel_dim) or (1, uttr_len, mel_dim)


        Returns:
            embed: (emb_dim)
        """
        assert utterance.ndim == 2 or (utterance.ndim == 3 and utterance.size(0) == 1)

        if utterance.ndim == 3:
            utterance = utterance.squeeze(0)

        if utterance.size(1) <= self.seg_len:
            embed = self.forward(utterance.unsqueeze(0)).squeeze(0)
        else:
            segments = utterance.unfold(0, self.seg_len, self.seg_len // 2)
            embeds = self.forward(segments)
            embed = embeds.mean(dim=0)
            embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True))

        return embed

    # @torch.jit.export
    def embed_utterances(self, utterances):
        """Embed utterances by averaging the embeddings of utterances

        Args:
            utterances: [(uttr_len, mel_dim), ...]

        Returns:
            embed: (emb_dim)
        """
        embeds = torch.stack([self.embed_utterance(uttr) for uttr in utterances])
        embed = embeds.mean(dim=0)
        return embed.div(embed.norm(p=2, dim=-1, keepdim=True))


class AttentivePooledLSTMDvector(DvectorInterface):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg_len = self.args.seg_len

        if args.delta and args.delta_delta:
            self.feat_dim_processed = args.feature_dim * 3
        elif args.delta:
            self.feat_dim_processed = args.feature_dim * 2
        else:
            self.feat_dim_processed = args.feature_dim

        # self.lstm = nn.LSTM(
        #     self.feat_dim_processed,
        #     self.args.dim_cell // 2,
        #     1,
        #     batch_first=True,
        # )

        self.lstm = nn.LSTM(
            self.feat_dim_processed,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        # self.lstm = nn.LSTM(
        #     self.feat_dim_processed,
        #     self.args.dim_cell,
        #     self.args.num_layers,
        #     batch_first=True,
        # )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        # self.drop_out = nn.Dropout()

        # self.embedding = nn.Linear(self.args.dim_cell, self.args.dim_emb)

        # self.gn = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.ln = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])

        # self.rms_norm = RMSNorm(self.args.seg_len * self.args.dim_emb)

        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.weight = nn.Parameter(
            torch.Tensor(self.args.n_speakers, self.args.dim_emb)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        # lstm_outs = self.drop_out(lstm_outs)
        embeds = self.embedding(lstm_outs)  # (batch, seg_len, dim_emb)
        embeds = torch.tanh(embeds)  # (batch, seg_len, dim_emb)

        # embeds = self.rms_norm(embeds.view(-1, self.args.seg_len * self.args.dim_emb))
        embeds = self.ln(embeds)

        # embeds = embeds.view(-1, self.args.seg_len, self.args.dim_emb)

        # embeds = self.drop_out(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class AttentivePooledLSTMDvectorLiterature(nn.Module):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg_len = self.args.seg_len

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell * 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell * 2, self.args.dim_emb)

        self.gn = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        # self.ln = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])
        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.weight = nn.Parameter(
            torch.Tensor(self.args.n_speakers, self.args.dim_emb)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        """Forward a batch through network."""
        inputs = inputs.view(-1, self.args.seg_len, self.args.feature_dim)

        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        embeds = self.gn(embeds)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class UnsupClsLatent(nn.Module):
    def __init__(self, args):
        super(UnsupClsLatent, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.ln = nn.LayerNorm([self.args.latent_dim])
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.relu = nn.ReLU()

        # self.featClassifier_training = nn.Sequential(
        #     self.embedding,
        #     self.relu,
        #     self.hidden,
        #     self.relu,
        # )

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        z = self.relu(self.embedding(z))
        z = self.ln(z)
        feat = self.relu(self.hidden(z))

        # feat = self.featClassifier_training(z)
        return feat


class UnsupClsLatentDR(nn.Module):
    def __init__(self, args):
        super(UnsupClsLatentDR, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.relu = nn.ReLU()

        # self.featClassifier_training = nn.Sequential(
        #     self.embedding,
        #     self.relu,
        #     self.hidden,
        #     self.relu,
        # )

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        z = self.relu(self.embedding(z))
        feat = self.relu(self.hidden(z))

        # feat = self.featClassifier_training(z)
        return feat


class SpeakerClassifierRec(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierRec, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.out = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.gn = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.Classifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            self.hidden,
            self.relu,
            self.gn,
            self.out,
            self.softmax,
        )
        self.featClassifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            self.hidden,
            self.relu,
            self.gn,
            self.out,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z, c=None, mode=None):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)
        if c is not None:
            c = self.idx2onehot(c, z, mode)
            z = torch.cat((z, c), dim=-1)

        out = self.Classifier_training(z)
        feat = self.featClassifier_training(z)
        return out, feat

    def idx2onehot(self, c, z, mode):
        if mode == "training":
            c_concat = [[c[i]] * int(z.shape[0] / len(c)) for i in range(len(c))]
        elif mode == "validation":
            c_concat = [[c[i]] * self.args.nv_utterances_labeled for i in range(len(c))]
        elif mode == "testing":
            c_concat = [[c[i]] * self.args.nt_utterances_labeled for i in range(len(c))]
        else:
            raise ValueError

        c_flatten = [i for s in c_concat for i in s]
        y_onehot = F.one_hot(
            torch.tensor(c_flatten), num_classes=self.args.spk_per_bucket
        )

        return y_onehot.to(c.device)


class SpeakerClassifierRec_v2(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierRec_v2, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)

        # self.drop_out = nn.Dropout()

        # self.gn = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)
        self.ln = nn.LayerNorm([self.args.latent_dim])
        # self.rms_norm = RMSNorm(self.args.latent_dim)

        # self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.out = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        # self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim // 2)
        # self.out = nn.Linear(self.args.latent_dim // 2, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.featClassifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            # self.drop_out,
            self.ln,
            # self.hidden,
            # self.relu,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        feat = self.featClassifier_training(z)

        feat_out = self.out(feat)
        out = self.softmax(feat_out)

        return out, feat_out


class SpeakerClassifierRec_DR(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierRec_DR, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.out = nn.Linear(
            self.args.latent_dim, self.args.n_speakers + self.args.n_speakers_other
        )

        self.gn = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.Classifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            self.hidden,
            self.relu,
            self.gn,
            self.out,
            self.softmax,
        )
        self.featClassifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            self.hidden,
            self.relu,
            self.gn,
            self.out,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z, c=None, mode=None):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)
        if c is not None:
            c = self.idx2onehot(c, z, mode)
            z = torch.cat((z, c), dim=-1)

        out = self.Classifier_training(z)
        feat = self.featClassifier_training(z)
        return out, feat

    def idx2onehot(self, c, z, mode):
        if mode == "training":
            c_concat = [[c[i]] * int(z.shape[0] / len(c)) for i in range(len(c))]
        elif mode == "validation":
            c_concat = [[c[i]] * self.args.nv_utterances_labeled for i in range(len(c))]
        elif mode == "testing":
            c_concat = [[c[i]] * self.args.nt_utterances_labeled for i in range(len(c))]
        else:
            raise ValueError

        c_flatten = [i for s in c_concat for i in s]
        y_onehot = F.one_hot(
            torch.tensor(c_flatten), num_classes=self.args.spk_per_bucket
        )

        return y_onehot.to(c.device)


class SpeakerClassifierRec_DR_v2(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierRec_DR_v2, self).__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.out = nn.Linear(
            self.args.latent_dim, self.args.n_speakers + self.args.n_speakers_other
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.featClassifier_training = nn.Sequential(
            self.embedding,
            self.relu,
            self.hidden,
            self.relu,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        feat = self.featClassifier_training(z)

        feat_out = self.out(feat)
        out = self.softmax(feat_out)

        return out, feat_out


class SpeakerClassifierE2E(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierE2E, self).__init__()
        self.args = args

        self.att_lstm = AttentivePooledLSTMDvector(args)
        self.spk_cls = SpeakerClassifierRec_v2(args)

    def forward(self, x):
        feat = self.att_lstm(x)
        out, feat_out = self.spk_cls(feat)

        return out, feat_out, feat


class SpeakerClassifierE2ESupervisedV2(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierE2ESupervisedV2, self).__init__()
        self.args = args

        self.att_lstm = AttentivePooledLSTMDvectorLiterature(args)

    def forward(self, x):
        feat = self.att_lstm(x)

        return feat


class SpeakerClassifierE2EUnsupervised(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierE2EUnsupervised, self).__init__()
        self.args = args

        self.att_lstm = AttentivePooledLSTMDvector(args)
        self.spk_dvec_latent = UnsupClsLatent(args)

    def forward(self, x):
        feat = self.att_lstm(x)
        feat_out = self.spk_dvec_latent(feat)

        return feat_out


class SpeakerClassifierE2EUnsupervisedV2(nn.Module):
    def __init__(self, args):
        super(SpeakerClassifierE2EUnsupervisedV2, self).__init__()
        self.args = args

        self.att_lstm = AttentivePooledLSTMDvectorLiterature(args)

    def forward(self, x):
        feat = self.att_lstm(x)

        return feat


def moving_average(model_1, model_2, alpha=1):
    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        param_1.data *= 1.0 - alpha
        param_1.data += param_2.data * alpha
