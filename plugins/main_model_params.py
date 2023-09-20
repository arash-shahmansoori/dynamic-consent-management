import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from utils import parse_args


class AttentivePooledLSTMDvectorLiteratureLibri(nn.Module):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg_len = self.args.seg_len

        # Note: In pytorch the number of one-layer LSTM parameters can be computed as:
        # 4 * ((feature_dim + dim_cell) * dim_cell + dim_cell) + 4 * dim_cell LSTM parameters
        # In Keras, it is:
        # 4 * ((feature_dim + dim_cell) * dim_cell + dim_cell)  LSTM parameters

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell * 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell * 2, self.args.dim_emb)

        self.gn = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.linear = nn.Linear(self.args.dim_emb, 1)

    def forward(self, inputs):
        """Forward a batch through network."""
        inputs = inputs.view(-1, self.args.seg_len, self.args.feature_dim)

        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        embeds = self.gn(embeds)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class AttentivePooledLSTMDvectorLiteratureVox(nn.Module):
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

        self.ln = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])
        self.linear = nn.Linear(self.args.dim_emb, 1)

    def forward(self, inputs):
        """Forward a batch through network."""
        inputs = inputs.view(-1, self.args.seg_len, self.args.feature_dim)

        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        embeds = self.ln(embeds)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class AttentivePooledLSTMDvectorLibri(nn.Module):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg_len = self.args.seg_len

        # Note: In pytorch the number of one-layer LSTM parameters can be computed as:
        # 4 * ((feature_dim + dim_cell) * dim_cell + dim_cell) + 4 * dim_cell LSTM parameters
        # In Keras, it is:
        # 4 * ((feature_dim + dim_cell) * dim_cell + dim_cell)  LSTM parameters

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        self.gn = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.linear = nn.Linear(self.args.dim_emb, 1)

    def forward(self, inputs):
        """Forward a batch through network."""
        inputs = inputs.view(-1, self.args.seg_len, self.args.feature_dim)

        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)

        embeds = self.embedding(lstm_outs)  # (batch, seg_len, dim_emb)
        embeds = torch.tanh(embeds)  # (batch, seg_len, dim_emb)

        embeds = self.gn(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)

        embeds_ = embeds * attn_weights

        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class AttentivePooledLSTMDvectorVox(nn.Module):
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

        self.lstm = nn.LSTM(
            self.feat_dim_processed,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)
        self.ln = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])

        self.linear = nn.Linear(self.args.dim_emb, 1)

    def forward(self, inputs):
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)

        embeds = self.embedding(lstm_outs)  # (batch, seg_len, dim_emb)
        embeds = torch.tanh(embeds)  # (batch, seg_len, dim_emb)

        embeds = self.ln(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


class SpeakerClassifierLibri(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.out = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z):
        """Forward a batch through network."""

        z = z.view(-1, self.args.dim_emb)

        z = self.relu(self.embedding(z))
        feat = self.relu(self.hidden(z))

        feat_out = self.out(feat)
        out = self.softmax(feat_out)

        return out, feat_out


class SpeakerClassifierVox(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)

        self.ln = nn.LayerNorm([self.args.latent_dim])

        self.out = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        feat = self.ln(self.relu(self.embedding(z)))

        feat_out = self.out(feat)
        out = self.softmax(feat_out)

        return out, feat_out


class UnsupClsLatentLibri(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        z = self.relu(self.embedding(z))
        feat = self.relu(self.hidden(z))

        return feat


class UnsupClsLatentVox(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embedding = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.ln = nn.LayerNorm([self.args.latent_dim])
        self.hidden = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward a batch through network."""
        z = z / torch.norm(z, dim=1).view(z.size()[0], 1)

        z = self.relu(self.embedding(z))
        z = self.ln(z)
        feat = self.relu(self.hidden(z))

        # feat = self.featClassifier_training(z)
        return feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main_execute():
    args = parse_args()

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    dvec_model_lit_libri = AttentivePooledLSTMDvectorLiteratureLibri(args).to(device)
    dvec_model_lit_vox = AttentivePooledLSTMDvectorLiteratureVox(args).to(device)

    dvec_model_prop_libri = AttentivePooledLSTMDvectorLibri(args).to(device)
    cls_model_prop_libri = SpeakerClassifierLibri(args).to(device)
    cls_latent_prop_libri = UnsupClsLatentLibri(args).to(device)

    dvec_model_prop_vox = AttentivePooledLSTMDvectorVox(args).to(device)
    cls_model_prop_vox = SpeakerClassifierVox(args).to(device)
    cls_latent_prop_vox = UnsupClsLatentVox(args).to(device)

    x = torch.rand([1, args.seg_len, args.feature_dim]).to(device)

    _ = dvec_model_prop_libri(x)

    num_params_lit_libri = count_parameters(dvec_model_lit_libri)
    num_params_lit_vox = count_parameters(dvec_model_lit_vox)

    num_params_dvec_prop_libri = count_parameters(dvec_model_prop_libri)
    num_params_cls_prop_libri = count_parameters(cls_model_prop_libri)
    num_params_latent_prop_libri = count_parameters(cls_latent_prop_libri)

    num_params_dvec_prop_vox = count_parameters(dvec_model_prop_vox)
    num_params_cls_prop_vox = count_parameters(cls_model_prop_vox)
    num_params_latent_prop_vox = count_parameters(cls_latent_prop_vox)

    num_params_prop_libri = num_params_dvec_prop_libri * 8 + num_params_cls_prop_libri
    num_params_prop_vox = num_params_dvec_prop_vox * 8 + num_params_cls_prop_vox

    num_params_prop_libri_unsup = (
        num_params_dvec_prop_libri * 8 + num_params_latent_prop_libri
    )
    num_params_prop_vox_unsup = (
        num_params_dvec_prop_vox * 8 + num_params_latent_prop_vox
    )

    num_params_ratio_libri = (
        (num_params_prop_libri - num_params_lit_libri) / num_params_lit_libri
    ) * 100
    num_params_ratio_vox = (
        (num_params_prop_vox - num_params_lit_vox) / num_params_lit_vox
    ) * 100

    num_params_ratio_libri_unsup = (
        (num_params_prop_libri_unsup - num_params_lit_libri) / num_params_lit_libri
    ) * 100
    num_params_ratio_vox_unsup = (
        (num_params_prop_vox_unsup - num_params_lit_vox) / num_params_lit_vox
    ) * 100

    print(
        f"sup-ratio-libri:{num_params_ratio_libri:.3f}(%), sup-ratio-vox:{num_params_ratio_vox:.3f}(%)"
    )
    print(
        f"unsup-ratio-libri:{num_params_ratio_libri_unsup:.3f}(%), unsup-ratio-vox:{num_params_ratio_vox_unsup:.3f}(%)"
    )
