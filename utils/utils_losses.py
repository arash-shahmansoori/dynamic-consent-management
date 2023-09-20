import copy
import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F


def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    kd_loss = (-1 * targets_norm * log_scores_norm).sum(dim=1).mean() * T**2
    return kd_loss


class KdManager:
    """Knowledge distillation class to compute the distillation loss."""

    def __init__(self):
        self.teacher_model = None

    def update_teacher(self, model):
        self.teacher_model = copy.deepcopy(model)

    def get_kd_loss(self, cur_model_logits, x):
        if self.teacher_model is not None:
            with torch.no_grad():
                _, prev_model_logits = self.teacher_model.forward(x)
            dist_loss = loss_fn_kd(cur_model_logits, prev_model_logits)
        else:
            dist_loss = 0
        return dist_loss


class GE2ELoss(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        GE2E loss for contrastive learning (first implementation).
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super(GE2ELoss, self).__init__()
        self.args = args
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )

                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.einsum(
                            "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    @staticmethod
    def calc_acc(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = y.view(spk_per_bucket, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            total_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(total_target)

            acc = 100 * (correct.sum() / total_target.shape[0])

            return acc

    @staticmethod
    def calc_acc_train(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        pred_list, target_list = [], []
        # for i in range(cos_sim_matrix.shape[1]):
        for i in range(num_utt_per_spk):
            pred = cos_sim_matrix[:, i, :].argmax(dim=0)
            target = y.view(spk_per_bucket, -1)[:, i]

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

        total_pred = torch.tensor(pred_list).view(-1)
        total_target = torch.tensor(target_list).view(-1)

        correct = total_pred.eq(total_target)

        acc = 100 * (correct.sum() / total_target.shape[0])

        return acc

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """

        cos_sim_matrix = self.compute_similarity_matrix(dvecs)
        L = self.embed_loss(dvecs, cos_sim_matrix)

        return L.sum()


class GE2ELossSup(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0):
        """
        GE2E loss for contrastive learning (first implementation).
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super(GE2ELossSup, self).__init__()

        self.args = args

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )

                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.einsum(
                            "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    def embed_loss_sup(self, dvecs, cos_sim_matrix, label):
        N, M, _ = dvecs.shape
        label = torch.from_numpy(np.asarray(range(0, N))).cuda()
        label = torch.repeat_interleave(label, repeats=M, dim=0).cuda()

        label = label.clone().detach().to(dtype=torch.long)

        loss = self.criterion(cos_sim_matrix.view(-1, N), label)

        return loss

    @staticmethod
    def calc_acc(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = y.view(spk_per_bucket, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            total_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(total_target)

            acc = 100 * (correct.sum() / total_target.shape[0])

            return acc

    @staticmethod
    def calc_acc_train(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        pred_list, target_list = [], []
        # for i in range(cos_sim_matrix.shape[1]):
        for i in range(num_utt_per_spk):
            pred = cos_sim_matrix[:, i, :].argmax(dim=0)
            target = y.view(spk_per_bucket, -1)[:, i]

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

        total_pred = torch.tensor(pred_list).view(-1)
        total_target = torch.tensor(target_list).view(-1)

        correct = total_pred.eq(total_target)

        acc = 100 * (correct.sum() / total_target.shape[0])

        return acc

    def forward(self, dvecs, label):
        """
        Calculates the GE2E loss for an input of dimensions:
        (num_speakers, num_utts_per_speaker, dvec_feats)
        """

        cos_sim_matrix = self.compute_similarity_matrix(dvecs)
        loss = self.embed_loss_sup(dvecs, cos_sim_matrix, label)

        return loss


class AngleProtoLoss(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0):
        super(AngleProtoLoss, self).__init__()

        self.args = args

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

        self.criterion = nn.CrossEntropyLoss().cuda()

    def calc_new_centroids(self, dvecs, centroids, spkr):
        """
        Calculates the new centroids excluding the reference utterance
        """
        _, M, _ = dvecs.shape

        excl = dvecs[spkr, :M]
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        _, M, _ = dvecs.shape

        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx)

            cs_row = []
            for utt_indx, utterance in enumerate(speaker):
                # vector based cosine similarity for speed
                if utt_indx == M - 1:
                    cs_row.append(
                        torch.clamp(
                            torch.einsum(
                                "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                            )
                            / (
                                torch.norm(utterance) * torch.norm(new_centroids, dim=1)
                            ),
                            1e-6,
                        )
                    )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    @staticmethod
    def calc_acc(cos_sim_matrix, y, spk_per_bucket):
        num_utt_per_spk = cos_sim_matrix.shape[1]

        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = y.view(spk_per_bucket, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            total_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(total_target)

            acc = 100 * (correct.sum() / total_target.shape[0])

            return acc

    def embed_loss_softmax_sup(self, cos_sim_matrix, label):
        N, M, _ = cos_sim_matrix.shape

        nloss = self.criterion(
            cos_sim_matrix.view(-1, N),
            torch.repeat_interleave(label, repeats=M, dim=0).cuda(),
        )

        return nloss

    def forward(self, x):
        out_anchor = torch.mean(x[:, 1:, :], 1)
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = self.compute_similarity_matrix(x)

        label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        label = label.clone().detach().to(dtype=torch.long)

        loss = self.embed_loss_softmax_sup(cos_sim_matrix, label)

        return loss


class StableSupContLoss(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0):
        super(StableSupContLoss, self).__init__()

        self.args = args

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

        self.criterion = nn.CrossEntropyLoss()

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))

        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )

                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.einsum(
                            "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        cos_sim_matrix_reshape = cos_sim_matrix.reshape(
            (cos_sim_matrix.shape[0] * cos_sim_matrix.shape[1], cos_sim_matrix.shape[2])
        )

        torch.clamp(self.w, 1e-6)
        z = cos_sim_matrix_reshape * self.w + self.b

        return z.reshape(
            (cos_sim_matrix.shape[0], cos_sim_matrix.shape[1], cos_sim_matrix.shape[2])
        )

    @staticmethod
    def calc_acc(cos_sim_matrix):
        num_utt_per_spk = cos_sim_matrix.shape[1]

        N, M, _ = cos_sim_matrix.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        _target = torch.repeat_interleave(label, repeats=M, dim=0).view(-1)

        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = _target.view(N, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            true_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(true_target)

            acc = 100 * (correct.sum() / true_target.shape[0])

            return acc

    def embed_loss_softmax_sup(self, cos_sim_matrix):
        N, M, _ = cos_sim_matrix.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        nloss = self.criterion(
            cos_sim_matrix.view(-1, N),
            torch.repeat_interleave(label, repeats=M, dim=0).cuda(),
        )

        return nloss

    def forward(self, x):
        cos_sim_matrix = self.compute_similarity_matrix(x)
        loss = self.embed_loss_softmax_sup(cos_sim_matrix)

        return loss


class GE2ELossLatent(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        GE2E loss for contrastive learning in the latent space.
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        """
        super(GE2ELossLatent, self).__init__()
        self.args = args
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "softmax-sup", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax

        if self.loss_method == "softmax-sup":
            self.embed_loss = self.embed_loss_softmax_sup

        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )

                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.einsum(
                            "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)

        return torch.stack(L).sum() / len(torch.stack(L).view(-1))

    def embed_loss_softmax_sup(self, dvecs, cos_sim_matrix):
        self.criterion = torch.nn.CrossEntropyLoss()

        N, M, _ = dvecs.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        nloss = self.criterion(
            cos_sim_matrix.view(-1, N),
            torch.repeat_interleave(label, repeats=M, dim=0).cuda(),
        )

        return nloss

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L).sum() / len(torch.stack(L).view(-1))

    @staticmethod
    # def calc_acc(cos_sim_matrix, batch_size, y, spk_per_bucket):
    def calc_acc(cos_sim_matrix, y, spk_per_bucket):
        # num_utt_per_spk = batch_size // (spk_per_bucket)

        num_utt_per_spk = cos_sim_matrix.shape[1]

        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = y.view(spk_per_bucket, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            total_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(total_target)

            acc = 100 * (correct.sum() / total_target.shape[0])

            return acc

    @staticmethod
    def pred_indx(cos_sim_matrix):
        num_utt_per_spk = cos_sim_matrix.shape[1]

        with torch.no_grad():
            pred_list = []
            for i in range(num_utt_per_spk):
                _, pred = torch.max(cos_sim_matrix[:, i, :], dim=0)
                pred_list.append(pred.tolist())
            total_pred = torch.tensor(pred_list).view(-1)

            return total_pred

    @staticmethod
    def logits_probs(cos_sim_matrix):
        softmax = nn.Softmax(dim=1)

        num_spk_so_far = cos_sim_matrix.shape[0]
        num_utts = cos_sim_matrix.shape[1]

        with torch.no_grad():
            logit_list = []
            for i in range(num_spk_so_far):
                logit = cos_sim_matrix[i, :, :].view(num_utts, num_spk_so_far)
                logit_list.append(logit)
            total_logit = torch.cat(logit_list, dim=0).reshape(-1, num_spk_so_far)

            return total_logit, softmax(total_logit)

    @staticmethod
    def real_indx(t_buffer, spks_per_buckets_sofar, epoch_test):
        t_buffer = t_buffer.reshape(spks_per_buckets_sofar * epoch_test, -1)

        num_utt_per_spk = t_buffer.shape[1]
        target_list = []
        for i in range(num_utt_per_spk):
            target = t_buffer[:, i]
            target_list.append(target.tolist())

        total_target = torch.tensor(target_list).view(-1)

        return total_target

    @staticmethod
    def calc_acc_train(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        pred_list, target_list = [], []
        # for i in range(cos_sim_matrix.shape[1]):
        for i in range(num_utt_per_spk):
            pred = cos_sim_matrix[:, i, :].argmax(dim=0)
            target = y.view(spk_per_bucket, -1)[:, i]

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

        total_pred = torch.tensor(pred_list).view(-1)
        total_target = torch.tensor(target_list).view(-1)

        correct = total_pred.eq(total_target)

        acc = 100 * (correct.sum() / total_target.shape[0])

        return acc

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        cos_sim_matrix = self.compute_similarity_matrix(dvecs)
        L = self.embed_loss(dvecs, cos_sim_matrix)

        return L


class AngProtoLossStable(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0):
        """
        AngProtoLoss for contrastive learning in the latent space.
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        """
        super(AngProtoLossStable, self).__init__()
        self.args = args
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def calc_new_centroids(self, dvecs, centroids, spkr):
        """
        Calculates the new centroids excluding the reference utterance
        """
        _, M, _ = dvecs.shape

        excl = dvecs[spkr, :M]
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        _, M, _ = dvecs.shape

        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx)

            cs_row = []
            for utt_indx, utterance in enumerate(speaker):
                # vector based cosine similarity for speed

                if utt_indx == M - 1:
                    cs_row.append(
                        torch.clamp(
                            torch.einsum(
                                "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                            )
                            / (
                                torch.norm(utterance) * torch.norm(new_centroids, dim=1)
                            ),
                            1e-6,
                        )
                    )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax_sup(self, dvecs, cos_sim_matrix):
        self.criterion = torch.nn.CrossEntropyLoss()

        N, _, _ = dvecs.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        nloss = self.criterion(cos_sim_matrix.view(-1, N), label.cuda())

        return nloss

    @staticmethod
    def calc_acc(cos_sim_matrix):
        N, _, _ = cos_sim_matrix.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        _target = label

        with torch.no_grad():
            pred_list, target_list = [], []

            pred = cos_sim_matrix[:, :].argmax(dim=0)
            target = _target.view(N, -1)

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            true_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(true_target)

            acc = 100 * (correct.sum() / true_target.shape[0])

            return acc

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        cos_sim_matrix = self.compute_similarity_matrix(dvecs)
        L = self.embed_loss_softmax_sup(dvecs, cos_sim_matrix)

        return L


class GE2ELossStable(nn.Module):
    def __init__(self, args, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        GE2E loss for contrastive learning in the latent space.
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        """
        super(GE2ELossStable, self).__init__()
        self.args = args
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "softmax-sup", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax

        if self.loss_method == "softmax-sup":
            self.embed_loss = self.embed_loss_softmax_sup

        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )

                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.einsum(
                            "ij, ki -> jk", utterance.unsqueeze(1), new_centroids
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)

        return torch.stack(L).sum() / len(torch.stack(L).view(-1))

    def embed_loss_softmax_sup(self, dvecs, cos_sim_matrix):
        self.criterion = torch.nn.CrossEntropyLoss()

        N, M, _ = dvecs.shape

        label = torch.from_numpy(np.asarray(range(0, N))).cuda()
        label = label.clone().detach().to(dtype=torch.long)

        nloss = self.criterion(
            cos_sim_matrix.view(-1, N),
            torch.repeat_interleave(label, repeats=M, dim=0).cuda(),
        )

        return nloss

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L).sum() / len(torch.stack(L).view(-1))

    @staticmethod
    def calc_acc(cos_sim_matrix):
        num_utt_per_spk = cos_sim_matrix.shape[1]

        N, M, _ = cos_sim_matrix.shape

        label = torch.from_numpy(np.asarray(range(0, N)))
        label = label.clone().detach().to(dtype=torch.long)

        _target = torch.repeat_interleave(label, repeats=M, dim=0).view(-1)

        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = _target.view(N, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            true_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(true_target)

            acc = 100 * (correct.sum() / true_target.shape[0])

            return acc

    @staticmethod
    def pred_indx(cos_sim_matrix):
        num_utt_per_spk = cos_sim_matrix.shape[1]

        with torch.no_grad():
            pred_list = []
            for i in range(num_utt_per_spk):
                _, pred = torch.max(cos_sim_matrix[:, i, :], dim=0)
                pred_list.append(pred.tolist())
            total_pred = torch.tensor(pred_list).view(-1)

            return total_pred

    @staticmethod
    def logits_probs(cos_sim_matrix):
        softmax = nn.Softmax(dim=1)

        num_spk_so_far = cos_sim_matrix.shape[0]
        num_utts = cos_sim_matrix.shape[1]

        with torch.no_grad():
            logit_list = []
            for i in range(num_spk_so_far):
                logit = cos_sim_matrix[i, :, :].view(num_utts, num_spk_so_far)
                logit_list.append(logit)
            total_logit = torch.cat(logit_list, dim=0).reshape(-1, num_spk_so_far)

            return total_logit, softmax(total_logit)

    @staticmethod
    def real_indx(t_buffer, spks_per_buckets_sofar, epoch_test):
        t_buffer = t_buffer.reshape(spks_per_buckets_sofar * epoch_test, -1)

        num_utt_per_spk = t_buffer.shape[1]
        target_list = []
        for i in range(num_utt_per_spk):
            target = t_buffer[:, i]
            target_list.append(target.tolist())

        total_target = torch.tensor(target_list).view(-1)

        return total_target

    @staticmethod
    def calc_acc_train(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        pred_list, target_list = [], []
        # for i in range(cos_sim_matrix.shape[1]):
        for i in range(num_utt_per_spk):
            pred = cos_sim_matrix[:, i, :].argmax(dim=0)
            target = y.view(spk_per_bucket, -1)[:, i]

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

        total_pred = torch.tensor(pred_list).view(-1)
        total_target = torch.tensor(target_list).view(-1)

        correct = total_pred.eq(total_target)

        acc = 100 * (correct.sum() / total_target.shape[0])

        return acc

    def compute_similarity_matrix(self, dvecs):
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        return cos_sim_matrix

    def forward(self, dvecs):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """
        cos_sim_matrix = self.compute_similarity_matrix(dvecs)
        L = self.embed_loss(dvecs, cos_sim_matrix)

        return L


# These functions and class are used for the second implementation
def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1], -1
    )
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1],
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.size(0))
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def calc_acc_cont(cos_sim_matrix, x, y, spk_per_bucket):
    num_utt_per_spk = x.shape[0] // (spk_per_bucket)
    with torch.no_grad():
        pred_list, target_list = [], []
        for i in range(num_utt_per_spk):
            pred = cos_sim_matrix[:, i, :].argmax(dim=0)
            target = y.view(spk_per_bucket, -1)[:, i]

            pred_list.append(pred.tolist())
            target_list.append(target.tolist())

        total_pred = torch.tensor(pred_list).view(-1)
        total_target = torch.tensor(target_list).view(-1)

        correct = total_pred.eq(total_target)

        acc = 100 * (correct.sum() / total_target.shape[0])

        return acc


class GE2ELossB(nn.Module):
    def __init__(self, device):
        super(GE2ELossB, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def compute_similarity_matrix(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        return sim_matrix

    @staticmethod
    def calc_acc_cont(cos_sim_matrix, batch_size, y, spk_per_bucket):
        num_utt_per_spk = batch_size // (spk_per_bucket)
        with torch.no_grad():
            pred_list, target_list = [], []
            for i in range(num_utt_per_spk):
                pred = cos_sim_matrix[:, i, :].argmax(dim=0)
                target = y.view(spk_per_bucket, -1)[:, i]

                pred_list.append(pred.tolist())
                target_list.append(target.tolist())

            total_pred = torch.tensor(pred_list).view(-1)
            total_target = torch.tensor(target_list).view(-1)

            correct = total_pred.eq(total_target)

            acc = 100 * (correct.sum() / total_target.shape[0])

            return acc

    def forward(self, embeddings):
        sim_matrix = self.compute_similarity_matrix(embeddings)
        loss, _ = calc_loss(sim_matrix)
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # It is not required !
        # contrast_feature = features.view(
        #     -1, self.args.dim_emb
        # )  # just convert to two dimension for dim_emb dimension

        contrast_feature = features.view(
            -1, features.shape[2]
        )  # just convert to two dimension for flexible dimension

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask (we don't need to do this!)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask = mask

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos

        loss = loss.view(anchor_count, -1).mean()

        return loss


# Class for the supervised contrastive loss (version 2)
class SupConLossV2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode="all"):
        super(SupConLossV2, self).__init__()
        self.args = args
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, features_aug, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # if labels.shape[0] != batch_size:
            #     raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # It is not required !
        contrast_feature = features.view(
            -1, self.args.dim_emb
        )  # just convert to two dimension

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            # anchor_feature = features.view(
            #     -1, self.args.dim_emb
            # )  # just convert to two dimension
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask (we don't need to do this!)
        # mask = mask.repeat(anchor_count, contrast_count)
        mask = mask

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # index
            0,
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss
