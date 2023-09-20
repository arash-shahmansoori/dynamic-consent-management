import torch
import torch.nn as nn

from utils import (
    model_loader_dvec_per_bkt,
    contloss_loader_per_bkt,
    save_model_ckpt_dvec,
    save_model_ckpt_cls,
    num_correct,
)


# def set_n_training(m):
#     if isinstance(m, nn.LayerNorm):
#         m.train()


class AgentSupervisedStable:
    def __init__(self, args, device, hparams):
        self.args = args
        self.device = device
        self.hparams = hparams

        self.valid_every = args.valid_every

    def train_dvec(
        self,
        model_dvec,
        opt_dvec_type,
        cont_loss,
        bucket_id,
        input_data,
        epoch,
        filename_dvec,
        filename_dvec_dir,
    ):
        # # Load the available checkpoints
        # model_dvec, _ = model_loader_dvec_per_bkt(
        #     self.hparams,
        #     dvector,
        #     self.args,
        #     filename_dvec,
        # )

        # cont_loss, _ = contloss_loader_per_bkt(
        #     self.hparams,
        #     contrastive_loss,
        #     self.args,
        #     filename_dvec,
        # )

        # optimizer_dvec = opt_dvec_type(
        #     [
        #         {
        #             "params": list(model_dvec.parameters())
        #             + list(cont_loss.parameters()),
        #             "weight_decay": self.hparams.weight_decay,
        #         }
        #     ],
        #     lr=1e-3,
        #     momentum=self.hparams.momentum,
        #     nesterov=self.hparams.nesterov,
        #     dampening=self.hparams.dampening,
        # )

        optimizer_dvec = opt_dvec_type(
            list(model_dvec.parameters()) + list(cont_loss.parameters()),
            lr=3e-4,
            amsgrad=True,
        )

        # optimizer_dvec = opt_dvec_type(
        #     list(model_dvec.parameters()) + list(cont_loss.parameters()),
        #     lr=1e-3,
        #     betas=(0.9, 0.95),
        #     rho=0.05,
        # )
        # optimizer_dvec = opt_dvec_type(
        #     list(model_dvec.parameters()) + list(cont_loss.parameters()),
        #     lr=3e-4,
        #     betas=(0.9, 0.95),
        #     rho=0.05,
        # )

        # Set up model for training
        model_dvec.train()
        cont_loss.train()

        x = input_data["x"].to(self.device)
        y = input_data["y"].to(self.device)

        for _ in range(self.args.epochs_per_dvector):
            optimizer_dvec.zero_grad()

            output = model_dvec(x)

            # loss = cont_loss(
            #     output.view(self.args.spk_per_bucket, -1, self.args.dim_emb)
            # )

            loss = cont_loss(
                output.view(self.args.spk_per_bucket, -1, self.args.dim_emb)
            )

            loss.backward()

            optimizer_dvec.step()

            # cos_sim_matrix = cont_loss.compute_similarity_matrix(
            #     output.view(self.args.spk_per_bucket, -1, self.args.dim_emb)
            # )

            # acc = cont_loss.calc_acc(
            #     cos_sim_matrix,
            #     y - bucket_id * self.args.spk_per_bucket,
            #     self.args.spk_per_bucket,
            # )

        # Save the checkpoints for "model_dvector"
        if epoch % self.args.save_every == 0:
            save_model_ckpt_dvec(
                epoch,
                model_dvec,
                optimizer_dvec,
                cont_loss,
                loss,
                bucket_id,
                filename_dvec_dir,
            )

    def train_cls(
        self,
        model,
        model_ma,
        optimizer,
        ce_loss,
        input_buffer,
        epoch,
        ma_n,
        filename_dir,
    ):
        # Set up model for training
        model.train()

        x_buffer = input_buffer["feat"]
        t_buffer = input_buffer["label"]

        for _ in range(self.args.epochs_per_cls):
            optimizer.zero_grad()

            _, out = model(x_buffer)

            loss = ce_loss(out, t_buffer)
            loss.backward()

            optimizer.step()

        # Save the checkpoint for "model"
        if epoch % self.args.save_every == 0:
            model.to("cpu")

            save_model_ckpt_cls(
                epoch,
                self.hparams.round_num,
                model,
                model_ma,
                optimizer,
                ce_loss,
                loss,
                ma_n,
                filename_dir,
            )

            model.to(self.device)

    def accuracy_loss(
        self,
        model,
        model_ma,
        ce_loss,
        batch_x,
        batch_y,
    ):
        total_num_correct_ = 0

        eval_model = model if self.hparams.ma_mode == "no_ma" else model_ma

        eval_model.eval()
        # eval_model.apply(set_n_training)

        with torch.no_grad():
            prob, out = eval_model(batch_x)

            # corr_spk, _, _ = num_correct(prob, batch_y.view(-1), topk=1)
            corr_spk, _, _ = num_correct(out, batch_y.view(-1), topk=1)

            total_num_correct_ += corr_spk
            acc = (total_num_correct_ / len(batch_y)) * 100

            loss = ce_loss(out, batch_y)

        return acc, loss

    def predict_index(
        self,
        model,
        model_ma,
        batch_x,
        batch_y,
    ):
        eval_model = model if self.hparams.ma_mode == "no_ma" else model_ma

        eval_model.eval()

        with torch.no_grad():
            prob, _ = eval_model(batch_x)
            _, predicted_index = torch.max(prob, dim=1)
            ground_truth_indx = batch_y

        return predicted_index, ground_truth_indx
