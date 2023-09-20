from utils import (
    # model_loader_dvec_dynamic_reg_per_bkt,
    # cont_loss_loader_dvec_dynamic_reg_per_bkt,
    # model_loader_dvec_latent_dynamic_reg,
    # cont_loss_loader_dvec_latent_dynamic_reg,
    create_calibrated_length,
    save_model_ckpt_dvec,
    save_model_ckpt_cls,
)


class AgentUnSupervisedNewReg:
    def __init__(self, args, device, hparams):

        self.args = args
        self.device = device
        self.hparams = hparams

        self.valid_every = args.valid_every

    def train_dvec_proposed(
        self,
        model_dvec,
        opt_dvec_type,
        cont_loss,
        bucket_id,
        input_data,
        epoch,
        total_num_spk_per_bkt,
        filename_dvec,
        filename_dvec_reg,
        filename_dvec_reg_dir,
    ):
        # Load the available checkpoints
        # model_dvec, _ = model_loader_dvec_dynamic_reg_per_bkt(
        #     self.args,
        #     self.hparams,
        #     dvector,
        #     filename_dvec,
        #     filename_dvec_reg,
        # )

        # cont_loss, _ = cont_loss_loader_dvec_dynamic_reg_per_bkt(
        #     self.args,
        #     self.hparams,
        #     contrastive_loss,
        #     filename_dvec,
        #     filename_dvec_reg,
        # )

        optimizer_dvec = opt_dvec_type(
            [
                {
                    "params": list(model_dvec.parameters())
                    + list(cont_loss.parameters()),
                    "weight_decay": self.hparams.weight_decay,
                }
            ],
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            nesterov=self.hparams.nesterov,
            dampening=self.hparams.dampening,
        )

        # Set up model for training
        model_dvec.train()
        cont_loss.train()

        x = input_data["x"].to(self.device)

        for _ in range(self.args.epochs_per_dvector):

            optimizer_dvec.zero_grad()

            output = model_dvec(x)

            loss = cont_loss(output.view(total_num_spk_per_bkt, -1, self.args.dim_emb))

            loss.backward()

            optimizer_dvec.step()

        # Save the checkpoints for "model_dvector"
        if epoch % self.args.save_every == 0:
            model_dvec.to("cpu")

            save_model_ckpt_dvec(
                epoch,
                model_dvec,
                optimizer_dvec,
                cont_loss,
                loss,
                bucket_id,
                filename_dvec_reg_dir,
            )

            model_dvec.to(self.device)

    def train_dvec(
        self,
        model_dvec,
        opt_dvec_type,
        cont_loss,
        bucket_id,
        input_data,
        epoch,
        total_num_spk_per_bkt,
        filename_dvec,
        filename_dvec_reg,
        filename_dvec_reg_dir,
    ):

        # Load the available checkpoints
        # model_dvec, _ = model_loader_dvec_dynamic_reg_per_bkt(
        #     self.args,
        #     self.hparams,
        #     dvector,
        #     filename_dvec,
        #     filename_dvec_reg,
        # )

        # cont_loss, _ = cont_loss_loader_dvec_dynamic_reg_per_bkt(
        #     self.args,
        #     self.hparams,
        #     contrastive_loss,
        #     filename_dvec,
        #     filename_dvec_reg,
        # )

        optimizer_dvec = opt_dvec_type(
            [
                {
                    "params": list(model_dvec.parameters())
                    + list(cont_loss.parameters()),
                    "weight_decay": self.hparams.weight_decay,
                }
            ],
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            nesterov=self.hparams.nesterov,
            dampening=self.hparams.dampening,
        )

        # Set up model for training
        model_dvec.train()
        cont_loss.train()

        x = input_data["x"].to(self.device)

        optimizer_dvec.zero_grad()

        output = model_dvec(x)

        loss = cont_loss(output.view(total_num_spk_per_bkt, -1, self.args.dim_emb))
        loss.backward()

        optimizer_dvec.step()

        # Save the checkpoints for "model_dvector"
        model_dvec.to("cpu")

        save_model_ckpt_dvec(
            epoch,
            model_dvec,
            optimizer_dvec,
            cont_loss,
            loss,
            bucket_id,
            filename_dvec_reg_dir,
        )

        model_dvec.to(self.device)

    def train_dvec_latent_proposed(
        self,
        model_dvec_latent,
        model_ma,
        opt_cls_type,
        cont_loss,
        input_buffer,
        epoch,
        ma_n,
        filename_dvec_latent,
        filename_dvec_latent_reg,
        filename_dvec_latent_reg_dir,
    ):
        # Load the available checkpoints
        # model_dvec_latent, _ = model_loader_dvec_latent_dynamic_reg(
        #     self.args,
        #     self.hparams,
        #     model,
        #     filename_dvec_latent,
        #     filename_dvec_latent_reg,
        # )

        # cont_loss, _ = cont_loss_loader_dvec_latent_dynamic_reg(
        #     self.args,
        #     self.hparams,
        #     contrastive_loss,
        #     filename_dvec_latent,
        #     filename_dvec_latent_reg,
        # )

        optimizer = opt_cls_type(
            [
                {
                    "params": list(model_dvec_latent.parameters())
                    + list(cont_loss.parameters()),
                    "weight_decay": self.hparams.weight_decay,
                }
            ],
            lr=self.hparams.lr_cls,
            momentum=self.hparams.momentum,
            nesterov=self.hparams.nesterov,
            dampening=self.hparams.dampening,
        )

        # Set up model for training
        model_dvec_latent.train()
        cont_loss.train()

        x_buffer_noncalibrated = input_buffer["feat"]
        t_buffer_noncalibrated = input_buffer["label"]

        calibrated_length, num_spks, num_utts = create_calibrated_length(
            x_buffer_noncalibrated,
            t_buffer_noncalibrated,
            self.args.dim_emb,
        )

        x_buffer = (
            x_buffer_noncalibrated.view(-1)[:calibrated_length]
            .view(
                (num_spks * num_utts),
                self.args.dim_emb,
            )
            .to(self.device)
        )

        for _ in range(self.args.epochs_per_dvector_latent):

            optimizer.zero_grad()

            out = model_dvec_latent(x_buffer)

            loss = cont_loss(
                out.view(
                    num_spks,
                    num_utts,
                    self.args.latent_dim,
                )
            )

            # loss = cont_loss(out.view(-1, 1, self.args.latent_dim))

            loss.backward()

            optimizer.step()

        # Save the checkpoint for "model"
        if epoch % self.args.save_every == 0:
            model_dvec_latent.to("cpu")

            save_model_ckpt_cls(
                epoch,
                self.hparams.round_num,
                model_dvec_latent,
                model_ma,
                optimizer,
                cont_loss,
                loss,
                ma_n,
                filename_dvec_latent_reg_dir,
            )

            model_dvec_latent.to(self.device)

    def train_dvec_latent(
        self,
        model_dvec_latent,
        model_ma,
        opt_cls_type,
        cont_loss,
        input_buffer,
        epoch,
        ma_n,
        filename_dvec_latent,
        filename_dvec_latent_reg,
        filename_dvec_latent_reg_dir,
    ):
        # Load the available checkpoints
        # model_dvec_latent, _ = model_loader_dvec_latent_dynamic_reg(
        #     self.args,
        #     self.hparams,
        #     model,
        #     filename_dvec_latent,
        #     filename_dvec_latent_reg,
        # )

        # cont_loss, _ = cont_loss_loader_dvec_latent_dynamic_reg(
        #     self.args,
        #     self.hparams,
        #     contrastive_loss,
        #     filename_dvec_latent,
        #     filename_dvec_latent_reg,
        # )

        optimizer = opt_cls_type(
            [
                {
                    "params": list(model_dvec_latent.parameters())
                    + list(cont_loss.parameters()),
                    "weight_decay": self.hparams.weight_decay,
                }
            ],
            lr=self.hparams.lr_cls,
            momentum=self.hparams.momentum,
            nesterov=self.hparams.nesterov,
            dampening=self.hparams.dampening,
        )

        # Set up model for training
        model_dvec_latent.train()
        cont_loss.train()

        x_buffer_noncalibrated = input_buffer["feat"]
        t_buffer_noncalibrated = input_buffer["label"]

        calibrated_length, num_spks, num_utts = create_calibrated_length(
            x_buffer_noncalibrated,
            t_buffer_noncalibrated,
            self.args.dim_emb,
        )

        x_buffer = (
            x_buffer_noncalibrated.view(-1)[:calibrated_length]
            .view(
                (num_spks * num_utts),
                self.args.dim_emb,
            )
            .to(self.device)
        )

        optimizer.zero_grad()

        out = model_dvec_latent(x_buffer)

        loss = cont_loss(
            out.view(
                num_spks,
                num_utts,
                self.args.latent_dim,
            )
        )

        # loss = cont_loss(out.view(-1, 1, self.args.latent_dim))

        loss.backward()

        optimizer.step()

        # Save the checkpoint for "model"
        model_dvec_latent.to("cpu")

        save_model_ckpt_cls(
            epoch,
            self.hparams.round_num,
            model_dvec_latent,
            model_ma,
            optimizer,
            cont_loss,
            loss,
            ma_n,
            filename_dvec_latent_reg_dir,
        )

        model_dvec_latent.to(self.device)
