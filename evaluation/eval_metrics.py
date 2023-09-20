import torch


def eval_metrics_cont_loss(args, model, inputs, criterion, mode="val"):
    model.eval()

    with torch.no_grad():
        if mode == "val":
            out = model(inputs["x_val"])

            _out = out.view(-1, args.dim_emb)
            scale = args.spk_per_bucket * (_out.shape[0] // args.spk_per_bucket)
            out_reshaped = out[:scale].view(args.spk_per_bucket, -1, args.dim_emb)

            loss_cont = criterion(out_reshaped)

            cos_sim_matrix = criterion.compute_similarity_matrix(out_reshaped)

            acc = criterion.calc_acc(cos_sim_matrix)
            loss = loss_cont.item()
        else:
            out = model(inputs["x"])

            _out = out.view(-1, args.dim_emb)
            scale = args.spk_per_bucket * (_out.shape[0] // args.spk_per_bucket)
            out_reshaped = out[:scale].view(args.spk_per_bucket, -1, args.dim_emb)

            loss_cont = criterion(out_reshaped)

            cos_sim_matrix = criterion.compute_similarity_matrix(out_reshaped)

            acc = criterion.calc_acc(cos_sim_matrix)
            loss = loss_cont.item()

    return acc, loss
