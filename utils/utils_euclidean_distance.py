def pairwise_distances(x, y, matching_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """

    EPSILON = 1e-8
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == "l2":
        distances = (
            (x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1))
            .pow(2)
            .sum(dim=2)
        )
        return distances
    elif matching_fn == "cosine":
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == "dot":
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError("Unsupported similarity function"))


def compute_prototypes(support, k, n):
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes