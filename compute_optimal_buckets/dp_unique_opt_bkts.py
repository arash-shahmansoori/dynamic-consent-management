def unique_opt_seq_final(S):
    # This function returns the corresponding indices of unique values of
    # the optimal buckets in a list dynamically.
    # [Note]: "-1" is to indicate the optimal bucket ID selected for the speaker "i+1"
    # already exists for the speaker(s): 0,...,i.
    # Outputs:
    #           indx_out: indices of unique current values in a list
    #           S_out   : new list with the removed selected values from the previous step
    #           bkt_out : list of optimal buckets for registration
    #
    # Inputs:
    #           S     : the list of optimal buckets for registering the new speakers

    # L = len(S)

    bkt_out = []  # Base case

    # Dynamic program to find the first unique optimal bucket (Decision problem)
    for i, l in enumerate(S):
        if l in bkt_out:
            bkt_out[: i + 1] = bkt_out[:i] + [-1]
        elif l not in bkt_out:
            bkt_out[: i + 1] = bkt_out[:i] + [l]

    # To return the corresponding index of the unique optimal current buckets
    indx_out = []
    for j, b in enumerate(bkt_out):
        if b >= 0:
            indx_out.append(j)

    S_out = [i for j, i in enumerate(S) if j not in indx_out]

    return indx_out, S_out, bkt_out
