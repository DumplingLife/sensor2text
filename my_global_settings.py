settings = {
    "get_features": False,
    "feature_name": "",

    # use the post-qformer saved embeddings, subbed in for the video branch
    # "use_emg_embedding": True,
    "use_emg_embedding": False,

    # use the pre-qformer imagebind embeddings, subbed in for audio branch
    # "use_imagebind_embedding": True
    "use_imagebind_embedding": False
}

# turning both of these on is weird, probably bug
assert not (settings["use_emg_embedding"] and settings["use_imagebind_embedding"])