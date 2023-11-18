settings = {
    "get_features": False,
    "feature_name": "",
    # "use_emg_embedding": True,
    "use_emg_embedding": False,
    "use_imagebind_embedding": True
}

# turning both of these on is weird, probably bug
assert not (settings["use_emg_embedding"] and settings["use_imagebind_embedding"])