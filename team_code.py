# main.py (your single editable competition file)
MODEL_TYPE = "xgb"  # options: "xgb", "transfer", "cnn"

if MODEL_TYPE == "xgb":
    from model_xgb import train_model, load_model, run_model
elif MODEL_TYPE == "transfer":
    from model_transfer import train_model, load_model, run_model
elif MODEL_TYPE == "cnn":
    from model_cnn import train_model, load_model, run_model
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")