import torch


def download_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model with device: '{device}'", flush=True)
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    result = model("Priming the model to download weights, etc.")
    print(f"result: '{result}'")


if __name__ == "__main__":
    download_model()
