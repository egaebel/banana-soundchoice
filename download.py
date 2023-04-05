import torch


def download_model():
    device = 0 if torch.cuda.is_available() else -1
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    model("Priming the model to download weights, etc.")


if __name__ == "__main__":
    download_model()
