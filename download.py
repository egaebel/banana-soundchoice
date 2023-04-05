def download_model():
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")
    result = model("Priming the model to download weights, etc.")
    print(f"result: '{result}'")


if __name__ == "__main__":
    download_model()
