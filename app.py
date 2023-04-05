from potassium import Potassium, Request, Response
from typing import List

import itertools
import more_itertools
import torch

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    context = {"model": model, "soundchoice_batch_size": 32}

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    text_list = request.json.get("text_list")
    model = context.get("model")
    soundchoice_batch_size = context.get("soundchoice_batch_size")

    phoneme_list: List[List[str]] = list(
        itertools.chain.from_iterable(
            model(text_list_chunk)
            for text_list_chunk in more_itertools.chunked(
                text_list, soundchoice_batch_size
            )
        )
    )

    return Response(json={"outputs": phoneme_list}, status=200)


if __name__ == "__main__":
    app.serve()
