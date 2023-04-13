from potassium import Potassium, Request, Response
from tqdm import tqdm
from typing import List

import itertools
import more_itertools
import torch

app = Potassium("soundchoice-g2p")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    context = {
        "device": device,
        "model": model,
        "soundchoice_batch_size": 32,
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    text_list = request.json.get("text_list")

    device = context.get("device")
    model = context.get("model")
    soundchoice_batch_size: int = context.get("soundchoice_batch_size")

    text_list_str: str = str(text_list)[:100]
    print(
        f"Running soundchoice on device: '{str(device)}' "
        f"on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'....."
    )
    phoneme_list: List[List[str]] = list(
        itertools.chain.from_iterable(
            model(text_list_chunk)
            for text_list_chunk in tqdm(
                list(more_itertools.chunked(text_list, soundchoice_batch_size))
            )
        )
    )
    print(
        f"Finished running soundchoice  on device: '{str(device)}' "
        f"on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'!"
    )

    return Response(json={"outputs": phoneme_list}, status=200)


if __name__ == "__main__":
    app.serve()
