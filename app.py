from potassium import Potassium, Request, Response

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
    context = {"model": model}

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    text_list = request.json.get("text_list")
    model = context.get("model")
    outputs = model(text_list)

    return Response(json={"outputs": outputs}, status=200)


if __name__ == "__main__":
    app.serve()
