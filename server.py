# ==========================================================================
# Fastai code
# ==========================================================================
from fastai.learner import load_learner


class Brain:
    def __init__(self, model_path):
        self.model = load_learner(model_path)

    def predict(self, image_bytes):
        result = self.model.predict(image_bytes)
        idx = result[1]
        return {"label": result[0], "confidence": float(result[2][idx])}


model_path = "mountains-v1.pkl"

brain = Brain(model_path)

# ==========================================================================
# Web app code
# ==========================================================================
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse

templates = Jinja2Templates(directory="templates")


async def homepage(request):
    return templates.TemplateResponse("mountains.html", {"request": request})


# @app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    print(data)
    image_data = await (data["file"].read())
    result = brain.predict(image_data)
    return templates.TemplateResponse(
        "mountains.html",
        {
            "request": request,
            "label": result["label"],
            "confidence": result["confidence"],
        },
    )


routes = [
    Route("/", endpoint=homepage),
    Route("/upload", endpoint=upload, methods=["POST"]),
    Mount("/static", StaticFiles(directory="static"), name="static"),
]

app = Starlette(debug=True, routes=routes)
