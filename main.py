import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gliner import GLiNER

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from any origin
    allow_credentials=True,  # Whether to allow cookies
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

with open('modelos/classificador_mlp.pkl', 'rb') as file:
    classificador = pickle.load(file)

with open('modelos/vetorizador.pkl', 'rb') as file:
    vetorizador = pickle.load(file)

labels = ["person", "date"]
extrator_de_entidades = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

def vetorizar(comando: str):
    return vetorizador.transform([comando]).toarray()

@app.get("/classificar")
async def classifica(comando: str):
    retorno = classificador.predict(
        vetorizar(comando)
    )

    entities = extrator_de_entidades.predict_entities(comando, labels, threshold=0.5)

    return {"acao": retorno[0], "entidades": entities}
    