import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from any origin
    allow_credentials=True,  # Whether to allow cookies
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

with open('modelos/classificador_naive_bayes.pkl', 'rb') as file:
    classificador = pickle.load(file)

with open('modelos/vetorizador.pkl', 'rb') as file:
    vetorizador = pickle.load(file)

def vetorizar(comando: str):
    return vetorizador.transform([comando]).toarray()

@app.get("/classificar")
async def classifica(comando: str):
    retorno = classificador.predict(
        vetorizar(comando)
    )

    return {"acao": retorno[0]}
    