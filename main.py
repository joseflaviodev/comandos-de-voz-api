import pickle
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()

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
    