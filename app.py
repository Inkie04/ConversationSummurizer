from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Text Summarization API")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec le formulaire"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    """Générer le résumé"""
    try:
        if not text or len(text.strip()) < 10:
            error = "Le texte doit contenir au moins 10 caractères"
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": error, "input_text": text}
            )
        
        # Générer le résumé
        obj = PredictionPipeline()
        summary = obj.predict(text)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "input_text": text,
                "summary": summary,
                "input_length": len(text.split()),
                "summary_length": len(summary.split())
            }
        )
    except Exception as e:
        error = f"Erreur: {str(e)}"
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error, "input_text": text}
        )


@app.get("/train")
async def training():
    """Entraîner le modèle"""
    try:
        os.system("python main.py")
        return {"status": "success", "message": "Training successful!"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}


@app.post("/api/predict")
async def predict_api(text: str):
    """API endpoint pour prédiction (pour intégrations externes)"""
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return {
            "status": "success",
            "input": text,
            "summary": summary,
            "input_length": len(text.split()),
            "summary_length": len(summary.split())
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)