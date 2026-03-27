from fastapi import FastAPI

app = FastAPI(title="Azure Test API")

@app.get("/")
def home():
    return {"message": "Azure deployment works 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}