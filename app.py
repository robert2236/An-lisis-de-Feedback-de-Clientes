# backend/app.py
import sys
import os
import re
import joblib

# Añadir la carpeta actual al path para importaciones locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

# Importaciones locales con ruta correcta
from database import init_db, get_db, Product as DBProduct, Review as DBReview
from models import (
    ProductCreate, ProductResponse,
    ReviewCreate, ReviewResponse,
    SentimentAnalysisResponse,
    StatsResponse
)

# ==================== INICIALIZAR APP ====================

app = FastAPI(
    title="Sentiment Analysis API",
    description="API para análisis de sentimientos de reseñas de productos",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para desarrollo, permitir todos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CARGAR MODELO CON RUTA CORRECTA ====================

# Obtener la ruta absoluta al modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    print(f"  Buscando en: {MODEL_PATH}")
    model = None

# ==================== FUNCIONES DE UTILIDAD ====================

def clean_text(text: str) -> str:
    """Limpia el texto igual que en el entrenamiento"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text: str) -> tuple:
    """Analiza sentimiento usando el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    cleaned = clean_text(text)
    sentiment = model.predict([cleaned])[0]
    
    try:
        proba = model.predict_proba([cleaned])[0]
        confidence = float(max(proba))
    except:
        confidence = 0.5
    
    return sentiment, confidence, cleaned

# ==================== ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Inicializar base de datos al iniciar"""
    try:
        init_db()
        print("✓ Base de datos SQLite inicializada")
    except Exception as e:
        print(f"✗ Error inicializando base de datos: {e}")

@app.get("/")
def read_root():
    return {
        "message": "Sentiment Analysis API",
        "model_loaded": model is not None,
        "database": "SQLite (sentiment.db)",
        "status": "running",
        "model_path": MODEL_PATH if model else None
    }

# ==================== PRODUCTOS ====================

@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Crear un nuevo producto"""
    db_product = DBProduct(
        name=product.name,
        description=product.description,
        image_url=product.image_url
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    return ProductResponse(
        id=db_product.id,
        name=db_product.name,
        description=db_product.description,
        image_url=db_product.image_url,
        created_at=db_product.created_at,
        total_reviews=0,
        positive_count=0,
        negative_count=0
    )

@app.get("/products", response_model=List[ProductResponse])
def get_products(db: Session = Depends(get_db)):
    """Obtener todos los productos con estadísticas"""
    products = db.query(DBProduct).all()
    
    result = []
    for product in products:
        reviews = db.query(DBReview).filter(DBReview.product_id == product.id).all()
        positive_count = len([r for r in reviews if r.sentiment == "Positive"])
        
        result.append(ProductResponse(
            id=product.id,
            name=product.name,
            description=product.description,
            image_url=product.image_url,
            created_at=product.created_at,
            total_reviews=len(reviews),
            positive_count=positive_count,
            negative_count=len(reviews) - positive_count
        ))
    
    return result

@app.get("/products/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Obtener un producto específico"""
    product = db.query(DBProduct).filter(DBProduct.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    reviews = db.query(DBReview).filter(DBReview.product_id == product.id).all()
    positive_count = len([r for r in reviews if r.sentiment == "Positive"])
    
    return ProductResponse(
        id=product.id,
        name=product.name,
        description=product.description,
        image_url=product.image_url,
        created_at=product.created_at,
        total_reviews=len(reviews),
        positive_count=positive_count,
        negative_count=len(reviews) - positive_count
    )

@app.delete("/products/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    """Eliminar un producto y todas sus reseñas"""
    product = db.query(DBProduct).filter(DBProduct.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    db.delete(product)
    db.commit()
    return {"message": "Producto eliminado exitosamente"}

# ==================== RESEÑAS Y ANÁLISIS ====================

@app.post("/reviews/analyze", response_model=SentimentAnalysisResponse)
def analyze_review_only(review: ReviewCreate):
    """Analizar una reseña sin guardarla (solo predicción)"""
    sentiment, confidence, cleaned = analyze_sentiment(review.review_text)
    
    return SentimentAnalysisResponse(
        sentiment=sentiment,
        confidence=confidence,
        cleaned_text=cleaned
    )

@app.post("/reviews", response_model=ReviewResponse)
def create_review(review: ReviewCreate, db: Session = Depends(get_db)):
    """Analizar y guardar una reseña"""
    product = db.query(DBProduct).filter(DBProduct.id == review.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    sentiment, confidence, cleaned = analyze_sentiment(review.review_text)
    
    db_review = DBReview(
        product_id=review.product_id,
        review_text=review.review_text,
        cleaned_text=cleaned,
        sentiment=sentiment,
        confidence_score=confidence
    )
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    return ReviewResponse(
        id=db_review.id,
        product_id=db_review.product_id,
        review_text=db_review.review_text,
        cleaned_text=db_review.cleaned_text,
        sentiment=db_review.sentiment,
        confidence_score=db_review.confidence_score,
        created_at=db_review.created_at
    )

@app.get("/products/{product_id}/reviews", response_model=List[ReviewResponse])
def get_product_reviews(
    product_id: int, 
    skip: int = 0, 
    limit: int = 50, 
    db: Session = Depends(get_db)
):
    """Obtener todas las reseñas de un producto"""
    product = db.query(DBProduct).filter(DBProduct.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    reviews = db.query(DBReview).filter(
        DBReview.product_id == product_id
    ).order_by(DBReview.created_at.desc()).offset(skip).limit(limit).all()
    
    return reviews

@app.get("/products/{product_id}/stats", response_model=StatsResponse)
def get_product_stats(product_id: int, db: Session = Depends(get_db)):
    """Obtener estadísticas de sentimientos de un producto"""
    product = db.query(DBProduct).filter(DBProduct.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    reviews = db.query(DBReview).filter(DBReview.product_id == product_id).all()
    
    if not reviews:
        return StatsResponse(
            total_reviews=0,
            positive_count=0,
            negative_count=0,
            positive_percentage=0.0,
            negative_percentage=0.0,
            avg_confidence=0.0
        )
    
    positive_count = len([r for r in reviews if r.sentiment == "Positive"])
    negative_count = len(reviews) - positive_count
    total = len(reviews)
    
    return StatsResponse(
        total_reviews=total,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_percentage=(positive_count / total) * 100,
        negative_percentage=(negative_count / total) * 100,
        avg_confidence=sum(r.confidence_score for r in reviews) / total
    )

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)