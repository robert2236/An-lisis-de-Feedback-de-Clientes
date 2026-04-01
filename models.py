from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# Productos
class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None

class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime
    total_reviews: int = 0
    positive_count: int = 0
    negative_count: int = 0
    
    class Config:
        from_attributes = True

# Reseñas
class ReviewCreate(BaseModel):
    product_id: int
    review_text: str

class ReviewResponse(BaseModel):
    id: int
    product_id: int
    review_text: str
    cleaned_text: str
    sentiment: str
    confidence_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# Análisis de sentimiento
class SentimentAnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    cleaned_text: str

# Estadísticas
class StatsResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    positive_percentage: float
    negative_percentage: float
    avg_confidence: float