"""FastAPI application: app entry, routes, schemas, middleware."""
from src.api.main import app
from src.api.routes import router

__all__ = ["app", "router"]
