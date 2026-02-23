"""
API - 对外接口

FastAPI 服务，提供 RESTful API
"""

from .server import create_app

__all__ = ["create_app"]
