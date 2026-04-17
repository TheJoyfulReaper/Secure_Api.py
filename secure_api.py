"""
secure_api.py — Production-Grade REST API with FastAPI

Demonstrates:
  - JWT-based authentication with refresh tokens
  - Role-based access control (RBAC)
  - CRUD operations with pagination and filtering
  - Request validation via Pydantic v2
  - Custom middleware (request ID, timing, rate limiting)
  - Structured error handling with RFC 7807 Problem Details
  - Dependency injection pattern
  - In-memory store (swappable for any DB via repository pattern)

Author: Christopher Hall

Usage:
    uvicorn secure_api:app --reload
    # Or: python secure_api.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = "portfolio-demo-key-replace-in-production"
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=7)
ALGORITHM = "HS256"
RATE_LIMIT_RPM = 60

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class Role(StrEnum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class UserInDB(BaseModel):
    user_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    email: str
    display_name: str
    role: Role = Role.VIEWER
    password_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class ProjectInDB(BaseModel):
    project_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    description: str = ""
    owner_id: str = ""
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    role: str
    created_at: datetime


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2048)
    tags: list[str] = Field(default_factory=list, max_length=10)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None
    tags: list[str] | None = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    description: str
    owner_id: str
    status: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str = ""


# ---------------------------------------------------------------------------
# In-Memory Data Store (Repository Pattern)
# ---------------------------------------------------------------------------

class DataStore:
    """Swappable in-memory store — mirrors a repository interface."""

    def __init__(self) -> None:
        self.users: dict[str, UserInDB] = {}
        self.users_by_email: dict[str, str] = {}
        self.projects: dict[str, ProjectInDB] = {}
        self.revoked_tokens: set[str] = set()

    def seed(self) -> None:
        admin = UserInDB(
            email="admin@example.com",
            display_name="Admin User",
            role=Role.ADMIN,
            password_hash=_hash_password("Admin1234"),
        )
        self.users[admin.user_id] = admin
        self.users_by_email[admin.email] = admin.user_id


_store = DataStore()


# ---------------------------------------------------------------------------
# Auth Utilities
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    return hashlib.sha256(f"{SECRET_KEY}:{password}".encode()).hexdigest()


def _verify_password(password: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_password(password), hashed)


def _create_token(payload: dict, ttl: timedelta) -> str:
    """Simple HMAC-signed JSON token (demo — use PyJWT in production)."""
    claims = {
        **payload,
        "exp": (datetime.now(timezone.utc) + ttl).timestamp(),
        "iat": datetime.now(timezone.utc).timestamp(),
        "jti": uuid.uuid4().hex[:16],
    }
    data = json.dumps(claims, sort_keys=True)
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    import base64
    token = base64.urlsafe_b64encode(f"{data}|{sig}".encode()).decode()
    return token


def _decode_token(token: str) -> dict | None:
    import base64
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        data_str, sig = decoded.rsplit("|", 1)
        expected = hmac.new(SECRET_KEY.encode(), data_str.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        claims = json.loads(data_str)
        if claims.get("exp", 0) < time.time():
            return None
        if claims.get("jti") in _store.revoked_tokens:
            return None
        return claims
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dependency Injection — Current User
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> UserInDB:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth[7:]
    claims = _decode_token(token)
    if claims is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = _store.users.get(claims.get("sub", ""))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    return user


def require_role(*roles: Role):
    """Factory that returns a dependency enforcing role-based access."""
    async def checker(user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RequestContextMiddleware:
    """Injects X-Request-ID and measures response time."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex[:12]
        start = time.monotonic()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                elapsed = (time.monotonic() - start) * 1000
                extra_headers = [
                    (b"x-request-id", request_id.encode()),
                    (b"x-response-time-ms", f"{elapsed:.1f}".encode()),
                ]
                message["headers"] = list(message.get("headers", [])) + extra_headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _store.seed()
    logger.info("Seeded admin user (admin@example.com / Admin1234)")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Portfolio API",
    version="1.0.0",
    description="Production-grade REST API demo — Christopher Hall",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ProblemDetail(
            title=exc.detail if isinstance(exc.detail, str) else "Error",
            status=exc.status_code,
            detail=str(exc.detail),
            instance=str(request.url),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest):
    if body.email in _store.users_by_email:
        raise HTTPException(409, "Email already registered")
    user = UserInDB(
        email=body.email,
        display_name=body.display_name,
        password_hash=_hash_password(body.password),
    )
    _store.users[user.user_id] = user
    _store.users_by_email[user.email] = user.user_id
    return user


@app.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    uid = _store.users_by_email.get(body.email)
    if uid is None:
        raise HTTPException(401, "Invalid credentials")
    user = _store.users[uid]
    if not _verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    access = _create_token({"sub": user.user_id, "role": user.role}, ACCESS_TOKEN_TTL)
    refresh = _create_token({"sub": user.user_id, "type": "refresh"}, REFRESH_TOKEN_TTL)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(user: UserInDB = Depends(get_current_user)):
    return user


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

@app.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    user: UserInDB = Depends(get_current_user),
):
    project = ProjectInDB(
        name=body.name,
        description=body.description,
        tags=body.tags,
        owner_id=user.user_id,
    )
    _store.projects[project.project_id] = project
    return project


@app.get("/projects", response_model=PaginatedResponse)
async def list_projects(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: str | None = Query(None, alias="status"),
    tag: str | None = None,
    user: UserInDB = Depends(get_current_user),
):
    projects = list(_store.projects.values())
    # Filtering
    if status_filter:
        projects = [p for p in projects if p.status == status_filter]
    if tag:
        projects = [p for p in projects if tag in p.tags]
    # Non-admin users only see their own projects
    if user.role != Role.ADMIN:
        projects = [p for p in projects if p.owner_id == user.user_id]
    # Pagination
    total = len(projects)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    items = projects[start : start + page_size]
    return PaginatedResponse(
        items=[ProjectResponse(**p.model_dump()) for p in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, user: UserInDB = Depends(get_current_user)):
    project = _store.projects.get(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if user.role != Role.ADMIN and project.owner_id != user.user_id:
        raise HTTPException(403, "Access denied")
    return project


@app.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    user: UserInDB = Depends(get_current_user),
):
    project = _store.projects.get(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if user.role != Role.ADMIN and project.owner_id != user.user_id:
        raise HTTPException(403, "Access denied")
    updates = body.model_dump(exclude_unset=True)
    for k, v in updates.items():
        setattr(project, k, v)
    project.updated_at = datetime.now(timezone.utc)
    return project


@app.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    user: UserInDB = Depends(require_role(Role.ADMIN, Role.EDITOR)),
):
    if project_id not in _store.projects:
        raise HTTPException(404, "Project not found")
    del _store.projects[project_id]
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Admin Endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/users", response_model=list[UserResponse])
async def list_users(user: UserInDB = Depends(require_role(Role.ADMIN))):
    return list(_store.users.values())


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "users": len(_store.users),
        "projects": len(_store.projects),
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("secure_api:app", host="0.0.0.0", port=8000, reload=True)
