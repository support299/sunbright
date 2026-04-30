import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "unsafe-dev-secret")
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"
ALLOWED_HOSTS = [host for host in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if host]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework_simplejwt",
    "dashboard",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]
WSGI_APPLICATION = "backend.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.sqlite3"),
        "NAME": os.getenv("DB_NAME", BASE_DIR / "db.sqlite3"),
        "USER": os.getenv("DB_USER", ""),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", ""),
        "PORT": os.getenv("DB_PORT", ""),
    }
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PAGINATION_CLASS": "dashboard.pagination.DefaultPagination",
    "PAGE_SIZE": 20,
    "EXCEPTION_HANDLER": "dashboard.exceptions.custom_exception_handler",
}

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Google Sign-In (verify ID tokens from the SPA). Set VITE_GOOGLE_CLIENT_ID to the same Web client ID.
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
# Optional: restrict sign-in to addresses like "user@yourdomain.com"
GOOGLE_OAUTH_ALLOWED_DOMAIN = os.getenv("GOOGLE_OAUTH_ALLOWED_DOMAIN", "").strip().lower()
# Comma-separated emails that receive is_staff=True on first Google login (Django admin can also promote users)
GOOGLE_OAUTH_ADMIN_EMAILS = frozenset(
    e.strip().lower()
    for e in os.getenv("GOOGLE_OAUTH_ADMIN_EMAILS", "").split(",")
    if e.strip()
)

# AI insights (POST /api/insights/generate/): without BUILT_IN_FORGE_API_KEY, responses use local heuristics from DB metrics.
# With key: chat completions API (Forge default, or OpenAI). Same env names as sunbright-dashboard for Forge.
# Optional: BUILT_IN_FORGE_API_URL, BUILT_IN_FORGE_MODEL, BUILT_IN_FORGE_TIMEOUT_SECONDS (default 120).
# OpenAI: set BUILT_IN_FORGE_API_URL=https://api.openai.com and BUILT_IN_FORGE_MODEL=gpt-4o-mini (or gpt-4o, etc.).
# Forge-only `thinking` is omitted automatically when the URL contains openai.com (or set BUILT_IN_FORGE_SKIP_THINKING=true).
# max_tokens defaults to 16384 for api.openai.com (model limit); override with BUILT_IN_FORGE_MAX_TOKENS if needed.
