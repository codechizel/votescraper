# Default to local settings for development.
# Production deployments override via DJANGO_SETTINGS_MODULE env var.
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.local")
