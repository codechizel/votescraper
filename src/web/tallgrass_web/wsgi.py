"""WSGI config for Tallgrass."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tallgrass_web.settings.local")

application = get_wsgi_application()
