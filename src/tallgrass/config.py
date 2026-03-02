"""Configuration constants for the KS Legislature vote scraper."""

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("tallgrass")
except Exception:
    _VERSION = "dev"

BASE_URL = "https://www.kslegislature.gov"

REQUEST_DELAY = 0.15  # seconds between requests (rate-limited via lock)
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
MAX_WORKERS = 5  # concurrent fetch threads

# Retry wave settings — back off and retry transient failures in gentle passes
RETRY_WAVES = 3  # additional retry passes after initial fetch
WAVE_COOLDOWN = 90  # seconds to wait between retry waves
WAVE_WORKERS = 2  # reduced concurrency during retry waves
WAVE_DELAY = 0.5  # slower rate limit during retry waves (vs 0.15s normal)

BILL_TITLE_MAX_LENGTH = 500  # truncate excessively long titles for display

USER_AGENT = (
    f"Tallgrass/{_VERSION} "
    "(Research project; collecting public roll call vote data; "
    "https://github.com/codechizel/tallgrass)"
)
