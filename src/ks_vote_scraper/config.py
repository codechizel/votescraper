"""Configuration constants for the KS Legislature vote scraper."""

BASE_URL = "https://www.kslegislature.gov"

REQUEST_DELAY = 0.15  # seconds between requests (rate-limited via lock)
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
MAX_WORKERS = 5  # concurrent fetch threads

USER_AGENT = (
    "KSLegVoteScraper/0.2 "
    "(Research project; collecting public roll call vote data; "
    "contact: joseph.claeys@gmail.com)"
)
