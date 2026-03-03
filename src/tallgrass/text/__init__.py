"""Bill text retrieval — multi-state-ready architecture.

Public API re-exports for the text subpackage.
"""

from tallgrass.text.fetcher import BillTextFetcher as BillTextFetcher
from tallgrass.text.kansas import KansasAdapter as KansasAdapter
from tallgrass.text.models import BillDocumentRef as BillDocumentRef
from tallgrass.text.models import BillText as BillText
from tallgrass.text.openstates import OpenStatesAdapter as OpenStatesAdapter
from tallgrass.text.protocol import StateAdapter as StateAdapter
