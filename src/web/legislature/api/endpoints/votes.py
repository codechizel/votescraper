"""Vote endpoints."""

from django.db.models import F
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import Vote

from ..filters import VoteFilter
from ..pagination import TallgrassPagination
from ..schemas import VoteOut
from ..throttling import ListRateThrottle

router = Router()


@router.get("/", response=list[VoteOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_votes(request, filters: VoteFilter = Query(...)):
    qs = filters.filter(Vote.objects.select_related("rollcall", "legislator"))
    return qs.annotate(
        legislator_slug=F("legislator__legislator_slug"),
        legislator_name=F("legislator__name"),
        rollcall_vote_id=F("rollcall__vote_id"),
        rollcall_bill_number=F("rollcall__bill_number"),
    )
