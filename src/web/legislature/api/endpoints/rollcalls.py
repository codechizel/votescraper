"""Roll call endpoints."""

from django.shortcuts import get_object_or_404
from ninja import Query, Router
from ninja.pagination import paginate

from legislature.models import RollCall

from ..filters import RollCallFilter
from ..pagination import TallgrassPagination
from ..schemas import NestedVoteOut, RollCallDetail, RollCallOut
from ..throttling import DetailRateThrottle, ListRateThrottle

router = Router()


@router.get("/", response=list[RollCallOut], throttle=[ListRateThrottle()])
@paginate(TallgrassPagination)
def list_rollcalls(request, filters: RollCallFilter = Query(...)):
    return filters.filter(RollCall.objects.select_related("session"))


@router.get("/{int:rollcall_id}/", response=RollCallDetail, throttle=[DetailRateThrottle()])
def get_rollcall(request, rollcall_id: int):
    rc = get_object_or_404(
        RollCall.objects.prefetch_related("votes__legislator"),
        pk=rollcall_id,
    )
    votes = [
        NestedVoteOut(
            id=v.id,
            legislator_slug=v.legislator.legislator_slug,
            legislator_name=v.legislator.name,
            vote=v.vote,
        )
        for v in rc.votes.all()
    ]
    return RollCallDetail(
        id=rc.id,
        session_id=rc.session_id,
        bill_number=rc.bill_number,
        bill_title=rc.bill_title,
        vote_id=rc.vote_id,
        vote_url=rc.vote_url,
        vote_datetime=rc.vote_datetime,
        vote_date=rc.vote_date,
        chamber=rc.chamber,
        motion=rc.motion,
        vote_type=rc.vote_type,
        result=rc.result,
        short_title=rc.short_title,
        sponsor=rc.sponsor,
        sponsor_slugs=rc.sponsor_slugs,
        yea_count=rc.yea_count,
        nay_count=rc.nay_count,
        present_passing_count=rc.present_passing_count,
        absent_not_voting_count=rc.absent_not_voting_count,
        not_voting_count=rc.not_voting_count,
        total_votes=rc.total_votes,
        passed=rc.passed,
        votes=votes,
    )
