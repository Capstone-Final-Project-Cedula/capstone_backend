"""
app/api/routes/places.py
Read-only lists for the typing pickers (plan 002.1): the same place lists the voice matcher uses.

GET /api/places/birth?q=malay              places of birth matching what was typed ("Malaybalay City, Bukidnon")
GET /api/places/towns                      Region X towns, written as the address is ("Baungon, Bukidnon")
GET /api/places/towns/{code}/barangays     the barangays of one Region X town

Every value is written exactly as the voice path writes it, so a picked answer saves the same text.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.core.security import require_staff
from app.models.staff_user import StaffUser
from app.services import place_gazetteer
from app.services.answer_service import barangay_label, town_value
from app.services.barangay_gazetteer import region_town, region_towns, town_barangays

router = APIRouter()


class PlaceOption(BaseModel):
    code: str
    value: str


class BarangayOption(BaseModel):
    name: str
    label: str


def town_option_value(code: str, town: dict) -> str:
    """The address town as the voice path writes it ("Cagayan de Oro City, Misamis Oriental")."""
    place = place_gazetteer.place_by_code(code)
    if place:
        return town_value(place)
    return ", ".join(part for part in (town["town"], town["province"]) if part)


@router.get("/birth", response_model=List[PlaceOption])
def birth_places(
    q: str = Query("", max_length=80),
    limit: int = Query(20, ge=1, le=50),
    _: StaffUser = Depends(require_staff),
):
    return [PlaceOption(code=p.psgc, value=place_gazetteer.place_value(p))
            for p in place_gazetteer.search_birth_places(q, limit)]


@router.get("/towns", response_model=List[PlaceOption])
def towns(_: StaffUser = Depends(require_staff)):
    return [PlaceOption(code=code, value=town_option_value(code, town)) for code, town in region_towns()]


@router.get("/towns/{code}/barangays", response_model=List[BarangayOption])
def barangays(code: str, _: StaffUser = Depends(require_staff)):
    if not region_town(code):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not a Region X town.")
    names = sorted(town_barangays(code)[0], key=str.lower)
    return [BarangayOption(name=name, label=barangay_label(name)) for name in names]
