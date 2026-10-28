"""
app/api/routes/speech.py
Speech-to-text endpoint using OpenAI Whisper.
POST /api/speech/transcribe  — upload audio file, get transcript
POST /api/speech/verify      — TTS read-back of extracted fields (plays on server)
POST /api/speech/synthesize  — TTS as downloadable audio bytes (plays in browser)
GET  /api/speech/script      — get verification script text without speaking
"""
import os
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import Response

from app.core.security import get_current_user
from app.services import eval_service
from app.services.speech_service import SpeechService
from app.services.tts_service import speak_verification, get_verification_script, synthesize_bytes
from app.schemas.schemas import TranscriptionResponse

router = APIRouter()

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a",
    "audio/ogg", "audio/webm",
    "application/octet-stream",  # Generic binary (some browsers)
}

MAX_FILE_SIZE_MB = 25


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, OGG)"),
    field_key: Optional[str] = Form(None, description="Interview field key, used to bias transcription"),
    town: Optional[str] = Form(None, pattern=r"^\d{0,10}$",
                               description="PSGC code of the town answered before ('address_barangay' hint)"),
    province: Optional[str] = Form(None, pattern=r"^\d{0,10}$",
                                   description="PSGC code of the province of birth ('place_of_birth_town' hint)"),
    # Evaluation mode (plan 003): which logged interview this try belongs to, what was asked, and the
    # original compressed recording (saved after the response, never sent to Whisper).
    eval_session_id: Optional[int] = Form(None),
    eval_field: Optional[str] = Form(None, max_length=40),
    eval_step: Optional[str] = Form(None, max_length=10),
    eval_try: Optional[str] = Form(None, max_length=40, pattern=r"^[A-Za-z0-9_-]*$"),
    eval_clip: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
    current_user=Depends(get_current_user),
):
    """
    Upload an audio file and receive a Whisper transcription.

    Accepts WAV, MP3, M4A, OGG. Max 25MB.
    `field_key` (optional, e.g. 'surname', 'sex', 'civil_status') tells
    Whisper what kind of answer to expect, which improves accuracy a lot
    on short answers. `town` (with 'address_barangay') lists that town's
    barangays in the hint; `province` (with 'place_of_birth_town') lists that
    province's towns and cities.
    """
    if audio_file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{audio_file.content_type}'. "
                   "Accepted: WAV, MP3, M4A, OGG.",
        )

    audio_bytes = await audio_file.read()
    size_mb = len(audio_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    filename = audio_file.filename or "audio.wav"
    ext = os.path.splitext(filename)[-1] or ".wav"

    try:
        result = SpeechService.transcribe_bytes(audio_bytes, suffix=ext, field_key=field_key, town=town or None,
                                                province=province or None)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )

    dropped = result.pop("dropped", None)
    passes = result.pop("passes", [])
    eval_log = None
    if eval_session_id and eval_step and (eval_field or field_key):
        name, beam = _model_used(passes, field_key)
        eval_log = dict(
            session_id=eval_session_id, user_id=current_user.id, field=eval_field or field_key, step=eval_step,
            try_id=eval_try or None, engine="whisper", model=name, beam=beam,
            transcript=result.get("transcript", ""), seconds=result.get("processing_time_seconds"),
        )
    if not result.get("transcript"):
        if eval_log:
            eval_service.log_transcribe(**eval_log)   # a silent try is a row too (no clip: nothing to hear)
        # Why it is empty, for the practice log (plan 002.2): the message stays the same for every reason.
        reason = {"echo": "dropped-echo", "repetition": "dropped-repetition"}.get(dropped, "whisper-empty")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not detect any speech in the audio. "
                   "Please speak clearly and try again.",
            headers={"X-Empty-Reason": reason},
        )

    if eval_log:
        clip = await eval_clip.read() if eval_clip is not None else None
        # No compressed recording sent: the WAV is saved instead (plan 003 A).
        eval_log["clip"] = clip if clip else audio_bytes
        eval_log["clip_type"] = (eval_clip.content_type if clip else audio_file.content_type) or "audio/wav"
        background_tasks.add_task(eval_service.log_transcribe, **eval_log)

    return TranscriptionResponse(**result)


def _model_used(passes: list, field_key: Optional[str]):
    """The model and beam that really decoded the answer; a two-pass review is "tiny>medium"."""
    if not passes:
        return SpeechService.model_and_beam_for(field_key)
    names = ">".join(p["model"] for p in passes)
    return names, passes[-1]["beam"]


@router.post("/verify")
def verify_fields(
    fields: dict,
    current_user=Depends(get_current_user),
):
    """
    Trigger TTS voice read-back of the extracted CTC fields.
    Runs speech on the server (for desktop kiosk deployments where the
    server and the kiosk speakers are the same machine).
    Also returns the script text so the UI can display it.
    """
    script = get_verification_script(fields)
    speak_result = speak_verification(fields)

    return {
        "success": speak_result,
        "script": script,
        "message": "Voice verification complete." if speak_result else "TTS unavailable on this server.",
    }


@router.post("/synthesize")
def synthesize_speech(
    payload: dict,
    current_user=Depends(get_current_user),
):
    """
    Synthesize arbitrary text to speech and return the audio bytes
    directly, so the browser (not the server machine) plays it.
    Use this for the web frontend's voice interview flow.

    Body: {"text": "...", "rate": 185}
    """
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required.")

    try:
        rate = int(payload.get("rate", 185))
    except (TypeError, ValueError):
        rate = 185
    rate = max(120, min(240, rate))

    audio_bytes, mime_type = synthesize_bytes(text, rate=rate)
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech synthesis failed.",
        )

    return Response(content=audio_bytes, media_type=mime_type)


@router.post("/script")
def get_script(
    fields: dict,
    current_user=Depends(get_current_user),
):
    """
    Return the verification script text without triggering speech.
    """
    return {"script": get_verification_script(fields)}
