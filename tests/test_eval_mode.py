"""
tests/test_eval_mode.py
Plan 003: evaluation mode. Shares the throwaway SQLite database, the fixtures and the helpers of
test_backend.py. Run with: pytest tests/ -v
"""
# ruff: noqa: F811  (pytest fixtures imported from test_backend are re-used as test arguments)
from unittest.mock import patch

import pytest  # noqa: F401  (fixtures are imported below)


from tests.test_backend import (  # noqa: F401
    Base, TestIssuance, TestingSessionLocal, _auth, admin_token, admin_user, client, db,
    setup_test_db, staff_token, staff_user,
)
from app.services.speech_service import SpeechService  # after test_backend: it sets the throwaway database first


def _eval_put(client, admin_token, **body):
    resp = client.put("/api/eval/settings", json=body, headers=_auth(admin_token))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _eval_session(db, round_type="scripted", card_no=1, outcome="open", staff_id=None, round_no=99):
    """A bare session row for scoring tests (participant numbers are unique per round)."""
    from app.models.eval import EvalSession
    number = db.query(EvalSession).filter(EvalSession.round_no == round_no).count() + 1
    row = EvalSession(round_no=round_no, round_type=round_type, round_condition="quiet", participant_no=number,
                      participant_code=f"P{number:03d}", card_no=card_no if round_type == "scripted" else None,
                      input_method="voice", staff_user_id=staff_id, outcome=outcome)
    db.add(row)
    db.commit()
    return row


def _eval_try(db, session, field, step, transcript, extracted, ok=True, engine="whisper", model="medium"):
    from app.models.eval import EvalAnswer
    row = EvalAnswer(session_id=session.id, field=field, step=step, attempt_no=0, engine=engine, model=model,
                     beam=5, transcript=transcript, extracted_value=extracted, extract_ok=ok,
                     whisper_seconds=1.0)
    db.add(row)
    db.commit()
    return row


def _score(db, session, values, notes=None):
    from app.models.eval import EvalAnswer
    from app.services import eval_service
    eval_service.score_session(db, session, values, {n["field"]: n for n in (notes or [])})
    db.commit()
    return {r.field: r for r in db.query(EvalAnswer).filter(
        EvalAnswer.session_id == session.id, EvalAnswer.step == "final")}


CARD1 = {"surname": "Abarquez", "given_name": "Maria Luisa", "middle_name": "Tumampos",
         "address": "Purok 2, Barangay Imbatug, Baungon, Bukidnon", "date": "1990-03-14", "sex": "Female",
         "civil_status": "Married", "citizenship": "Filipino", "icr_no": "",
         "place_of_birth": "Malaybalay City, Bukidnon", "height": "158", "weight": "55", "occupation": "Teacher",
         "gross_annual_income": "180000"}


class TestEvalAdminOnly:
    ADMIN_ROUTES = [
        ("get", "/api/eval/settings"), ("put", "/api/eval/settings"), ("get", "/api/eval/stats"),
        ("get", "/api/eval/sessions"), ("get", "/api/eval/sessions/1"), ("get", "/api/eval/answers/1/audio"),
        ("get", "/api/eval/export/sessions.csv"), ("get", "/api/eval/export/answers.csv"),
        ("get", "/api/eval/export/summary.csv"), ("get", "/api/eval/delete-audio"), ("post", "/api/eval/delete-audio"),
    ]

    def _call(self, client, method, url, headers):
        kwargs = {"json": {}} if method in ("put", "post") else {}
        return getattr(client, method)(url, headers=headers, **kwargs)

    def test_staff_token_gets_403_on_every_admin_route(self, client, staff_token):
        for method, url in self.ADMIN_ROUTES:
            assert self._call(client, method, url, _auth(staff_token)).status_code == 403, url

    def test_bad_or_missing_token_gets_401_on_every_admin_route(self, client):
        for method, url in self.ADMIN_ROUTES:
            assert self._call(client, method, url, _auth("not-a-token")).status_code == 401, url
            assert self._call(client, method, url, {}).status_code == 401, url

    def test_staff_endpoints_are_write_only(self, client, staff_token, admin_token):
        from app.models.eval import EvalSession
        _eval_put(client, admin_token, evaluation_mode=True, round_type="practice", round_condition="quiet")
        started = client.post("/api/eval/sessions", json={"input_method": "voice"}, headers=_auth(staff_token))
        assert started.status_code == 200 and set(started.json()) == {"session_id"}
        sid = started.json()["session_id"]
        assert client.get(f"/api/eval/sessions/{sid}", headers=_auth(staff_token)).status_code == 403
        sneaky = client.post("/api/eval/sessions", json={"input_method": "voice", "round_type": "uat"},
                             headers=_auth(staff_token)).json()["session_id"]
        check = TestingSessionLocal()
        assert check.get(EvalSession, sneaky).round_type == "practice"     # staff cannot set a tag
        check.close()
        finished = client.post(f"/api/eval/sessions/{sid}/finish", json={}, headers=_auth(staff_token))
        assert finished.json() == {"ok": True}

    def test_staff_cannot_write_to_someone_elses_session(self, client, staff_token, admin_token):
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(admin_token)).json()["session_id"]
        assert client.post(f"/api/eval/sessions/{sid}/finish", json={}, headers=_auth(staff_token)).status_code == 404

    def test_consent_flag_only_says_whether_the_recording_is_on(self, client, staff_token, admin_token):
        _eval_put(client, admin_token, evaluation_mode=True)
        assert client.get("/api/eval/consent", headers=_auth(staff_token)).json() == {"recording": True}
        _eval_put(client, admin_token, evaluation_mode=False)
        assert client.get("/api/eval/consent", headers=_auth(staff_token)).json() == {"recording": False}
        started = client.post("/api/eval/sessions", json={"input_method": "voice"}, headers=_auth(staff_token))
        assert started.json() == {"session_id": None}                      # off: nothing is logged
        _eval_put(client, admin_token, evaluation_mode=True)

    def test_practice_log_on_keeps_evaluation_logging_off(self, client, staff_token):
        from app.services import eval_service
        with patch.object(eval_service.env_settings, "PRACTICE_LOG_ENABLED", True):
            assert client.get("/api/eval/consent", headers=_auth(staff_token)).json() == {"recording": False}


class TestEvalRoundsAndCards:
    def test_participant_codes_count_per_round_and_cards_wrap(self, client, staff_token, admin_token):
        from app.models.eval import EvalSession
        from app.services import eval_service
        with patch.dict(eval_service.load_cards()["sets"], {"pair": [1, 2]}):
            _eval_put(client, admin_token, evaluation_mode=True, round_type="scripted", card_set="pair",
                      new_round=True)
            ids = [client.post("/api/eval/sessions", json={"input_method": "voice"},
                               headers=_auth(staff_token)).json()["session_id"] for _ in range(3)]
            check = TestingSessionLocal()
            rows = [check.get(EvalSession, i) for i in ids]
            assert [r.participant_code for r in rows] == ["P001", "P002", "P003"]
            assert [r.card_no for r in rows] == [1, 2, 1]
            assert all(r.round_type == "scripted" and r.card_set == "pair" for r in rows)
            check.close()
        _eval_put(client, admin_token, round_type="practice", new_round=True)

    def test_a_round_change_starts_a_new_participant_count(self, client, staff_token, admin_token):
        from app.models.eval import EvalSession
        _eval_put(client, admin_token, round_type="uat", round_condition="noisy", new_round=True)
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        check = TestingSessionLocal()
        row = check.get(EvalSession, sid)
        assert row.participant_code == "P001" and row.round_condition == "noisy" and row.card_no is None
        check.close()
        _eval_put(client, admin_token, round_type="practice", round_condition="quiet", new_round=True)

    def test_unknown_card_set_is_refused(self, client, admin_token):
        resp = client.put("/api/eval/settings", json={"round_type": "scripted", "card_set": "nope"},
                          headers=_auth(admin_token))
        assert resp.status_code == 422

    def test_reading_cards_cover_14_fields_on_8_cards(self):
        from app.services import eval_service
        cards = eval_service.load_cards()["cards"]
        assert len(cards) == 8
        for card in cards:
            assert set(card["answers"]) == set(eval_service.FIELDS14)
            assert all(a["read"] for a in card["answers"].values())
            assert set(card["answers"]["address"]["parts"]) == set(eval_service.PARTS["address"])
            assert "place_of_birth_province" in card["answers"]["place_of_birth"]["parts"]
        assert sum(len(c["answers"]) for c in cards) == 8 * 14 == 112

    def test_perfect_transcripts_give_zero_wer_for_address_and_birthplace_parts(self, db):
        from app.services import eval_service
        s = _eval_session(db, card_no=1)
        card = eval_service.load_cards()["cards"][0]["answers"]
        for key in eval_service.PARTS["address"]:
            part = card["address"]["parts"][key]
            _eval_try(db, s, key, "spoken", part["read"], part["expected"])
        for key in eval_service.PARTS["place_of_birth"]:
            part = card["place_of_birth"]["parts"][key]
            _eval_try(db, s, key, "spoken", part["read"], part["expected"])
        rows = _score(db, s, CARD1)
        for field in ("address", "place_of_birth"):
            assert rows[field].wer_errors == 0 and rows[field].wer_words > 0
            assert rows[field].correct_first and rows[field].correct_ladder and rows[field].correct_finished


class TestEvalWer:
    def test_known_pairs(self):
        from app.services.eval_service import wer_counts
        assert wer_counts("dela cruz", "de la cruz") == (2, 2)     # 1 substitution + 1 insertion over 2 words
        assert wer_counts("My surname is Dela Cruz po.", "my surname is dela cruz po") == (0, 6)
        assert wer_counts("Single.", "") == (1, 1)

    def test_pooled_over_answers_not_averaged(self, db):
        s = _eval_session(db, card_no=1)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
        _eval_try(db, s, "sex", "spoken", "male", "Male")           # 1 error over 1 word ("Female.")
        rows = _score(db, s, CARD1)
        assert (rows["surname"].wer_errors, rows["surname"].wer_words) == (0, 1)
        assert (rows["sex"].wer_errors, rows["sex"].wer_words) == (1, 1)

    def test_not_a_scripted_round_has_no_wer(self, db):
        s = _eval_session(db, round_type="uat")
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
        assert _score(db, s, CARD1)["surname"].wer_words is None


class TestEvalAccuracyNumbers:
    """One fixture per finished_in value: the three numbers (first try / after the ladder / finished)."""

    GOOD = dict(transcript="Abarquez", extracted="Abarquez")
    BAD = dict(transcript="Abarkez", extracted="Abarkez", ok=False)

    def _row(self, db, tries, final="Abarquez", notes=None, field="surname"):
        s = _eval_session(db, card_no=1)
        for t in tries:
            _eval_try(db, s, field, **t)
        return _score(db, s, {**CARD1, field: final}, notes)[field]

    @staticmethod
    def flags(row):
        return row.finished_in, row.correct_first, row.correct_ladder, row.correct_finished

    def test_spoken_1(self, db):
        assert self.flags(self._row(db, [dict(step="spoken", **self.GOOD)])) == ("spoken_1", True, True, True)

    def test_spoken_2(self, db):
        row = self._row(db, [dict(step="spoken", **self.BAD), dict(step="spoken", **self.GOOD)])
        assert self.flags(row) == ("spoken_2", False, True, True)

    def test_spoken_3(self, db):
        tries = [dict(step="spoken", **self.BAD)] * 2 + [dict(step="spoken", **self.GOOD)]
        assert self.flags(self._row(db, tries)) == ("spoken_3", False, True, True)

    def test_spelled_1_and_2(self, db):
        spoken = [dict(step="spoken", **self.BAD)] * 3
        one = self._row(db, spoken + [dict(step="spelled", **self.GOOD)])
        assert self.flags(one) == ("spelled_1", False, True, True)
        two = self._row(db, spoken + [dict(step="spelled", **self.BAD), dict(step="spelled", **self.GOOD)])
        assert self.flags(two) == ("spelled_2", False, True, True)

    def test_end_said_is_not_a_ladder_success(self, db):
        tries = [dict(step="spoken", **self.BAD)] * 3 + [dict(step="spelled", **self.BAD)] * 2
        row = self._row(db, tries + [dict(step="end_said", **self.GOOD)])
        assert self.flags(row) == ("end_said", False, False, True)

    def test_typed_is_never_a_voice_hit(self, db):
        typed = dict(step="end_said", transcript="", extracted="Abarquez", engine="typed")
        notes = [{"field": "surname", "accepted_engine": "typed", "accepted_value": "Abarquez"}]
        assert self.flags(self._row(db, [typed], notes=notes)) == ("end_typed", False, False, True)

    def test_staff_typed_on_the_review_page(self, db):
        row = self._row(db, [dict(step="spoken", transcript="Abarkez", extracted="Abarkez", ok=True)])
        assert self.flags(row) == ("staff_typed", False, False, True)

    def test_blank_and_skipped_are_misses_until_finished(self, db):
        notes = [{"field": "surname", "accepted_engine": "skipped", "skipped": True}]
        row = self._row(db, [dict(step="spoken", **self.BAD)] * 3, final="", notes=notes)
        assert self.flags(row) == ("blank", False, False, False) and row.skipped is True

    def test_a_wrong_final_is_a_miss_in_a_scripted_round(self, db):
        row = self._row(db, [dict(step="spoken", transcript="Reyes", extracted="Reyes")], final="Reyes")
        assert self.flags(row) == ("spoken_1", False, False, False)

    def test_an_accepted_none_is_a_hit_but_an_unanswered_blank_is_a_miss(self, db):
        s = _eval_session(db, card_no=3)
        _eval_try(db, s, "middle_name", "spoken", "None.", "", ok=True)
        _eval_try(db, s, "icr_no", "spoken", "none", "", ok=True)
        rows = _score(db, s, {**CARD1, "middle_name": "", "icr_no": ""})
        assert self.flags(rows["middle_name"]) == ("spoken_1", True, True, True)
        assert self.flags(rows["icr_no"]) == ("spoken_1", True, True, True)
        s2 = _eval_session(db, card_no=3)
        assert self.flags(_score(db, s2, {**CARD1, "middle_name": ""})["middle_name"]) == ("blank", False, False, False)

    def test_fillers_and_silence_are_not_tries(self, db):
        tries = [dict(step="spoken", transcript="Um.", extracted="", ok=False),
                 dict(step="spoken", transcript="", extracted=None, ok=False),
                 dict(step="spoken", **self.GOOD)]
        assert self.flags(self._row(db, tries)) == ("spoken_1", True, True, True)

    def test_non_scripted_rounds_compare_with_the_final_value(self, db):
        s = _eval_session(db, round_type="uat")
        _eval_try(db, s, "surname", "spoken", "Reyez", "Reyez", ok=True)
        _eval_try(db, s, "surname", "spoken", "Reyes", "Reyes", ok=True)
        row = _score(db, s, {**CARD1, "surname": "Reyes"})["surname"]
        assert self.flags(row) == ("spoken_2", False, True, True)

    def test_dates_and_numbers_compare_in_the_saved_form(self, db):
        s = _eval_session(db, card_no=1)
        _eval_try(db, s, "date", "spoken", "March 14, 1990.", "1990-03-14")
        _eval_try(db, s, "height", "spoken", "158 centimeters.", "158")
        _eval_try(db, s, "gross_annual_income", "spoken", "180,000 pesos.", "180000")
        rows = _score(db, s, {**CARD1, "height": "158.0"})
        assert rows["date"].correct_first and rows["height"].correct_first
        assert rows["gross_annual_income"].correct_first

    def test_model_and_seconds_are_kept_on_the_final_row(self, db):
        s = _eval_session(db, card_no=1)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez", model="tiny")
        row = _score(db, s, CARD1, [{"field": "surname", "seconds": 4.2}])["surname"]
        assert row.model == "tiny" and row.seconds == 4.2


class TestEvalEndToEnd:
    FAKE = {"transcript": "Abarquez", "dropped": None, "language": "en", "confidence": 0.9, "segments": [],
            "processing_time_seconds": 1.5,
            "passes": [{"model": "medium", "beam": 5, "seconds": 1.4, "second_pass": False, "text": "Abarquez"}]}

    def test_a_scripted_interview_is_logged_and_scored(self, client, staff_token, admin_token, db):
        from app.models.eval import EvalAnswer, EvalSession
        _eval_put(client, admin_token, evaluation_mode=True, round_type="scripted", round_condition="quiet",
                  card_set="all", new_round=True)
        sid = client.post("/api/eval/sessions", json={"input_method": "voice", "device": "Test PC"},
                          headers=_auth(staff_token)).json()["session_id"]
        with patch.object(SpeechService, "transcribe_bytes", return_value=dict(self.FAKE)):
            resp = client.post(
                "/api/speech/transcribe", headers=_auth(staff_token),
                files={"audio_file": ("answer.wav", b"RIFFwav", "audio/wav"),
                       "eval_clip": ("answer.webm", b"webm-bytes", "audio/webm")},
                data={"field_key": "surname", "eval_session_id": str(sid), "eval_field": "surname",
                      "eval_step": "spoken", "eval_try": "t1"})
        assert resp.status_code == 200 and resp.json()["transcript"] == "Abarquez"
        assert "passes" not in resp.json()
        ext = client.post("/api/extraction/answer", headers=_auth(staff_token), json={
            "field_key": "surname", "transcript": "Abarquez", "spoken": True,
            "eval_session_id": sid, "eval_step": "spoken", "eval_try": "t1"})
        assert ext.status_code == 200
        db.expire_all()
        rows = db.query(EvalAnswer).filter(EvalAnswer.session_id == sid).all()
        assert len(rows) == 1               # one row per try: transcribe and extraction joined on the try id
        row = rows[0]
        assert (row.model, row.beam, row.engine, row.transcript, row.whisper_seconds) == (
            "medium", 5, "whisper", "Abarquez", 1.5)
        assert row.extracted_value is not None and row.extract_ok is True
        body = {"applicant_data": {**TestIssuance.SAMPLE_APPLICANT, "surname": "Abarquez"}, "input_method": "voice",
                "eval_session_id": sid, "eval_batch": {"notes": [{"field": "surname", "seconds": 3.0}], "errors": 2}}
        with patch("app.api.routes.forms.generate_cedula_pdf", return_value="pdfs/test.pdf"), \
             patch("app.api.routes.forms.generate_receipt_pdf", return_value="pdfs/test_receipt.pdf"):
            issued = client.post("/api/forms/issue", json=body, headers=_auth(staff_token))
        assert issued.status_code == 201
        db.expire_all()
        session = db.get(EvalSession, sid)
        assert session.outcome == "issued" and session.scored and session.error_count == 2
        assert session.app_seconds is not None
        final = db.query(EvalAnswer).filter(EvalAnswer.session_id == sid, EvalAnswer.step == "final",
                                            EvalAnswer.field == "surname").one()
        assert final.correct_first and final.finished_in == "spoken_1" and final.seconds == 3.0
        stats = client.get("/api/eval/stats?round_type=scripted", headers=_auth(admin_token)).json()
        assert stats["sessions"]["issued"] >= 1 and stats["fields"]["surname"]["answers"] >= 1
        for kind in ("sessions", "answers", "summary"):
            csv_resp = client.get(f"/api/eval/export/{kind}.csv", headers=_auth(admin_token))
            assert csv_resp.status_code == 200 and csv_resp.headers["content-type"].startswith("text/csv")
        detail = client.get(f"/api/eval/sessions/{sid}", headers=_auth(admin_token)).json()
        assert detail["participant_code"].startswith("P") and "Reyes" not in str(detail)
        _eval_put(client, admin_token, round_type="practice", new_round=True)

    def test_a_silent_try_and_both_review_passes_are_logged(self, client, staff_token, admin_token, db):
        from app.models.eval import EvalAnswer
        _eval_put(client, admin_token, evaluation_mode=True, round_type="practice")
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        empty = {"transcript": "", "dropped": None, "language": "en", "confidence": 0.0, "segments": [],
                 "processing_time_seconds": 0.5, "passes": []}
        with patch.object(SpeechService, "transcribe_bytes", return_value=empty):
            resp = client.post("/api/speech/transcribe", headers=_auth(staff_token),
                               files={"audio_file": ("a.wav", b"RIFFwav", "audio/wav")},
                               data={"field_key": "surname", "eval_session_id": str(sid), "eval_field": "surname",
                                     "eval_step": "spoken", "eval_try": "silent1"})
        assert resp.status_code == 422
        review = {"transcript": "no it is cruz", "dropped": None, "language": "en", "confidence": 0.9, "segments": [],
                  "processing_time_seconds": 2.0, "passes": [
                      {"model": "tiny", "beam": 5, "seconds": 0.3, "second_pass": False, "text": "no"},
                      {"model": "medium", "beam": 5, "seconds": 1.7, "second_pass": True, "text": "no it is cruz"}]}
        with patch.object(SpeechService, "transcribe_bytes", return_value=review):
            client.post("/api/speech/transcribe", headers=_auth(staff_token),
                        files={"audio_file": ("a.wav", b"RIFFwav", "audio/wav")},
                        data={"field_key": "review", "eval_session_id": str(sid), "eval_field": "surname",
                              "eval_step": "review", "eval_try": "rev1"})
        db.expire_all()
        rows = {r.try_id: r for r in db.query(EvalAnswer).filter(EvalAnswer.session_id == sid)}
        assert rows["silent1"].transcript == "" and rows["silent1"].audio_ref is None
        assert rows["rev1"].model == "tiny>medium" and rows["rev1"].yes_no == "no"

    def test_failed_logins_and_denied_calls_are_counted(self, client, staff_token, admin_token):
        before = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["security"]
        client.post("/api/auth/login", json={"username": "nobody", "password": "wrong-password"})
        client.get("/api/eval/stats", headers=_auth(staff_token))
        after = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["security"]
        assert after["login_failed"] == before["login_failed"] + 1
        assert after["forbidden"] == before["forbidden"] + 1


class TestEvalAudio:
    """Clips are saved only with evaluation mode on, a Whisper answer and a voice session."""

    def _log(self, db, input_method="voice", engine="whisper", mode_on=True, try_id="c1", clip=b"clip"):
        from app.models.eval import EvalAnswer
        from app.services import eval_service, storage_service
        s = _eval_session(db)
        s.input_method = input_method
        db.commit()
        uploads = []

        def upload(bucket, path, data, content_type):
            uploads.append((bucket, path, content_type))
            return f"{bucket}/{path}"

        with patch.object(eval_service, "recording_on", return_value=mode_on), \
             patch.object(storage_service, "is_enabled", return_value=True), \
             patch.object(storage_service, "upload_bytes", side_effect=upload):
            eval_service.log_transcribe(s.id, s.staff_user_id, field="surname", step="spoken", try_id=try_id,
                                        engine=engine, model="medium", beam=5, transcript="abarquez", seconds=1.0,
                                        clip=clip, clip_type="audio/webm")
        db.expire_all()
        return uploads, db.query(EvalAnswer).filter(EvalAnswer.session_id == s.id).one()

    def test_saved_when_on(self, db):
        uploads, row = self._log(db, try_id="on1")
        assert len(uploads) == 1 and uploads[0][0] == "eval-audio" and uploads[0][2] == "audio/webm"
        assert row.audio_ref and row.audio_ref.startswith("eval-audio/")

    def test_not_saved_when_off_browser_engine_typed_or_qr(self, db):
        for kwargs in (dict(mode_on=False, try_id="off1"), dict(engine="browser", try_id="br1"),
                       dict(engine="typed", try_id="ty1"), dict(input_method="qr", try_id="qr1")):
            uploads, row = self._log(db, **kwargs)
            assert uploads == [] and row.audio_ref is None, kwargs

    def test_a_session_started_as_manual_becomes_voice_on_the_first_whisper_try(self, db):
        """Typed input never calls Whisper, so a manual session has no clips; the first Whisper try makes it voice."""
        from app.models.eval import EvalSession
        uploads, row = self._log(db, input_method="manual", try_id="man1")
        assert len(uploads) == 1
        assert db.get(EvalSession, row.session_id).input_method == "voice"

    def test_route_sends_the_compressed_clip_and_falls_back_to_the_wav(self, client, staff_token, admin_token):
        from app.services import eval_service
        _eval_put(client, admin_token, evaluation_mode=True, round_type="practice")
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        fake = {"transcript": "hi", "dropped": None, "language": "en", "confidence": 0.9, "segments": [],
                "processing_time_seconds": 0.5, "passes": []}
        with patch.object(SpeechService, "transcribe_bytes", return_value=fake), \
             patch.object(eval_service, "log_transcribe") as logged:
            data = {"field_key": "surname", "eval_session_id": str(sid), "eval_field": "surname",
                    "eval_step": "spoken", "eval_try": "x1"}
            client.post("/api/speech/transcribe", headers=_auth(staff_token), data=data,
                        files={"audio_file": ("a.wav", b"WAVBYTES", "audio/wav"),
                               "eval_clip": ("a.webm", b"WEBMBYTES", "audio/webm")})
            client.post("/api/speech/transcribe", headers=_auth(staff_token), data={**data, "eval_try": "x2"},
                        files={"audio_file": ("a.wav", b"WAVBYTES", "audio/wav")})
        first, second = (c.kwargs for c in logged.call_args_list)
        assert first["clip"] == b"WEBMBYTES" and first["clip_type"] == "audio/webm"
        assert second["clip"] == b"WAVBYTES" and second["clip_type"] == "audio/wav"

    def test_the_logging_runs_after_the_response_is_built(self):
        """The route hands the log to a background task; it never calls it while building the answer."""
        import inspect
        from app.api.routes import speech
        source = inspect.getsource(speech.transcribe_audio)
        assert "background_tasks.add_task(eval_service.log_transcribe" in source

    def test_playback_goes_through_the_backend_for_admins(self, client, admin_token, staff_token, db):
        from app.services import storage_service
        s = _eval_session(db)
        row = _eval_try(db, s, "surname", "spoken", "x", "x")
        row.audio_ref = "eval-audio/s1/clip.webm"
        db.commit()
        with patch.object(storage_service, "download_bytes", return_value=(b"audio", "audio/webm")):
            assert client.get(f"/api/eval/answers/{row.id}/audio", headers=_auth(admin_token)).content == b"audio"
            assert client.get(f"/api/eval/answers/{row.id}/audio", headers=_auth(staff_token)).status_code == 403


class TestEvalStatsAndExport:
    def test_stats_query_count_stays_constant(self, client, admin_token, db):
        from sqlalchemy import event
        from app.core.database import engine as app_engine
        from app.models.eval import EvalAnswer, EvalSession
        db.query(EvalAnswer).delete()
        db.query(EvalSession).delete()
        db.commit()

        def count_queries():
            seen = []

            def listener(conn, cursor, statement, params, context, executemany):
                seen.append(statement)

            event.listen(app_engine, "before_cursor_execute", listener)
            try:
                assert client.get("/api/eval/stats?round_type=scripted", headers=_auth(admin_token)).status_code == 200
            finally:
                event.remove(app_engine, "before_cursor_execute", listener)
            return len(seen)

        def add_sessions(n):
            for _ in range(n):
                s = _eval_session(db, outcome="issued", round_no=7)
                _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
                _score(db, s, CARD1)

        add_sessions(2)
        small = count_queries()
        add_sessions(8)
        assert count_queries() == small
        data = client.get("/api/eval/stats?round_type=scripted", headers=_auth(admin_token)).json()
        assert data["sessions"]["issued"] == 10 and data["overall"]["answers"] == 10 * 14

    def test_headline_hard5_and_easy9(self, client, admin_token):
        data = client.get("/api/eval/stats?round_type=scripted", headers=_auth(admin_token)).json()
        assert data["hard5"]["answers"] == 10 * 5 and data["easy9"]["answers"] == 10 * 9
        assert data["overall"]["answers"] == data["hard5"]["answers"] + data["easy9"]["answers"]

    def test_practice_and_live_are_left_out_by_default(self, client, admin_token, db):
        s = _eval_session(db, round_type="live", outcome="issued", round_no=8)
        _eval_try(db, s, "surname", "spoken", "Reyes", "Reyes")
        _score(db, s, {**CARD1, "surname": "Reyes"})
        default = client.get("/api/eval/stats", headers=_auth(admin_token)).json()
        live = client.get("/api/eval/stats?round_type=live", headers=_auth(admin_token)).json()
        assert default["overall"]["answers"] == 10 * 14 and live["overall"]["answers"] == 14

    def test_csv_is_streamed_in_pages(self, client, admin_token, db):
        import inspect
        from app.api.routes import eval as eval_routes
        assert inspect.isgenerator(eval_routes._answer_lines())
        assert inspect.isgenerator(eval_routes._session_lines())
        with patch.object(eval_routes, "CSV_PAGE", 3):      # several pages: every row still comes out, once
            text = client.get("/api/eval/export/answers.csv", headers=_auth(admin_token)).text
        from app.models.eval import EvalAnswer
        assert len(text.strip().splitlines()) - 1 >= db.query(EvalAnswer).count()

    def test_long_names_survive_the_csv(self, client, admin_token, db):
        s = _eval_session(db, outcome="issued", round_no=9)
        name = "Ma. Concepcion Dela Cruz-Villanueva"
        _eval_try(db, s, "surname", "spoken", name, name)
        assert name in client.get("/api/eval/export/answers.csv", headers=_auth(admin_token)).text


class TestEvalDeleteNow:
    def test_nothing_is_deleted_without_a_click_and_numbers_stay(self, client, admin_token, db):
        from app.models.eval import EvalAnswer, EvalSession
        from app.services import eval_service, storage_service
        db.query(EvalAnswer).delete()
        db.query(EvalSession).delete()
        db.commit()
        s = _eval_session(db, outcome="issued", round_no=10)
        for i in range(5):
            row = _eval_try(db, s, "surname", "spoken", f"heard {i}", "Abarquez")
            row.audio_ref = f"eval-audio/s{s.id}/{i}.webm"
        db.commit()
        _score(db, s, CARD1)
        before = eval_service.pending_delete_count(db)
        assert before > 5
        assert client.get("/api/eval/delete-audio", headers=_auth(admin_token)).json()["remaining"] == before
        assert client.post("/api/eval/delete-audio", json={"size": 100}, headers=_auth(admin_token)).status_code == 400
        assert eval_service.pending_delete_count(db) == before              # no click, nothing removed
        deleted = []
        with patch.object(storage_service, "delete_object", side_effect=deleted.append):
            first = client.post("/api/eval/delete-audio", json={"confirm": True, "size": 4},
                                headers=_auth(admin_token)).json()
            assert first["rows"] == 4 and first["remaining"] == before - 4   # a batch, not everything
            while client.post("/api/eval/delete-audio", json={"confirm": True, "size": 4},
                              headers=_auth(admin_token)).json()["remaining"]:
                pass
        assert len(deleted) == 5
        db.expire_all()
        rows = db.query(EvalAnswer).filter(EvalAnswer.session_id == s.id).all()
        assert all(r.audio_ref is None and r.transcript is None and r.final_value is None for r in rows)
        final = next(r for r in rows if r.field == "surname" and r.step == "final")
        assert final.correct_first is True and final.wer_words is not None   # the numbers stay


class TestEvalStartupAndReview:
    def test_start_up_refuses_a_database_without_the_migration(self):
        from sqlalchemy import create_engine as make_engine, text
        from app.core.database import MISSING_EVAL_MIGRATION, missing_eval_message
        old = make_engine("sqlite://")
        with old.begin() as conn:
            conn.execute(text("create table app_settings (id integer primary key, barangay_name text)"))
        assert missing_eval_message(old) == MISSING_EVAL_MIGRATION
        assert missing_eval_message(make_engine("sqlite://")) is None        # empty database (tests)

    def test_the_evaluation_tables_are_not_built_by_create_all_at_start_up(self):
        from app.models.eval import EVAL_TABLE_NAMES
        assert all(Base.metadata.tables[n].info.get("eval") for n in EVAL_TABLE_NAMES)

    def test_review_passes_are_reported_without_changing_decoding(self):
        import numpy as np
        from app.services import speech_service
        audio = np.zeros(1600, dtype=np.float32)

        def run(texts, **patches):
            queue = iter(texts)
            with patch.object(SpeechService, "get_model", return_value=object()), \
                 patch.object(speech_service, "_run",
                              side_effect=lambda *a, **k: {"text": next(queue), "segments": []}):
                return SpeechService._decode(audio, "review", None)

        plain = run(["yes"])
        assert [p["model"] for p in plain["eval_passes"]] == ["tiny"]
        corrected = run(["no, it is cruz", "no, my surname is cruz"])
        assert [(p["model"], p["second_pass"]) for p in corrected["eval_passes"]] == [
            ("tiny", False), ("medium", True)]
        assert corrected["text"] == "no, my surname is cruz"
        with patch.object(speech_service.settings, "WHISPER_FIELD_MODELS", False):
            off = run(["no, it is cruz"])
        assert [p["model"] for p in off["eval_passes"]] == [speech_service.settings.WHISPER_MODEL]
