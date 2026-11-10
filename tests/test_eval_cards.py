"""
tests/test_eval_cards.py
Plan 009: generated cards, pooled WER with S / D / I, number-lenient text rules, field extraction accuracy
(a skip is a miss), counted sessions, seconds per question and hand re-scoring. Shares the throwaway SQLite
database and the fixtures of test_backend.py and the helpers of test_eval_mode.py. Fake people only.
"""
# ruff: noqa: F811  (pytest fixtures imported from test_backend are re-used as test arguments)
import json
import random
import re

import pytest  # noqa: F401  (fixtures are imported below)

from tests.test_backend import (  # noqa: F401
    TestIssuance, TestingSessionLocal, _auth, admin_token, admin_user, client, db, setup_test_db, staff_token,
    staff_user,
)
from tests.test_eval_mode import CARD1, _eval_put, _eval_try, _score


@pytest.fixture(scope="module", autouse=True)
def release_connections(setup_test_db):
    yield
    from app.core.database import engine as app_engine
    app_engine.dispose()


def _clear_cards(db):
    """Every test here starts with no card waiting (cards are shared by the whole throwaway database)."""
    from app.models.eval import EvalCard, EvalSession
    db.query(EvalSession).filter(EvalSession.card_id.isnot(None)).update({EvalSession.card_id: None})
    db.query(EvalCard).delete()
    db.commit()


def _wipe(db):
    """No sessions, answers or cards: for tests that check totals."""
    from app.models.eval import EvalAnswer, EvalSession
    _clear_cards(db)
    db.query(EvalAnswer).delete()
    db.query(EvalSession).delete()
    db.commit()


def _card_row(db, answers=None):
    """A stored card (by default the fixed card 1's words), not yet claimed."""
    from app.models.eval import EvalCard
    from app.services import eval_service
    answers = answers or json.loads(json.dumps(eval_service.load_cards()["cards"][0]["answers"]))
    card = EvalCard(style="A", card=json.dumps({"style": "A", "answers": answers}))
    db.add(card)
    db.commit()
    return card


def _card_session(db, answers=None, round_type="uat", outcome="issued", counts=True, with_card=True, round_no=909):
    """A session that read a generated card (plan 009)."""
    from app.models.eval import EvalSession
    number = db.query(EvalSession).filter(EvalSession.round_no == round_no).count() + 1
    session = EvalSession(round_no=round_no, round_type=round_type, round_condition="quiet", participant_no=number,
                          participant_code=f"P{number:03d}", input_method="voice", outcome=outcome, counts=counts)
    db.add(session)
    db.commit()
    if with_card:
        card = _card_row(db, answers)
        card.claimed_by_session_id = session.id
        session.card_id = card.id
        db.commit()
    return session


def _with_read(field, read, expected=None):
    from app.services import eval_service
    answers = json.loads(json.dumps(eval_service.load_cards()["cards"][0]["answers"]))
    answers[field] = {"expected": expected if expected is not None else answers[field]["expected"], "read": read}
    return answers


class TestGeneratedCards:
    READ_CHARS = re.compile(r"^[A-Za-zÑñ0-9 ,.'\-]+$")

    def test_a_card_covers_the_14_questions_and_the_5_steps(self):
        from app.services import eval_card_pdf, eval_cards, eval_service
        rng = random.Random(9)
        for i in range(25):
            card = eval_cards.generate_card(eval_cards.style_for(i), rng)
            answers = card["answers"]
            assert set(answers) == set(eval_service.FIELDS14)
            assert set(answers["address"]["parts"]) == set(eval_service.PARTS["address"])
            assert set(answers["place_of_birth"]["parts"]) == set(eval_service.PARTS["place_of_birth"])
            lines = eval_card_pdf.card_lines(answers)
            assert len(lines) == 12 + 5          # 12 one-step questions + 3 address steps + 2 birthplace steps
            assert [line["key"] for line in lines][:4] == ["surname", "given_name", "middle_name", "address_town"]
            for line in lines:
                assert self.READ_CHARS.match(line["read"]), line

    def test_two_clicks_give_two_different_cards_and_styles_cycle(self, client, admin_token, db):
        from app.models.eval import EvalCard
        _clear_cards(db)
        client.post("/api/eval/cards", headers=_auth(admin_token))
        client.post("/api/eval/cards", headers=_auth(admin_token))
        db.expire_all()
        a, b = db.query(EvalCard).order_by(EvalCard.id).all()
        assert json.loads(a.card)["answers"] != json.loads(b.card)["answers"]
        assert (a.style, b.style) == ("A", "B")
        _clear_cards(db)

    def test_cards_come_from_the_fixed_lists_never_from_the_database(self, db):
        """A resident saved in the database never reaches a card: the generator has no database at all."""
        import inspect
        from app.models.applicant import ApplicantRecord
        from app.services import eval_cards
        assert "db" not in inspect.signature(eval_cards.generate_card).parameters
        db.add(ApplicantRecord(surname="Zyxwresident", given_name="Qwertyresident", middle_name="Asdfresident",
                               address="Purok 1, Barangay Imbatug, Baungon, Bukidnon"))
        db.commit()
        rng = random.Random(3)
        for i in range(40):
            answers = eval_cards.generate_card(eval_cards.style_for(i), rng)["answers"]
            text = json.dumps(answers)
            assert "resident" not in text.lower()
            assert answers["surname"]["expected"] in eval_cards.surnames()
            assert answers["address"]["parts"]["address_barangay"]["expected"] in eval_cards.baungon_barangays()
            assert answers["place_of_birth"]["parts"]["place_of_birth_province"]["expected"] == "Bukidnon"

    def test_every_generated_place_is_what_the_real_extraction_returns(self):
        """Each barangay and Bukidnon town a card can name is understood as the card's expected value."""
        from app.core.database import engine as app_engine
        from app.services import eval_cards, eval_service
        from app.services.answer_service import understand_answer
        # The interview passes the step before's code (the town, the province), as the web does.
        town = understand_answer("address_town", "Baungon", spoken=True).normalized
        province = understand_answer("place_of_birth_province", "Bukidnon", spoken=True).normalized
        for barangay in eval_cards.baungon_barangays():
            result = understand_answer("address_barangay", barangay, spoken=True, town=town)
            assert result.ok and eval_service.norm("address_barangay", result.value) == eval_service.norm(
                "address_barangay", barangay), (barangay, result.value)
        for name in eval_cards.bukidnon_towns():
            result = understand_answer("place_of_birth_town", name, spoken=True, province=province)
            assert result.ok and eval_service.norm("place_of_birth_town", result.value) == eval_service.norm(
                "place_of_birth_town", f"{name}, Bukidnon"), (name, result.value)
        app_engine.dispose()

    def test_the_card_pdf_is_admin_only_and_a_missing_card_is_404(self, client, admin_token, staff_token, db):
        _clear_cards(db)
        card = client.post("/api/eval/cards", headers=_auth(admin_token)).json()["next"]
        pdf = client.get(f"/api/eval/cards/{card['id']}.pdf", headers=_auth(admin_token))
        assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
        assert pdf.content.startswith(b"%PDF")
        assert client.get("/api/eval/cards/999999.pdf", headers=_auth(admin_token)).status_code == 404
        assert client.get(f"/api/eval/cards/{card['id']}.pdf", headers=_auth(staff_token)).status_code == 403
        assert client.get(f"/api/eval/cards/{card['id']}.pdf").status_code == 401
        assert "Zyxwresident" not in json.dumps(card)
        _clear_cards(db)


class TestCardClaims:
    def test_the_card_shown_is_the_one_the_next_session_claims(self, client, admin_token, staff_token, db):
        """Codex plan round 1, #2: generate A then B; the page shows A; the session reads A; then B is shown."""
        from app.models.eval import EvalSession
        _clear_cards(db)
        _eval_put(client, admin_token, evaluation_mode=True, round_type="uat", counts=True, new_round=True)
        first = client.post("/api/eval/cards", headers=_auth(admin_token)).json()
        a = first["next"]["id"]
        second = client.post("/api/eval/cards", headers=_auth(admin_token)).json()
        assert second["next"]["id"] == a and second["waiting"] == 2           # B waits behind A
        b = max(c["id"] for c in second["recent"])
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        check = TestingSessionLocal()
        assert check.get(EvalSession, sid).card_id == a
        check.close()
        after = client.get("/api/eval/cards", headers=_auth(admin_token)).json()
        assert after["next"]["id"] == b and after["waiting"] == 1
        claimed = client.get(f"/api/eval/cards/{a}", headers=_auth(admin_token)).json()
        assert claimed["claimed_by"].startswith("P")
        _eval_put(client, admin_token, round_type="practice", new_round=True)
        _clear_cards(db)

    def test_no_card_is_given_twice_and_none_left_means_no_card(self, client, admin_token, staff_token, db):
        from app.models.eval import EvalSession
        _clear_cards(db)
        _eval_put(client, admin_token, evaluation_mode=True, round_type="uat", new_round=True)
        client.post("/api/eval/cards", headers=_auth(admin_token))
        ids = [client.post("/api/eval/sessions", json={"input_method": "voice"},
                           headers=_auth(staff_token)).json()["session_id"] for _ in range(2)]
        check = TestingSessionLocal()
        cards = [check.get(EvalSession, i).card_id for i in ids]
        check.close()
        assert cards[0] is not None and cards[1] is None
        listed = client.get("/api/eval/sessions?view=all", headers=_auth(admin_token)).json()
        no_card = next(s for s in listed if s["id"] == ids[1])
        assert no_card["card_id"] is None and no_card["counted"] is False
        _eval_put(client, admin_token, round_type="practice", new_round=True)
        _clear_cards(db)

    def test_a_session_left_without_a_whisper_try_gives_its_card_back(self, client, admin_token, staff_token, db):
        from app.models.eval import EvalCard
        _clear_cards(db)
        _eval_put(client, admin_token, evaluation_mode=True, round_type="uat", new_round=True)
        card_id = client.post("/api/eval/cards", headers=_auth(admin_token)).json()["next"]["id"]
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        client.post(f"/api/eval/sessions/{sid}/finish", json={}, headers=_auth(staff_token))
        db.expire_all()
        assert db.get(EvalCard, card_id).claimed_by_session_id is None
        assert client.get("/api/eval/cards", headers=_auth(admin_token)).json()["next"]["id"] == card_id
        _eval_put(client, admin_token, round_type="practice", new_round=True)
        _clear_cards(db)

    def test_a_no_card_session_has_no_wer(self, db):
        s = _card_session(db, with_card=False)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
        row = _score(db, s, CARD1)["surname"]
        assert row.wer_words is None and row.expected_value is None


class TestWerSplitAndText:
    def test_hand_counted_substitution_insertion_and_deletion(self):
        from app.services.eval_service import wer_split
        assert wer_split("maria luisa tumampos", "maria luisa tumpos") == (1, 0, 0, 3)
        assert wer_split("maria luisa tumampos", "maria luisa tumampos po") == (0, 0, 1, 3)
        assert wer_split("maria luisa tumampos", "maria tumampos") == (0, 1, 0, 3)

    def test_numbers_are_lenient_but_names_are_not(self):
        from app.services.eval_service import wer_counts, wer_words
        assert wer_counts("fifty five", "55") == (0, 1)
        assert wer_counts("55 kilos.", "fifty-five kilos") == (0, 2)
        assert wer_counts("March 14, 1990.", "march 14th 1990") == (0, 3)
        assert wer_counts("March 14, 1990.", "March fourteenth, nineteen ninety") == (0, 3)
        assert wer_counts("180,000 pesos.", "one hundred eighty thousand pesos") == (0, 2)
        assert wer_words("one hundred twenty thousand") == ["120000"]
        assert wer_counts("Tumampos.", "Tumampus") == (1, 1)                 # a misheard name is an error
        assert wer_counts("Akong apelyido kay Jumawan.", "akong apelyido kay jumawan") == (0, 4)

    def test_the_split_is_stored_and_adds_up_to_the_errors(self, db):
        s = _card_session(db, answers=_with_read("surname", "maria luisa dela cruz"))
        _eval_try(db, s, "surname", "spoken", "Luisa Dela Crus po", "Abarquez")     # D maria, S crus, I po
        row = _score(db, s, CARD1)["surname"]
        assert (row.s_errors, row.d_errors, row.i_errors, row.wer_words) == (1, 1, 1, 4)
        assert row.wer_errors == row.s_errors + row.d_errors + row.i_errors


class TestPooledWer:
    def test_pooled_over_sessions_not_an_average_of_percentages(self, client, admin_token, db):
        _wipe(db)
        ten = " ".join(f"w{i}" for i in range(10))
        forty = " ".join(f"w{i}" for i in range(40))
        a = _card_session(db, answers=_with_read("surname", ten))
        _eval_try(db, a, "surname", "spoken", " ".join(f"w{i}" for i in range(8)) + " x y", "Abarquez")
        b = _card_session(db, answers=_with_read("surname", forty))
        _eval_try(db, b, "surname", "spoken", forty, "Abarquez")
        _score(db, a, CARD1)
        _score(db, b, CARD1)
        data = client.get("/api/eval/stats", headers=_auth(admin_token)).json()
        wer = data["wer"]
        assert (wer["errors"], wer["words"], wer["participants"]) == (2, 50, 2)
        assert wer["pct"] == 4.0                                               # not (20% + 0%) / 2 = 10%
        assert (wer["substitutions"], wer["deletions"], wer["insertions"]) == (2, 0, 0)
        assert data["targets"] == {"wer_pct": 15.0, "field_extraction_pct": 90.0}
        assert data["progress"] == {"counted": 2, "target": 30}

    def test_only_the_first_spoken_try_counts(self, db):
        """A second spoken try, a spelled try or a skip add nothing to WER; the spelled hit is extracted."""
        answers = _with_read("surname", "maria luisa tumampos", expected="Abarquez")
        spelled = _card_session(db, answers=answers)
        _eval_try(db, spelled, "surname", "spoken", "maria luisa tumpos", "Abarkez", ok=False)
        _eval_try(db, spelled, "surname", "spoken", "maria luisa tumampos", "Abarkez", ok=False)
        _eval_try(db, spelled, "surname", "spelled", "A B A R Q U E Z", "Abarquez")
        row = _score(db, spelled, CARD1)["surname"]
        assert (row.wer_errors, row.wer_words) == (1, 3)
        assert (row.correct_first, row.correct_ladder) == (False, True)
        skipped = _card_session(db, answers=answers)
        _eval_try(db, skipped, "surname", "spoken", "maria luisa tumpos", "Abarkez", ok=False)
        _eval_try(db, skipped, "surname", "spoken", "maria luisa tumampos", "Abarkez", ok=False)
        notes = [{"field": "surname", "accepted_engine": "skipped", "skipped": True}]
        row = _score(db, skipped, {**CARD1, "surname": ""}, notes)["surname"]
        assert (row.wer_errors, row.wer_words) == (1, 3)
        assert (row.correct_first, row.correct_ladder, row.skipped) == (False, False, True)

    def test_a_typed_or_browser_first_try_adds_no_words(self, db):
        s = _card_session(db)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez", engine="browser", model=None)
        _eval_try(db, s, "sex", "spoken", "", "Female", engine="typed", model=None)
        rows = _score(db, s, CARD1)
        assert rows["surname"].wer_words is None and rows["sex"].wer_words is None


class TestSkipsAreMisses:
    """Codex plan round 1, #1."""

    def test_a_recovered_skip_is_a_miss_even_after_a_matching_voice_try(self, db):
        s = _card_session(db)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez", ok=True)     # refused at the read-back
        notes = [{"field": "surname", "accepted_engine": "typed", "accepted_value": "Abarquez", "skipped": True}]
        row = _score(db, s, CARD1, notes)["surname"]
        assert (row.correct_first, row.correct_ladder, row.skipped) == (False, False, True)
        assert (row.finished_in, row.correct_finished) == ("end_typed", True)

    def test_one_skipped_address_part_makes_the_address_a_miss(self, db):
        from app.services import eval_service
        s = _card_session(db)
        card = eval_service.load_cards()["cards"][0]["answers"]["address"]["parts"]
        for key in eval_service.PARTS["address"]:
            _eval_try(db, s, key, "spoken", card[key]["read"], card[key]["expected"])
        notes = [{"field": "address_purok", "accepted_engine": "typed", "accepted_value": "Purok 2", "skipped": True}]
        rows = _score(db, s, CARD1, notes)
        assert rows["address_purok"].correct_ladder is False
        assert (rows["address"].correct_first, rows["address"].correct_ladder, rows["address"].skipped) == (
            False, False, True)
        assert rows["address"].correct_finished is True


class TestCountedSessions:
    def test_only_finished_ticked_card_non_practice_sessions_count(self, client, admin_token, db):
        before = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["progress"]["counted"]
        _card_session(db, counts=False)                         # "counts toward evaluation" unticked
        _card_session(db, round_type="practice")                # a Practice round
        _card_session(db, outcome="open")                       # unfinished
        _card_session(db, outcome="abandoned")
        _card_session(db, with_card=False)                      # no card
        after = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["progress"]["counted"]
        assert after == before
        _card_session(db)
        assert client.get("/api/eval/stats", headers=_auth(admin_token)).json()["progress"]["counted"] == before + 1

    def test_the_counts_box_is_read_by_the_next_session_and_starts_no_new_round(self, client, admin_token,
                                                                                staff_token):
        from app.models.eval import EvalSession
        cfg = _eval_put(client, admin_token, evaluation_mode=True, round_type="uat", new_round=True)
        off = _eval_put(client, admin_token, counts=False)
        assert off["counts"] is False and off["round_no"] == cfg["round_no"]
        sid = client.post("/api/eval/sessions", json={"input_method": "voice"},
                          headers=_auth(staff_token)).json()["session_id"]
        check = TestingSessionLocal()
        assert check.get(EvalSession, sid).counts is False
        check.close()
        _eval_put(client, admin_token, counts=True, round_type="practice", new_round=True)

    def test_toggle_off_logs_nothing_and_keeps_past_results(self, client, admin_token, staff_token):
        _eval_put(client, admin_token, evaluation_mode=False)
        started = client.post("/api/eval/sessions", json={"input_method": "voice"}, headers=_auth(staff_token))
        assert started.json() == {"session_id": None}
        stats = client.get("/api/eval/stats?view=all", headers=_auth(admin_token))
        assert stats.status_code == 200 and stats.json()["sessions"]["total"] > 0
        _eval_put(client, admin_token, evaluation_mode=True)


class TestSecondsPerQuestion:
    def test_mean_seconds_leave_skipped_questions_out(self, client, admin_token, db):
        _wipe(db)
        s = _card_session(db)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
        _eval_try(db, s, "sex", "spoken", "Female", "Female")
        notes = [{"field": "surname", "seconds": 4.0}, {"field": "sex", "seconds": 6.0},
                 {"field": "height", "accepted_engine": "skipped", "skipped": True, "seconds": 40.0}]
        _score(db, s, CARD1, notes)
        timing = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["timing"]
        assert timing["avg_seconds_per_question"] == 5.0                # (4 + 6) / 2, the skip left out
        assert (timing["skipped_questions"], timing["avg_seconds_skipped"]) == (1, 40.0)
        fields = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["fields"]
        assert fields["surname"]["avg_seconds"] == 4.0 and fields["height"]["avg_seconds"] is None


class TestHandScore:
    def _final(self, db, session, field):
        from app.models.eval import EvalAnswer
        return db.query(EvalAnswer).filter(EvalAnswer.session_id == session.id, EvalAnswer.step == "final",
                                           EvalAnswer.field == field).one()

    def test_a_hand_score_is_shown_beside_whisper_and_survives_scoring_again(self, client, admin_token, db):
        _wipe(db)
        s = _card_session(db, answers=_with_read("surname", "maria luisa tumampos"))
        _eval_try(db, s, "surname", "spoken", "maria luisa tumpos", "Abarquez")
        _score(db, s, CARD1)
        row = self._final(db, s, "surname")
        # The clip shows the participant really said "maria tumampos": Whisper's "luisa tumpos" -> 1 S + 1 I.
        saved = client.put(f"/api/eval/answers/{row.id}/hand-score", headers=_auth(admin_token),
                           json={"reference": "maria tumampos", "hypothesis": "maria luisa tumpos"})
        assert saved.status_code == 200
        assert (saved.json()["hand_s"], saved.json()["hand_i"], saved.json()["hand_words"]) == (1, 1, 2)
        wer = client.get("/api/eval/stats", headers=_auth(admin_token)).json()["wer"]
        assert (wer["errors"], wer["words"]) == (1, 3)                          # Whisper-only stays as it was
        assert (wer["hand_errors"], wer["hand_words"], wer["hand_scored"]) == (2, 2, 1)
        db.expire_all()
        _score(db, s, CARD1)                                                    # the late batch scores again
        db.expire_all()
        assert self._final(db, s, "surname").hand_words == 2
        cleared = client.delete(f"/api/eval/answers/{row.id}/hand-score", headers=_auth(admin_token))
        assert cleared.status_code == 404 or cleared.json()["hand_words"] is None

    def test_an_address_step_hand_score_adds_up_into_the_address(self, client, admin_token, db):
        from app.services import eval_service
        s = _card_session(db)
        card = eval_service.load_cards()["cards"][0]["answers"]["address"]["parts"]
        for key in eval_service.PARTS["address"]:
            _eval_try(db, s, key, "spoken", card[key]["read"], card[key]["expected"])
        _score(db, s, CARD1)
        purok = self._final(db, s, "address_purok")
        address = self._final(db, s, "address")
        assert client.put(f"/api/eval/answers/{address.id}/hand-score", headers=_auth(admin_token),
                          json={"reference": "x", "hypothesis": "y"}).status_code == 404   # score its steps
        client.put(f"/api/eval/answers/{purok.id}/hand-score", headers=_auth(admin_token),
                   json={"reference": "purok 2", "hypothesis": "purok 3"})
        db.expire_all()
        address = self._final(db, s, "address")
        assert (address.hand_s, address.hand_words) == (1, address.wer_words)
        client.delete(f"/api/eval/answers/{purok.id}/hand-score", headers=_auth(admin_token))
        db.expire_all()
        assert self._final(db, s, "address").hand_words is None

    def test_delete_evaluation_audio_now_also_erases_the_hand_texts(self, client, admin_token, db):
        s = _card_session(db)
        _eval_try(db, s, "surname", "spoken", "Abarquez", "Abarquez")
        _score(db, s, CARD1)
        row = self._final(db, s, "surname")
        client.put(f"/api/eval/answers/{row.id}/hand-score", headers=_auth(admin_token),
                   json={"reference": "abarquez", "hypothesis": "abarkes"})
        while client.post("/api/eval/delete-audio", json={"confirm": True, "size": 500},
                          headers=_auth(admin_token)).json()["remaining"]:
            pass
        db.expire_all()
        row = self._final(db, s, "surname")
        assert row.hand_reference is None and row.hand_hypothesis is None and row.hand_words == 1   # numbers stay


class TestStartup:
    def test_start_up_refuses_a_database_without_the_plan_009_migration(self):
        from sqlalchemy import create_engine as make_engine, text
        from app.core.database import MISSING_EVAL_CARDS_MIGRATION, missing_eval_message
        old = make_engine("sqlite://")
        with old.begin() as conn:
            conn.execute(text("create table app_settings (id integer primary key, evaluation_mode boolean)"))
            conn.execute(text("create table eval_sessions (id integer primary key)"))
            conn.execute(text("create table eval_answers (id integer primary key)"))
            conn.execute(text("create table security_events (id integer primary key)"))
        assert missing_eval_message(old) == MISSING_EVAL_CARDS_MIGRATION
