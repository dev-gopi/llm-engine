from training.reporting import archive_previous_report_files


def _args(tmp_path, *, resume=None):
    return tmp_path / "training.log", tmp_path / "training_report.json", resume


def test_new_stage_archives_old_log_and_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    log_file, report_json, resume = _args(tmp_path)
    log_file.write_text("old log", encoding="utf-8")
    report_json.write_text('{"old":true}', encoding="utf-8")

    archived = archive_previous_report_files(log_file, report_json, resume=bool(resume))

    assert len(archived) == 2
    assert not log_file.exists()
    assert not report_json.exists()
    assert {destination.read_text(encoding="utf-8") for _, destination in archived} == {
        "old log", '{"old":true}'
    }


def test_resume_keeps_existing_report_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    log_file, report_json, resume = _args(tmp_path, resume=tmp_path / "latest.pt")
    log_file.write_text("existing log", encoding="utf-8")
    report_json.write_text('{"existing":true}', encoding="utf-8")

    assert archive_previous_report_files(log_file, report_json, resume=bool(resume)) == []
    assert log_file.read_text(encoding="utf-8") == "existing log"
    assert report_json.read_text(encoding="utf-8") == '{"existing":true}'


def test_nonzero_rank_does_not_rotate_shared_report_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "1")
    log_file, report_json, resume = _args(tmp_path)
    log_file.write_text("rank-zero owns this", encoding="utf-8")

    assert archive_previous_report_files(log_file, report_json, resume=bool(resume)) == []
    assert log_file.read_text(encoding="utf-8") == "rank-zero owns this"
