from scripts.prepare_smoltalk_chat import clean_record


def record(subset="everyday-conversations", assistant="Hello! How can I help?"):
    return {
        "source_subset": subset,
        "messages": [
            {"role": "system", "content": "You are somebody else."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_keeps_only_clean_everyday_conversations_and_normalizes_identity():
    cleaned = clean_record(record())

    assert cleaned is not None
    assert cleaned["source"] == "smoltalk_everyday_conversations"
    assert cleaned["messages"][0]["content"].startswith("You are Gopi")
    assert clean_record(record(subset="numina-cot-100k")) is None
    assert clean_record(record(assistant="I am Open Assistant.")) is None
