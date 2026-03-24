import difflib

END_PHRASES = [
    "bye",
    "goodbye",
    "ok bye",
    "okay bye",
    "bye bye",
    "talk to you later",
    "see you later",
    "that's all",
    "that is all",
    "nothing else",
    "i'm done",
    "im done"
]
def is_end_phrase(text: str) -> bool:
    text = text.lower()
    return any(p in text for p in END_PHRASES)



def should_end_call(text: str) -> bool:
    text_lower = text.lower().strip()

    # strong phrases only
    for phrase in END_PHRASES:
        if phrase in text_lower:
            return True

    # fuzzy detection ONLY for "bye"
    words = text_lower.split()
    for w in words:
        if difflib.get_close_matches(w, ["bye"], cutoff=0.8):
            return True

    return False