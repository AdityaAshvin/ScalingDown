import re


def extract_answer(text):
    """
    Extracts the answer from the generated text.
    If "Final answer:" is present, extracts the answer following that phrase.
    Otherwise, extracts the first occurrence of an answer option (A-E).
    """
    answer_pattern = r'([A-E])'
    final_answer_pattern = r'answer:\s*([A-E])'

    final_answer_match = re.search(final_answer_pattern, text, re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).upper()
    else:
        answer_match = re.search(answer_pattern, text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
    return ""



def extract_rationale(text):
    # Case 1: Extract rationale before "The final answer:"
    final_answer_pos = text.lower().find(". The final")
    answer_pos = text.lower().find("answer:")
    equals_pos = text.find("=")
    if answer_pos != -1:
        rationale = text[equals_pos:final_answer_pos].strip() if final_answer_pos != -1 else text[equals_pos:final_answer_pos].strip()
        return rationale

    # Case 2: Extract rationale after '='
    if equals_pos != -1:
        rationale = text[equals_pos + 1:].strip()
        return rationale

    # Case 3: No clear rationale found
    return ""

