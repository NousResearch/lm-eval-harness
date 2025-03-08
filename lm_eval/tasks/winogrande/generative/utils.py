import datasets

def doc_to_text(doc):
    idx = doc["sentence"].index("_")
    return f"{doc['sentence'][:idx]}_\n1. {doc['option1']}\n2. {doc['option2']}"

def doc_to_target(doc):
    # Convert answer to integer index (0 or 1)
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]