import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process the documents to add 'answer' field.
    """
    def _process_doc(doc):
        # Convert numerical answer to letter index (A, B, C, D)
        if doc.get("answer_idx", -1) in [0, 1, 2, 3]:
            doc["answer"] = ["A", "B", "C", "D"][doc["answer_idx"]]
        return doc

    return dataset.map(_process_doc)