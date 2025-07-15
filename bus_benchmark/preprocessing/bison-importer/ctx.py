"""
Barebones Creativyst Table Exchange parser.
Only implements the part of the specification needed to parse KV7/8 turbo messages.
Full specification: https://www.creativyst.com/Doc/Std/ctx/ctx.shtml
"""


def parse_ctx_message(ctx_message):
    lines = ctx_message.splitlines()

    parsed_message = {"meta": {}, "tables": []}
    current_table = {"meta": {}, "data": []}
    current_labels = None

    for line in lines:
        if line.startswith("\G"):
            parsed_message["meta"] = _parse_header(line)
        elif line.startswith("\T"):
            current_table = {"meta": _parse_table_header(line), "data": []}
            parsed_message["tables"].append(current_table)
        elif line.startswith("\L"):
            current_labels = _parse_labels(line)
        elif line and current_table and current_labels:
            current_table["data"].append(_parse_table_data(line, current_labels))

    return parsed_message


def _parse_header(line):
    """
    Parses global information.
    """
    fields = [_preprocess_field(field) for field in line[2:].split("|")]
    return {
        "label": _get_element(fields, 0),
        "name": _get_element(fields, 1),
        "comment": _get_element(fields, 2),
        "path": _get_element(fields, 3),
        "endian": _get_element(fields, 4),
        "enc": _get_element(fields, 5),
        "res1": _get_element(fields, 6),
        "res2": _get_element(fields, 7),
        "res3": _get_element(fields, 8),
    }


def _parse_table_header(line):
    """
    Parses table information.
    """
    fields = [_preprocess_field(field) for field in line[2:].split("|")]
    return {
        "label": _get_element(fields, 0),
        "name": _get_element(fields, 1),
        "comment": _get_element(fields, 2),
        "path": _get_element(fields, 3),
        "endian": _get_element(fields, 4),
        "enc": _get_element(fields, 5),
        "res1": _get_element(fields, 6),
        "res2": _get_element(fields, 7),
        "res3": _get_element(fields, 8),
    }


def _parse_labels(line):
    """
    Parse table column names.
    """
    return [_preprocess_field(field) for field in line[2:].split("|")]


def _parse_table_data(line, labels):
    """
    Parse table rows.
    """
    fields = [_preprocess_field(field) for field in line.split("|")]
    return {label: field for label, field in zip(labels, fields)}


def _get_element(lst, i):
    """
    Returns a list element or none if the index is out of bounds.
    """
    try:
        return lst[i]
    except IndexError:
        return None


def _preprocess_field(field):
    """
    Unescapes a field value.
    """
    if field == "\\0":
        field = None
    else:
        field = field.replace("\\r", "\r")
        field = field.replace("\\n", "\n")
        field = field.replace("\\i", "\\")
        field = field.replace("\\p", "|")
    return field
