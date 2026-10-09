"""
Barebones Creativyst Table Exchange parser.
Only implements the part of the specification needed to parse KV7/8 turbo messages.
Full specification: https://www.creativyst.com/Doc/Std/ctx/ctx.shtml
"""

import re
from collections import Counter

# the escape sequences KV7/8 turbo allows within a field, \0 stands for a whole field
ESCAPE_SEQUENCES = re.compile(r"\\[rnip]")


def parse_ctx_message(ctx_message):
    """
    Parses a message into its tables. Rows that do not match their table's labels or
    contain an escape sequence KV7/8 turbo does not allow are left out and counted in
    "rejected" by table and reason.
    """
    lines = ctx_message.splitlines()

    parsed_message = {"meta": {}, "tables": [], "rejected": Counter()}
    current_table = None
    current_labels = None

    for line in lines:
        if line.startswith(r"\G"):
            parsed_message["meta"] = _parse_header(line)
        elif line.startswith(r"\T"):
            current_table = {"meta": _parse_table_header(line), "data": []}
            parsed_message["tables"].append(current_table)
            # every table has its own labels, never apply the previous table's
            current_labels = None
        elif line.startswith(r"\L"):
            current_labels = _parse_labels(line)
        elif line and current_table:
            fields = line.split("|")
            reason = _rejection_reason(fields, current_labels)
            if reason:
                parsed_message["rejected"][(current_table["meta"]["name"], reason)] += 1
            else:
                current_table["data"].append(_parse_table_data(fields, current_labels))

    return parsed_message


def _rejection_reason(fields, labels):
    """
    Returns why a table row cannot be parsed, or None if it can.
    """
    if labels is None:
        return "table has no labels"
    if len(fields) != len(labels):
        return "field count does not match labels"
    for field in fields:
        if field != "\\0" and "\\" in ESCAPE_SEQUENCES.sub("", field):
            return "invalid escape sequence"
    return None


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


def _parse_table_data(fields, labels):
    """
    Parse table rows.
    """
    return {label: _preprocess_field(field) for label, field in zip(labels, fields)}


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
        field = field.replace("\\p", "|")
        field = field.replace("\\i", "\\")
    return field
