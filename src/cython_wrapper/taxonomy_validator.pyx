# cython: language_level=3

cdef enum TaxonomyValidationFlags:
    INVALID_CAPITALIZATION = 1
    CONTAINS_STRAIN_INFO = 2
    INVALID_CHARACTERS = 3

cdef const char * INVALID_CAPITALIZATION_STR = b"invalid_capitalization"
cdef const char * CONTAINS_STRAIN_INFO_STR = b"contains_strain_info"
cdef const char * INVALID_CHARACTERS_STR = b"invalid_characters"

cdef inline str cstr_to_str(const char * cmsg) except*:
    return (<bytes> cmsg).decode("utf-8")

cdef inline str get_flag_string(TaxonomyValidationFlags flag):
    if flag == INVALID_CAPITALIZATION:
        return cstr_to_str(INVALID_CAPITALIZATION_STR)
    elif flag == CONTAINS_STRAIN_INFO:
        return cstr_to_str(CONTAINS_STRAIN_INFO_STR)
    elif flag == INVALID_CHARACTERS:
        return cstr_to_str(INVALID_CHARACTERS_STR)
    return ""

cdef list strain_keywords = ["strain", "isolate", "clone", "var.", "ATCC", "DSM"]

cpdef list validate_taxonomy(name):
    cdef list flags = []
    cdef str stripped_name, lower_name
    cdef Py_UCS4 c
    cdef int i, n

    if name is None:
        raise ValueError("Taxonomy name must not be None")

    stripped_name = name.strip()
    if not stripped_name:
        raise ValueError("Taxonomy name must not be empty")

    if len(stripped_name) < 3 or " " not in stripped_name:
        return flags

    if len(name) < 2 or not name[0][0].isupper():
        return flags
    if not stripped_name[0].isupper() or " " not in stripped_name:
        flags.append(get_flag_string(INVALID_CAPITALIZATION))

    lower_name = stripped_name.lower()

    for kw in strain_keywords:
        if kw.lower() in lower_name:
            flags.append(get_flag_string(CONTAINS_STRAIN_INFO))
            break

    n = len(stripped_name)
    for i in range(n):
        c = stripped_name[i]
        if c.isdigit() or not (c.isalpha() or c in " -."):
            flags.append(get_flag_string(INVALID_CHARACTERS))
            break

    return flags

cpdef dict batch_validate(list names):
    cdef dict results = {}
    cdef str name
    for name in names:
        results[name] = validate_taxonomy(name)
    return results
