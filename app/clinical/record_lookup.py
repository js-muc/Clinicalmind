def find_record_by_identifier(records, identifier):

    identifier = identifier.upper().strip()

    for record in records:

        fields = record.get("fields", {})

        entity = fields.get("entity", "").upper()

        if entity == identifier:
            return {
                "fields": fields,
                "source": record.get("source"),
                "page": record.get("page")
            }

    return None