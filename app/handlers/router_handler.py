from routing.classifier import classify_query

from handlers.lookup_handler import handle_lookup

from handlers.reasoning_handler import handle_reasoning

from clinical.aggregation_engine import handle_aggregation

from clinical.numeric_engine import handle_numeric_reasoning


def route_query(question, model):

    query_type = classify_query(question)

    print("\n=== QUERY TYPE ===")
    print(query_type)

    if query_type == "lookup":
        return handle_lookup(question)

    if query_type == "reasoning":
        return handle_reasoning(
            question,
            model
        )

    if query_type == "aggregation":
        return handle_aggregation(
            question
        )

    if query_type == "numeric":
        return handle_numeric_reasoning(
            question,
            model
        )

    return "Query type not yet implemented."