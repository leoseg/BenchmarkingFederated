def evaluate_metrics_aggregation_fn(results):
    """
    Aggregates metrics of all clients by averaging their metrisc weighted by the num of examples they processed
    :param results: list of tuples in form num examples and metrics
    :return: dictionary with summarized metrics
    """

    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    total_metrics = {}
    for num_examples, metrics in results:
        metrics.update((x, y * num_examples) for x, y in metrics.items())
        for key,value in metrics.items():
            value_before = total_metrics.setdefault(key,0)
            total_metrics[key] = value_before + value
    total_metrics.update((x,y/num_total_evaluation_examples) for x,y in total_metrics.items())
    return total_metrics




