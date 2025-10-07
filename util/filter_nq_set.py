def clean_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x['start_token'])

    cleaned_intervals = []
    current_interval = intervals[0]

    for interval in intervals[1:]:
        if (interval['start_token'] >= current_interval['start_token'] and
                interval['end_token'] <= current_interval['end_token']):
            continue
        else:
            cleaned_intervals.append(current_interval)
            current_interval = interval

    cleaned_intervals.append(current_interval)

    return cleaned_intervals


def filter_nq_set_dev(dataset):
    return_dataset = []
    for item in dataset:
        new_annotations = []
        for annotation in item["annotations"]:
            if annotation["long_answer"]["start_token"] != -1 and annotation["long_answer"]["end_token"] != -1:
                flag = True
                for annotation_candidate in new_annotations:
                    if annotation_candidate["long_answer"]["start_token"] == annotation["long_answer"][
                        "start_token"] and annotation_candidate["long_answer"]["end_token"] == \
                            annotation["long_answer"]["end_token"]:
                        flag = False
                        break
                if flag:
                    new_annotations.append(annotation)
        if len(new_annotations) > 0:
            item["annotations"] = new_annotations
            item["long_answer_candidates"] = clean_intervals(item["long_answer_candidates"])
            return_dataset.append(item)
    return return_dataset
