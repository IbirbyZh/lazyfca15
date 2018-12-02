import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--all", action="store_true")

    return parser.parse_args()


def read_file(filename):
    lines = list()
    with open(filename) as i:
        i.readline()
        for line in map(lambda line: line.strip(), i):
            features, label = line.rsplit(",", 1)
            lines.append(dict(label=label == "positive", features=features.split(",")))
    return lines


class Object(object):
    def __init__(self, features, label=None):
        self.label = label
        self.features = frozenset({(i, value) for i, value in enumerate(features)})


class Context(object):
    def __init__(self, objects):
        self.objects = [obj.features for obj in objects]

    @property
    def size(self):
        return len(self.objects)

    def iter_over_intersection(self, object):
        for obj in self.objects:
            intersection = obj & object.features
            if intersection:
                yield intersection

    def count_objects_with(self, features):
        return sum(1 for obj in self.objects if obj.issuperset(features))


def read_contexts_and_test(train_file, test_file):
    train_objects = [Object(parsed_line["features"], parsed_line["label"]) for parsed_line in read_file(train_file)]
    test_objects = [Object(parsed_line["features"], parsed_line["label"]) for parsed_line in read_file(test_file)]
    positive_context = Context((obj for obj in train_objects if obj.label))
    negative_context = Context((obj for obj in train_objects if not obj.label))
    return positive_context, negative_context, test_objects


def make_support(obj, context, opposite_context):
    support = 0
    for intersection in context.iter_over_intersection(obj):
        support += context.count_objects_with(intersection)
    return float(support) / (context.size * context.size)


def make_confidence(obj, context, opposite_context):
    confidence = 0
    for intersection in context.iter_over_intersection(obj):
        confidence += opposite_context.count_objects_with(intersection)
    return float(confidence) / (opposite_context.size * context.size)


def make_intermediate_features(positive_context, negative_context, obj):
    intermediate_features = dict(label=obj.label)
    for label, context, opposite_context in [(0, negative_context, positive_context), (1, positive_context, negative_context)]:
        intermediate_features[label] = dict(
            support=make_support(obj, context, opposite_context),
            confidence=make_confidence(obj, context, opposite_context),
        )
    return intermediate_features


def make_all_intermediate_features(positive_context, negative_context, objects):
    all_intermediate_features = list()
    for obj in objects:
        all_intermediate_features.append(make_intermediate_features(positive_context, negative_context, obj))
    return all_intermediate_features


class Metrics(object):
    def __init__(self):
        self.positive_positive = 0
        self.positive_negative = 0
        self.negative_positive = 0
        self.negative_negative = 0
        self.contradictory = 0
        self.count = 0

    def add(self, label, is_positive, is_negative):
        self.count += 1
        if is_positive == is_negative:
            self.contradictory += 1
            return
        if label:
            if is_positive:
                self.positive_positive += 1
            else:
                self.positive_negative += 1
        else:
            if is_positive:
                self.negative_positive += 1
            else:
                self.negative_negative += 1

    def merge(self, metric):
        self.positive_positive += metric.positive_positive
        self.positive_negative += metric.positive_negative
        self.negative_positive += metric.negative_positive
        self.negative_negative += metric.negative_negative
        self.contradictory += metric.contradictory
        self.count += metric.count

    @property
    def accuracy(self):
        return float(self.positive_positive + self.negative_negative) / self.count

    def __repr__(self):
        args = (self.contradictory,
                self.positive_positive,
                self.positive_negative,
                self.negative_positive,
                self.negative_negative,)
        return ("Metrics\n"
                "contradictory={:.3f}\n"
                "positive_positive={:.3f}\n"
                "positive_negative={:.3f}\n"
                "negative_positive={:.3f}\n"
                "negative_negative={:.3f}").format(
            *(float(arg) / self.count for arg in args)
        )


def calc_metrics(all_intermediate_features, function):
    metrics = Metrics()
    for intermediate_features in all_intermediate_features:
        metrics.add(intermediate_features["label"], function(intermediate_features[1]), function(intermediate_features[0]))
    return metrics


class ThresholdFunction(object):
    def __init__(self, support=None, confidence=None):
        self.support = support
        self.confidence = confidence

        if self.support is not None:
            self.support = abs(self.support)
            self.greater_then_support = support > 0
        if self.confidence is not None:
            self.confidence = abs(confidence)
            self.greater_then_confidence = confidence > 0

    def __call__(self, intermediate_features):
        if (self.support is not None and
                (
                            (self.greater_then_support and self.support < intermediate_features["support"]) or
                            (not self.greater_then_support and self.support >= intermediate_features["support"])
                )

            ):
            return False
        if (self.confidence is not None and
                (
                            (self.greater_then_confidence and self.confidence < intermediate_features["confidence"]) or
                            (not self.greater_then_confidence and self.confidence >= intermediate_features["confidence"])
                )

            ):
            return False
        return True

    def __repr__(self):
        result = ["ThresholdFunction"]
        if self.support is not None:
            support = self.support
            if not self.greater_then_support:
                support *= -1
            result.append("support={}".format(support))
        if self.confidence is not None:
            confidence = self.confidence
            if not self.greater_then_confidence:
                confidence *= -1
            result.append("confidence={}".format(confidence))
        return " ".join(result)


def one_file_main(train_file, test_file, functions):
    positive_context, negative_context, test_objects = read_contexts_and_test(train_file, test_file)
    all_intermediate_features = make_all_intermediate_features(positive_context, negative_context, test_objects)
    results = list()
    for function in functions:
        results.append(calc_metrics(all_intermediate_features, function))
    return results


def main():
    args = parse_args()

    if args.all:
        indexes = range(1, 11)
    else:
        indexes = [1]

    files = []
    for index in indexes:
        train_file = os.path.join(args.data_dir, "train{}.csv".format(index))
        test_file = os.path.join(args.data_dir, "test{}.csv".format(index))
        files.append((train_file, test_file))

    functions = list()
    r = [-.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.12, -.11, -.1, None, .1, .11, .12, .2, .3, .4, .5, .6, .7, .8, .9]
    r_2 = [abs(v) * v if v is not None else v for v in r]
    r_3 = [abs(v * v) * v if v is not None else v for v in r]
    for r_array in [r, r_2, r_3]:
        for confidence in r_array:
            for support in r_array:
                functions.append(ThresholdFunction(support=support, confidence=confidence))
    metrics = [Metrics() for _ in functions]

    for train_file, test_file in files:
        results = one_file_main(train_file, test_file, functions)
        for metric, result in zip(metrics, results):
            metric.merge(result)

    for function, metric in sorted(zip(functions, metrics), key=lambda args: args[1].accuracy):
        print function
        print metric


if __name__ == "__main__":
    main()
