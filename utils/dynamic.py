import sys
import os
import six
import importlib

import yaml


def import_string(dotted_path: str):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    if dotted_path is None:
        return None
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])


def import_evaluate(module_name: str):
    """
    Different from import_string (only imports the module), this method imports the module and returns not only the
    module, but also the additional params required by the module

    The reason to design it this way is, sklearn module's interface is not designed by us
    Some of the method may require different params other than just y_ground truth and y_pred
    We need to use if-else to get the additional params manually

    :param module_name: string, name of the module, not the actual module name, like it can be f1_score_micro,
                           the function then looks up the config file and get the actual module sklearn.metric.f1_score
    :return: evaluate method (class/function),
             additional params (dict, if needed; None, if not needed; should be loaded from external config files)
             inputs required by the metric, can be
             y_gt: ground truth label
             y_pred: predicted label
             y_pred_scores: scores for all the labels in each test point
    """
    if module_name is None:
        return None

    # load sklearn metric config
    with open('./config/metric.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # raise errors if the module is not in the config file
    if module_name not in cfg:
        raise NotImplementedError("{} not found in config.metric.yaml".format(module_name))

    actual_module_name = cfg[module_name]["module"]

    try:
        module_path, class_name = actual_module_name.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % actual_module_name
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    module = importlib.import_module(module_path)

    evaluate_class = None
    additional_params = cfg[module_name]["params"]

    try:
        evaluate_class = getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    return evaluate_class, additional_params, cfg[module_name]["inputs"]
