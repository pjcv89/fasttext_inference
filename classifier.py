import fasttext
import pandas as pd
from pyspark import SparkFiles
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType

modelPath = SparkFiles.get('ft_tuned.ftz')
model = fasttext.load_model(modelPath)


def get_predictions(sentence, threshold=0.10, k=3):
    """
    Get label prediction(s) for a sentence from a given fastText model.

    It returns at most a number of k labels whose probabilities are greater or equal
    than a threshold. If no labels met the criteria given a certain combination
    of k and threshold, a None value is returned.

    Parameters
    ----------
    sentence: str
            The sentence for which label predictions are retrieved.
    threshold: float
            Probability threshold.
    k:          int
            Maximum number of label predictions to retrieve.

    Returns
    -------
    output: str or list (with str elements)
            If k==1 a string with the unique label prediction is returned,
            otherwise a list with strings corresponding to label predictions is returned.

    Examples
    --------
    >>> get_predictions("python and sql",k=1,threshold=0.30)
    'python'
    >>> get_predictions("python and sql",k=3,threshold=0.20)
    ['python', 'sql']
    """
    labels, probs = model.predict(sentence.lower(), k=k)
    output = list(map(lambda y: y[0].replace("__label__", ""), filter(lambda x: x[1] >= threshold, zip(labels, probs))))
    if len(output) == 0:
        return None
    else:
        return output[0] if k == 1 else output


def make_udf(multi_prediction=False):
    """
    Function to make a Spark udf from the get_predictions function,
    specifying whether to retrieve single label prediction or
    multiple label predictions.

    :param multi_prediction: bool
        Whether to retrieve one single prediction or multiple predictions.
    :return: udf_predict: pyspark.sql.functions.udf
        Spark user defined function for the get_predictions function.

    Example
    -------
    >>> from classifier import make_udf
    >>> udf_predict = make_udf(multi_prediction=False)
    """
    if not multi_prediction:
        udf_predict = udf(lambda sentence: get_predictions(sentence, k=1), StringType())
    else:
        udf_predict = udf(lambda sentence: get_predictions(sentence, k=3), ArrayType(StringType()))
    return udf_predict


def candidates_fn(labels, probs, threshold=0.10):
    """
    Given labels predictions and probabilities that are output from a fastText model predict method call
    (for a single sentence), ang given a threshold, return those labels whose probabilities are greater or equal
    than the threshold.

    :param labels: list (with str elements)
    :param probs: numpy.ndarray (with float elements)
    :param threshold: float
    :return: list (with str elements)
    """
    return list(map(lambda y: y[0].replace("__label__", ""), filter(lambda x: x[1] >= threshold, zip(labels, probs))))


def predict_series(series_input, multi_prediction=False, rowwise=False):
    """
    Function to get label prediction(s) for sentences in pandas.Series objects,
    from a given fastText model.
    
    :param series_input: pandas.Series (with str elements)
    :param multi_prediction: bool
        Whether to retrieve one single prediction or multiple predictions.
    :param rowwise: bool
        Whether to get label predictions for a chunk of sentences
        using pandas.Series.apply method (i.e. "rowwise") or
        using fastText's own method (i.e. "native").
    :return: series_output: pandas.Series (with str elements, or with lists with str elements)

    Example
    >>> from classifier import predict_series
    >>> # If 'pdf' is a pandas.DataFrame and 'input' is the name of a column with str elements,
    >>> # then 'pdf.input' is a pandas.Series with str elements and we can call:
    >>> predict_series(pdf.input,True,False)
    """
    if not multi_prediction:
        if rowwise:
            series_output = series_input.apply(lambda sentence: get_predictions(sentence, k=1))
        else:
            labels, probs = model.predict(series_input.tolist(), k=1)
            candidates = list(map(lambda lab, prob: candidates_fn(lab, prob), labels, probs))
            series_output = pd.Series(list(map(lambda element: element[0] if len(element) > 0 else None, candidates)))
    else:
        if rowwise:
            series_output = series_input.apply(lambda sentence: get_predictions(sentence, k=3))
        else:
            labels, probs = model.predict(series_input.tolist(), k=3)
            candidates = list(map(lambda lab, prob: candidates_fn(lab, prob), labels, probs))
            series_output = pd.Series(list(map(lambda element: element if len(element) > 0 else None, candidates)))
    return series_output


def make_pandas_udf(multi_prediction=False, rowwise=False):
    """
    Function to make a Spark pandas udf from the predict_series function,
    specifying whether to retrieve single label prediction or
    multiple label predictions.

    :param multi_prediction: bool
        Whether to retrieve one single prediction or multiple predictions.
    :param rowwise: bool
        Whether to get label predictions for a chunk of sentences
        using pandas.Series.apply method (i.e. "rowwise") or
        using fastText's own method (i.e. "native").
    :return: pyspark.sql.functions.pandas_udf object
        Spark pandas user defined function for the predict_series function.

    Example
    -------
    >>> from classifier import make_pandas_udf
    >>> udf_predict = make_pandas_udf(multi_prediction=False,rowwise=True)
    """
    if not multi_prediction:
        return pandas_udf(lambda series_input: predict_series(series_input, False, rowwise=rowwise),
                          returnType=StringType())
    else:
        return pandas_udf(lambda series_input: predict_series(series_input, True, rowwise=rowwise),
                          returnType=ArrayType(StringType()))
