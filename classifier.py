import fasttext 
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType

model = fasttext.load_model('models/ft_tuned.ftz')

def get_predictions(sentence,threshold=None,k=3):
    '''
    Function to get a list with predictions for a sentence, given a fastText model
    and given one of: a probability threshold, or a desired number k of top-k predictions.
    '''
    if threshold:
        labels, probs = model.predict(sentence.lower(),k=k)
        candidates = [labels[index]  for index in range(0,k) if probs[index] >= threshold]
    else:
        candidates = model.predict(sentence.lower(),k=k)[0]
    output = [candidate.replace("__label__","") for candidate in candidates]
    return output

def make_udf(multi_prediction=False):
    if not multi_prediction:
        udf_predict = udf(lambda sentence: get_predictions(sentence,k=1)[0])
    else:
        udf_predict = udf(lambda sentence: get_predictions(sentence,k=3))
    return udf_predict

def predict_serie(serie_input,multi_prediction=False):
    if not multi_prediction:
        serie_output = serie_input.apply(lambda sentence: get_predictions(sentence,k=1)[0])
    else:
        serie_output = serie_input.apply(lambda sentence: get_predictions(sentence,k=3))
    return serie_output

def make_pandas_udf(multi_prediction=False): 
    if not multi_prediction:
        predict_serie_fn = lambda serie_input: predict_serie(serie_input, False)
        return pandas_udf(predict_serie_fn,returnType=StringType())
    else:
        predict_serie_fn = lambda serie_input: predict_serie(serie_input, True)
        return pandas_udf(predict_serie_fn,returnType=ArrayType(StringType()))