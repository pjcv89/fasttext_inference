import fasttext 
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType
import pandas as pd

model = fasttext.load_model('models/ft_tuned.ftz')

def get_predictions(sentence,threshold=0.10,k=3):
    '''
    Function to get a list with predictions for a sentence, given a fastText model
    and given one of: a probability threshold, or a desired number k of top-k predictions.
    '''
    if threshold:
        labels, probs = model.predict(sentence.lower(),k=k)
        candidates = map(lambda y: y[0], filter(lambda x: x[1] >= threshold, zip(labels, probs)))
    else:
        candidates = model.predict(sentence.lower(),k=k)[0]
    output = list(map(lambda x: x.replace("__label__",""), candidates))
    if len(output)==0:
        return None
    else:
        return output[0] if k==1 else output
    
def make_udf(multi_prediction=False):
    if not multi_prediction:
        udf_predict = udf(lambda sentence: get_predictions(sentence,k=1),StringType())
    else:
        udf_predict = udf(lambda sentence: get_predictions(sentence,k=3),ArrayType(StringType()))
    return udf_predict

def candidates_fn(labels,probs,threshold=0.10):
    return list(map(lambda y: y[0].replace("__label__",""), filter(lambda x: x[1] >= threshold, zip(labels, probs))))

def predict_serie(serie_input,multi_prediction=False,rowwise=False):
    if not multi_prediction:
        if rowwise:
            serie_output = serie_input.apply(lambda sentence: get_predictions(sentence,k=1))
        else:
            labels, probs = model.predict(serie_input.tolist(),k=1)
            candidates = list(map(lambda lab, prob: candidates_fn(lab,prob), labels, probs))
            serie_output = pd.Series(list(map(lambda element: element[0] if len(element)>0 else None,candidates)))
    else:
        if rowwise:
            serie_output = serie_input.apply(lambda sentence: get_predictions(sentence,k=3))
        else:
            labels, probs = model.predict(serie_input.tolist(),k=3)
            candidates = list(map(lambda lab, prob: candidates_fn(lab,prob), labels, probs))
            serie_output = pd.Series(list(map(lambda element: element if len(element)>0 else None,candidates)))
    return serie_output
   
def make_pandas_udf(multi_prediction=False,rowwise=False): 
    if not multi_prediction:
        predict_serie_fn = lambda serie_input: predict_serie(serie_input, False, rowwise=rowwise)
        return pandas_udf(predict_serie_fn,returnType=StringType())
    else:
        predict_serie_fn = lambda serie_input: predict_serie(serie_input, True, rowwise=rowwise)
        return pandas_udf(predict_serie_fn,returnType=ArrayType(StringType()))