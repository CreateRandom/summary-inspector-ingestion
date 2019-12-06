import pandas as pd
import odo

def store_dataframe_in_db(df, collection):
    odo.odo(df, collection)

def load_collection_into_frame(collection, dshape=None):
    return odo.odo(collection, pd.DataFrame, dshape=dshape)