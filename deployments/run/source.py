import modelbit, sys
from typing import *
from sklearn.linear_model._base import LinearRegression

regressor = modelbit.load_value("data/regressor.pkl") # LinearRegression()

# main function
def run(num):
    if num == None:
        return None
    else:
        return regressor.predict([num])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = run(...)
#   print(result)