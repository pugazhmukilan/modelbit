import modelbit, sys
from typing import *
from sklearn.linear_model._base import LinearRegression

regressor1 = modelbit.load_value("data/regressor1.pkl") # LinearRegression()

# main function
def first_regression(num):
    answer  = regressor1.predict([[num]])
    
    return answer

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = first_regression(...)
#   print(result)