import modelbit, sys
from typing import *
from sklearn.linear_model._base import LinearRegression

regressor = modelbit.load_value("data/regressor.pkl") # LinearRegression()

# main function
def run(num1,num2,num3,num4,num5,num6):
    if num1 == None:
        return None
    else:
        return regressor.predict([num1,num2,num3,num4,num5,num6])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = run(...)
#   print(result)