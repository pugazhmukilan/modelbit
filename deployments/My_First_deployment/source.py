import modelbit, sys
from typing import *

# main function
def first_regression(num):
    answer  = regressor.predict([[num]])
    
    return answer

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = first_regression(...)
#   print(result)