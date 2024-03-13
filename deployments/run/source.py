import modelbit, sys
from typing import *

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