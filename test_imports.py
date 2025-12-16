print("Starting test script")
try:
    import pandas
    print("pandas imported")
    import numpy
    print("numpy imported")
    import sklearn
    print("sklearn imported")
    import nltk
    print("nltk imported")
    from dotenv import load_dotenv
    print("dotenv imported")
except Exception as e:
    print(f"Error: {e}")
print("Finished test script")
