class EligibilityRule:

    def __init__(self):
        self.degree_order = {
            "preUG": 0,
            "UG" : 1,
            "PG" : 2,
            "PhD" : 3
        }                            

    @staticmethod
    def is_degree_matching():
        pass