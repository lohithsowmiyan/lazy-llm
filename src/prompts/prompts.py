from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows,row

class Template():
    """
    
    """
    def __init__(self, prefix : str = None, suffix : str = None, **kwargs):
        self.prefix = prefix
        self.suffix = suffix

    def __repr__(self):
        return f'Template(prefix = {self.prefix}, suffix = {self.suffix})'

    #def _createPairs():
    def getTemplate(self, best : rows =  None, rest : rows =  None, current : row = None, cols = None):
        attr = ",".join([col.txt for col in cols])
        messages = [
        {"role": "system", "content": f"You are an excellent assistant, you need to evaluate the specifications of a configuration and answer in one word if the example falls into best or rest categories. Here are the attributes provided for each configuration in the same order: {attr}"},
        {"role": "user", "content": f"""These are the examples for best configuration: {best}, These are the examples of rest configuration: {rest}"""},
        {"role": "user", "content": f""" Based on the above examples attributes, answer in one word if the following example is similar to best or rest {current}"""}
        ]

        return messages

    
class Auto93Template(Template):
    """

    """
    def __init__(self, prefix : str = None, suffix : str = None):
        super().__init__(prefix, suffix)
        if not self.prefix:
            self.prefix = "You are an excellent car sales consultant, you need to evaluate the specifications of a car and answer in one word if the car falls into best or rest categories. Here are the attributes provided for each car in the same order: Number of Cylinders, Volume, Horsepower, Model, Origin."
 
        if not self.suffix:
            self.suffix = "Based on the above examples attributes, Striclty answer in one word if the following car is similar to  best cars or rest cars"

    def getZeroShot(self, best : rows =  None, rest : rows =  None) -> ChatPromptTemplate:
        """
            Returns a Zero Shot template
            ------------------------------------
            Params:
                best: contains the best rows of type rows(list[row]) after the current iterations  
                rest: contains the rest rows of type rows(list[row]) after the current iterations  
            -------------------------------------
            Returns:
                ChatPromptemplate with the updated best rest examples.
            -------------------------------------
            Raises:
                None
            -------------------------------------
        """
        examples = f"  These are the examples for best cars: {[b[:5] for b in best]}, These are the examples of rest cars: {[r[:5] for r in rest]}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prefix),
                ("human", examples),
                ("human", self.suffix + "{input}"),   
            ]
        )
        return prompt

    # def getFewShot(self, nshots : int) -> ChatPromptTemplate:
    #     """
    #         Returns a Few Shot template 
    #         ------------------------------------
    #         Params:
    #             best: contains the best rows of type rows(list[row]) after the current iterations  
    #             rest: contains the rest rows of type rows(list[row]) after the current iterations
    #             nshots: number of examples of for few shot learning
    #         -------------------------------------
    #         Returns:
    #             ChatPromptemplate with the updated best rest examples.
    #         -------------------------------------
    #         Raises:
    #             None
    #         -------------------------------------
    #     """

    def getTemplate(self, best : rows =  None, rest : rows =  None, current : row = None, **kwargs):

        messages = [
        {"role": "system", "content": " You are an excellent car sales consultant, you need to evaluate the specifications of a car and answer in one word if the car falls into best or rest categories. Here are the attributes provided for each car in the same order: Number of Cylinders, Volume, Horsepower, Model, Origin"},
        {"role": "user", "content": f"""These are the examples for best cars: {best}, These are the examples of rest cars: {rest}"""},
        {"role": "user", "content": f""" Based on the above examples attributes, answer in one word if the following car is similar to best cars or rest cars {current}"""}
        ]

        return messages


class HpoEasyTemplate(Template):
    """

    """
    def __init__(self, prefix : str = None, suffix : str = None):
        super().__init__(prefix, suffix)
        self.prefix = prefix
        self.suffix = suffix


    def getTemplate(self, best : rows =  None, rest : rows =  None, current : row = None, **kwargs):

        messages = [
        {"role": "system", "content": " You are an Machine Learning Expert, your task is to evaluate the configurations of a model and answer in one word if the configuration falls into best or rest categories. Here are the attributes provided for each configuration in the same order: N Estimators, criterion, Minimum Sample Leaves, Minimum Impurity Decrease, Max Depth"},
        {"role": "user", "content": f"""These are the examples for best configurations: {best}, These are the examples of rest configurations: {rest}"""},
        {"role": "user", "content": f""" Based on the above examples, answer in one word if the following configuration is similar to best or rest {current}"""}
        ]

        return messages


        


        
