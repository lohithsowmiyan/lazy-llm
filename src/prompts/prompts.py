from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows

class Template():
    """
    
    """
    def __init__(self, system_message : str = None, human_message : str = None, **kwargs):
        self.system_message = system_message
        self.human_message = human_message

    def __repr__(self):
        return f'Template(system_message = {self.system_message}, human_message = {self.human_message})'

    #def _createPairs():
    
class Auto93Template(Template):
    """

    """
    def __init__(self, prefix : str = None, suffix : str = None):
        super().__init__(system_message, human_message)
        if not self.system_message:
            self.prefix = "You are an excellent car sales consultant, you need to evaluate the specifications of a car and answer in one word if the car falls into best or rest categories. Here are the attributes provided for each car in the same order: Number of Cylinders, Volume, Horsepower, Model, Origin."

        if not self.human_message:
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


        
