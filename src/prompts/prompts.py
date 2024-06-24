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
    def __init__(self, system_message : str = None, human_message : str = None):
        super().__init__(system_message, human_message)
        if not self.system_message:
            self.system_message = "As an excellent car sales consultant, you are provided with the specifications of two cars. Your task is to analyze the given attributes and recommend which car is better. Here are the attributes for each car provided in the same order in a list: Number of Clndrs (Number of cylinders), Volume (Volume of the car), HpX (Horsepower of the car), Model (Model of the car), origin (Origin of the car)."

        if not self.human_message:
            self.human_message = """
            Based on these attributes, answer in one word if the following car falls into best or rest or an outlier.
            """

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
        examples = f"""
          Best: {[b[:5] for b in best]}
          Rest: {[r[:5] for r in rest]}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                ("human", examples),
                ("human", "{input}"),
                ("human", self.human_message)
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


        
