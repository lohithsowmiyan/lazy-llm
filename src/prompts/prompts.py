from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

class Template():
    """
    
    """
    def __init__(self, system_message : str = None, human_message : str = None):
        self.system_message = system_message
        self.human_message = human_message

    def _createPairs():
    
class Auto93Template(Template):
    """

    """
    def __init__(self, system_message : str = None, human_message : str = None):
        super().__init__(system_message, human_message)
        if not self.system_message:
            self.system_message = """
            As an excellent car sales consultant, you are provided with the specifications of two cars. Your task is to analyze the given attributes and recommend which car is better. Here are the attributes for each car:
            - Number of cylinders
            - Volume
            - Horsepower
            - Model
            - Origin
            """

        if not self.human_message:
            self.human_message = """
            Based on these attributes, suggest which car is better and why
            """

    def getZeroShot() -> ChatPromptTemplate:

    def getFewShot(self, nshots : int) -> ChatPromptTemplate:

        
