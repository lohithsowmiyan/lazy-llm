from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows,row,data, cols, NUM, SYM

class SYNTHETIC():

    def __init__(self, i:data, best: rows, rest:rows):
        self.meta = {
        NUM : 'NUMBER',
        SYM : 'SYMBOL'
        }
        self.i = i
        self.best = best
        self.rest = rest
        self.Features = [
        f'Feature {col.at + 1} : (name : {col.txt}, type : {self.meta[col.this]}, range : [{col.lo} - {col.hi}])'
        if col.this == NUM
        else f'Feature {col.at + 1} : (name : {col.txt}, type : {self.meta[col.this]})'
        for col in i.cols.x
        ]
        self.x = len(i.cols.x)

    def get_template(self):

        messages = [
        {"role": "system", "content": f'''
        You are given a dataset with several features. The rows has been categorized into "best" and "rest" examples based on their overall performance. Below are the key features and their descriptions from the dataset:
        ...
        {self.Features}
        '''},
        {"role": "user", "content": f'''
        **Given Examples:**

        1. **Best Example 1**: {self.best[0][:self.x]}
        2. **Best Example 2**: {self.best[0][:self.x]}
        3. **Rest Example 1**: {self.rest[1][:self.x]}
        4. **Rest Example 2**: {self.rest[1][:self.x]}
       '''
        },
        {"role": "user", "content": f'''
        **Task:**
        1. **Generate Two New Examples that are Better**: These should outperform the given "Best" examples by optimizing the relevant features to better combinations.
        2. **Generate Two New Examples that are Poorer**: These should underperform the given "Rest" examples by modifying the relevant features to worse combinations.

        Consider the interdependencies between features, and ensure that the generated examples follow logical consistency within the dataset's context.
        **Return the output as a JSON object with the following structure:**
        ```json
        {{
            "better_examples" : [
                {{
                    "features" : [],
                    "explanation" : "...",
                }},
                {{
                    "features" : [],
                    "explanation" : "...",
                }}   
            ],
            "poorer_examples" : [
                {{
                    "features" : [],
                    "explanation" : "...",
                }},
                {{
                    "features" : [],
                    "explanation" : "...",
                }}   
            ]
        }}
        '''
        }]
        

        return messages

    def get_langchain_template(self):

        

        
        messages = [
        ("system" , f'''
        You are given a dataset with several features. The rows has been categorized into "best" and "rest" examples based on their overall performance. Below are the key features and their descriptions from the dataset:
        ...
        {self.Features}
        '''),
        ("human",  f'''
        **Given Examples:**

        1. **Best Example 1**: {self.best[0][:self.x]}
        2. **Best Example 2**: {self.best[0][:self.x]}
        3. **Rest Example 1**: {self.rest[1][:self.x]}
        4. **Rest Example 2**: {self.rest[1][:self.x]}
       '''
        ),
        ("human",  f'''
        **Task:**
        1. **Generate Two New Examples that are Better**: These should outperform the given "Best" examples by optimizing the relevant features to better combinations.
        2. **Generate Two New Examples that are Poorer**: These should underperform the given "Rest" examples by modifying the relevant features to worse combinations.

        Consider the interdependencies between features, and ensure that the generated examples follow logical consistency within the dataset's context.
        **Return the output as a JSON object with the following structure:**
        ```json
        {{
            "better_examples" : [
                {{
                    "features" : [],
                    "explanation" : "...",
                }},
                {{
                    "features" : [],
                    "explanation" : "...",
                }}   
            ],
            "poorer_examples" : [
                {{
                    "features" : [],
                    "explanation" : "...",
                }},
                {{
                    "features" : [],
                    "explanation" : "...",
                }}   
            ]
        }}
        '''
        )]
        

        return messages






