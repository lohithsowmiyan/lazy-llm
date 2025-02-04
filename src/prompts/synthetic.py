from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows,row,data, cols, NUM, SYM, correlation

def rows_to_markdown(table:data):
    """
    Converts a 2D list (table) to Markdown table format.

    :param table: List of lists where each sublist represents a row in the table.
    :return: String representing the table in Markdown format.
    """
    if not table:
        return ""

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
    header = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(table[0])) + " |"
    separator = "|-" + "-|-".join("-" * width for width in col_widths) + "-|"
    rows = [
        "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        for row in table[1:]
    ]
    markdown_table = "\n".join([header, separator] + rows)
    
    return markdown_table

class SYNTHETIC():

    def __init__(self, i:data, best: rows, rest:rows):
        self.meta = {
        NUM : 'NUMBER',
        SYM : 'SYMBOL'
        }
        self.i = i
        self.dff = 0

        if(len(i.cols.names) != len(i.cols.x) + len(i.cols.y)):
            self.dff = len(i.cols.names)  - len(i.cols.x) - len(i.cols.y)

        self.best = [b[:len(i.cols.x) + self.dff] for b in best]
        self.rest = [r[:len(i.cols.x) + self.dff] for r in rest]

        self.best_wy = best
        self.rest_wy = rest

        self.col_names = i.cols.names
    
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

    def get_template_markdown(self):

        table = [[col for col in self.i.cols.names[:self.x + self.dff]] + ['Class']] + [b + ['Best'] for b in self.best] + [r + ['Rest'] for r in self.rest]

        headers = ['S.No', 'Name', 'Type', 'Lowest', 'Highest', 'Unique']

        new_Features = [headers] + [
        [col.at+1 , col.txt, self.meta[col.this], col.lo, col.hi, 'N/A']
        if col.this == NUM 
        else [col.at+1, col.txt, self.meta[col.this], 'N/A', 'N/A', col.has]
        for col in self.i.cols.x
        ]

        messages = [
        ("system" , f'''
        You are given a dataset with several features. The rows has been categorized into "best" and "rest" examples based on their overall performance. Below are the key features and their descriptions from the dataset:
        ...
        {rows_to_markdown(new_Features)}
        '''),
        ("human",  f'''
        **Given Examples:**
        {rows_to_markdown(table)}
       '''
        ),
        ("human",  f'''
        **Task:**
        1. **Generate Two New Examples that are Better**: These should outperform the given "Best" examples by optimizing the relevant features to better combinations.
        2. **Generate Two New Examples that are Poorer**: These should underperform the given "Rest" examples by modifying the relevant features to worse combinations.

        Consider the interdependencies between features, and ensure that the generated examples follow logical consistency within the dataset's context.
        **Return the output in the same mardown structure: !Just the output table alone and Don't add the input rows to the table**
        '''
        )]

        return messages


    def get_template_correlation(self):

        table = [[col for col in self.col_names] + ['Class']] + [b + ['Best'] for b in self.best_wy] + [r + ['Rest'] for r in self.rest_wy]

        headers = ['S.No', 'Name', 'Type', 'Lowest', 'Highest', 'Unique']

        new_Features = [headers] + [
        [col.at+1 , col.txt, self.meta[col.this], col.lo, col.hi, 'N/A']
        if col.this == NUM 
        else [col.at+1, col.txt, self.meta[col.this], 'N/A', 'N/A', col.has]
        for col in self.i.cols.x
        ] 

        y_meta  = [['S.No', 'Name', 'Objective']] + [
            [col.at+1, col.txt, 'Lower is Better' if col.txt[-1] == '-' else 'Higher is Better']
            for col in self.i.cols.y
        ]



        corr = []
        for col1 in self.i.cols.x:
            temp = [col1.txt]
            for col2 in self.i.cols.x:
                temp.append(correlation(self.i, col1, col2))
            corr.append(temp)

        corr = [['Names'] + table[0][:-2]] +  corr

        

        
        messages = [
        ("system" , f'''
        You are given a dataset with several features. The rows has been categorized into "best" and "rest" examples based on their overall performance along with the Dependent variable that needs to either maximized or minimized. Below are the key features and their descriptions from the dataset:
        ...
        Description of Independent Variables
        {rows_to_markdown(new_Features)}

        Description of Dependent Variables
        {rows_to_markdown(y_meta)}

        Correlation between Features:
        {rows_to_markdown(corr)}
               
        '''),
        ("human",  f'''
        **Given Examples:**
        {rows_to_markdown(table)}
       '''
        ),
        ("human",  f'''
        **Task:**
        1. **Generate a New Example that is Better**: These should outperform the given "Best" examples by optimizing the relevant features to better combinations.
        2. **Generate a New Examples that is Poorer**: These should underperform the given "Rest" examples by modifying the relevant features to worse combinations.

        Consider the interdependencies between features, and ensure that the generated examples follow logical consistency within the dataset's context.
        **Return the output in the same mardown structure: !Just the output table alone and Don't add the input rows to the table**
        '''
        )]

        return messages





