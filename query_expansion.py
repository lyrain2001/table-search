import tiktoken
import openai
import json


class QueryExpansion:
    def __init__(self, openai_key, query):
        self.query = query
        openai.api_key = (openai_key)
        
    def expand(self):
        expanded_queries = []
        for query in queries:
            expanded_query = self.expand_query(query)
            expanded_queries.append(expanded_query)

        with open('expanded_queries.txt', 'w') as f:
            for item in expanded_queries:
                f.write("%s\n" % item)
        
        
    def num_tokens_from_string(self, string, encoding_name="gpt-3.5-turbo"):
            encoding = tiktoken.encoding_for_model(encoding_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens


    def expand_query(self, context, model="gpt-3.5-turbo"):
            description = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {
                            "role": "system", 
                            "content": "You are an assistant for a dataset search engine.\
                                        Your goal is to increase the performance of this dataset search engine for keyword queries."},
                        {
                            "role": "user", 
                            "content": """Instruction:
    Answer the questions while using the input and context.
    The input includes dataset title, headers, a random sample, and profiler result of the large dataset.

    Input:
    """ + context + """
    Question:
    Describe the dataset covering the nine aspects above in one complete and coherent paragraph.

    Answer: """},
                    ],
                temperature=0.3)
            description_content = description.choices[0]['message']['content']
            return description_content
        
def main():
    def read_query(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = []
        for line in lines:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                data.append(parts[1].strip())  
        return data

    query_path = '../GTR/data/wikitables/queries.txt'
    queries = read_query(query_path)
    print(queries)
    
    OPENAI_API_KEY = "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"
    generator = QueryExpansion(OPENAI_API_KEY, queries)
    genertor.expand()