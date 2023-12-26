import tiktoken
import openai
import json


class QueryExpansion:
    def __init__(self, openai_key, query):
        self.query = query
        openai.api_key = (openai_key)
        
    def expand(self):
        expanded_queries = []
        for query in self.query:
            expanded_query = self.expand_query(query)
            expanded_query = query + " [SEP] " + expanded_query
            expanded_queries.append(expanded_query)

        with open('expanded_queries.txt', 'w') as f:
            for idx in range(len(expanded_queries)):
                f.write("%s %s\n" % (idx, expanded_queries[idx]))

        
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
                            "content": "You are an assistant for expanding queries.\
                                        Your goal is to provide illustrative details to the following queries to help dataset search."},
                        {
                            "role": "user", 
                            "content": """Instruction:
    Provide illustrative details to the following query in a single paragraph:
    Query: demographics of Hungary
    Answer:  Demographics of Hungary refer to statistical data about the population composition and characteristics of Hungary as a whole and its various regions. This information includes factors such as the total population, population growth or decline, age distribution, gender ratio, ethnic composition, educational attainment, employment rates, and more. It also encompasses trends in population changes over time, migration patterns, and urbanization levels within the country. 

    Query: """ + context + """
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
    
    OPENAI_API_KEY = "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"
    generator = QueryExpansion(OPENAI_API_KEY, queries)
    generator.expand()
    
if __name__ == "__main__":
    main()