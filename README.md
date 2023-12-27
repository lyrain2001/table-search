# Table Search Using Large Language Models

This is the code for the project "Table2Doc and Query2Doc: Table Search Using Large Language Models" by Yurong Liu for course CS 6913: Web Search Engines (Fall 2023) taught by Prof. Torsten Suel.

### Environment
```
    pip install -r requirements.txt
```
### Description Generation
```
    python src/description.py
 ```   
### Query Expansion
```
    python src/query_expansion.py
```

### Wikitables Ranking
Ranking using descriptions:
```
   python src/description_rank.py [--expanded]
```

Ranking using tables:
```
   python src/table_rank.py [--expanded]
```