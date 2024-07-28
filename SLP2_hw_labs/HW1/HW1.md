# Task
Use pretrained word embeddings to build a recommender system that makes
suggestions based on a user query.

**Dataset**: <br> 
ted_main.csv <br>
**Logic**: <br>
- Description vector: Create a single vector for each entry in the description column in ted_main.csv.
- Query vector: Create a single vector for the user query, which can contain multiple words.
- Find the description vectors that are most similar to the query vector, and recommend those talks to the user.