QA_PROMPT = """Answer the question by thinking step by step.
                       Question: {question}
                       Your output MUST follow the following format:
                       ```json 
                       {{"reason": "your reasoning steps","answer": "your final answer in ONE OR FEW WORDS"}}
                       ```
                       Here are some examples:
                        # 
                        Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt’s father-in-law?
                        Expected Output:
                        ```json
                        {{"reason": "The husband of Catherine of Pomerania, Countess Palatine of Neumarkt is John, Count Palatine of
                        Neumarkt. The father of John, Count Palatine of Neumarkt is Rupert III of the Palatinate. So the final answer is: Rupert III of the Palatinate.",
                        "answer": "Rupert III of the Palatinate"}}
                        ```
                        #
                        Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
                        Expected Output:
                        ```json
                        {{"reason":"The director of Crimen a las tres is Luis Saslavsky.The director of The Working Class Goes to Heaven is Elio Petri. Luis Saslavsky died on March 20, 1995.Elio Petri died on 10 November 1982. So the director of The Working Class Goes to Heaven died first.",
                        "answer":"The Working Class Goes to Heaven"}}
                        ```
                        #
                        Question: Which genus of moth in the world's seventh-largest country contains only one species?
                        Expected Output:
                        ```json
                        {{"reason": "The world's seventh-largest country is India. Indogrammodes contains only one species and is found in India, and the genus of Indogrammodes is Crambidae.So the final answer is: Crambidae.",
                        "answer": "Crambidae"}}
                        ```
                        # 
                        Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
                        Expected Output:
                        ```json
                        {{"reason": "The track where the 2013 Liqui Moly Bathurst 12 Hour was staged is Mount Panorama Circuit. And the length of Mount Panorama Circuit is 6.213 km long. So the final answer is: 6.213 km long.",
                        "answer": "6.213 km long"}}
                        ```
"""
